import os
import logging
import time
import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import label, distance_transform_edt
from dataloader.Maxillo import Maxillo
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator


eps = 1e-10
class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json'
        if self.debug:
            filename = 'splits.json.small'

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        # in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        in_ch = 2
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)

        self.scheduler = SchedulerFactory(
                sched_name,
                self.optimizer,
                milestones=sched_milestones,
                gamma=sched_gamma,
                mode='max',
                verbose=True,
                patience=15
            ).get()

        # load loss
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='train',
                transform=tio.Compose([
                    tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                    self.config.data_loader.preprocessing,
                    self.config.data_loader.augmentations,
                    ]),
                # dist_map=['sparse','dense']
                # config_line=self.config.data_loader.line_select
        )
        self.val_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='val',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
                # config_line=self.config.data_loader.line_select
        )
        self.test_dataset = Maxillo(
                root=self.config.data_loader.dataset,
                filename=filename,
                splits='test',
                transform=self.config.data_loader.preprocessing,
                # dist_map=['sparse', 'dense']
                # config_line=self.config.data_loader.line_select
        )

        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)

        if self.config.trainer.reload:
            self.load()



    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.config.title,
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)



    def load(self):
        path = self.config.trainer.checkpoint
        logging.info(f'Loading checkpoint from {path}')
        state = torch.load(path)

        if 'title' in state.keys():
            self_title_header = self.config.title[:-11]
            load_title_header = state['title'][:-11]
            if self_title_header == load_title_header:
                self.config.title = state['title']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['state_dict'])
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def extract_data_from_patch(self, patch):
        volume = patch['data'][tio.DATA].float().cuda()
        gt = patch['dense'][tio.DATA].float().cuda()
        sparse = patch['sparse'][tio.DATA].float().cuda()
        images = torch.cat([volume, sparse], dim=1)

        emb_codes = torch.cat((
            patch[tio.LOCATION][:,:3],
            patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        return images, gt, emb_codes



    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader
        if self.config.data_loader.training_set == 'generated':
            logging.info('using the generated dataset')
            data_loader = self.synthetic_loader

        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt, emb_codes = self.extract_data_from_patch(d)

            partition_weights = 1
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
            if torch.sum(gt_count) == 0: continue
            partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            preds = self.model(images, emb_codes)

            assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
            loss = self.loss(preds, gt, partition_weights)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)            
            
            self.optimizer.step()

            preds = (preds > 0.5).squeeze().detach()

            gt = gt.squeeze()
            self.evaluator.compute_metrics(preds, gt)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')

        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }

        return epoch_train_loss, epoch_iou



    def compute_surface_dice(self, output, gt, tolerance=1.0, spacing=(0.15, 0.15, 0.15)):
        from scipy.ndimage import binary_erosion, binary_dilation
        
        output_np = output.cpu().numpy().astype(np.uint8)
        gt_np = gt.cpu().numpy().astype(np.uint8)
        
        if output_np.sum() == 0 or gt_np.sum() == 0:
            return 0.0

        try:
            output_surface = binary_dilation(output_np.astype(bool)) ^ binary_erosion(output_np.astype(bool))
            gt_surface = binary_dilation(gt_np.astype(bool)) ^ binary_erosion(gt_np.astype(bool))
        except:
            output_surface = output_np.astype(bool) ^ binary_erosion(output_np.astype(bool))
            gt_surface = gt_np.astype(bool) ^ binary_erosion(gt_np.astype(bool))
        
        if output_surface.sum() == 0 or gt_surface.sum() == 0:
            return 0.0
        
        dist_gt = distance_transform_edt(~gt_surface, sampling=spacing)
        dist_pred = distance_transform_edt(~output_surface, sampling=spacing)
        
        output_surface_coords = np.argwhere(output_surface)
        gt_surface_coords = np.argwhere(gt_surface)
        
        if len(output_surface_coords) == 0 or len(gt_surface_coords) == 0:
            return 0.0
        
        distances_pred_to_gt = dist_gt[output_surface]
        pred_within_tolerance = np.sum(distances_pred_to_gt <= tolerance)
        
        distances_gt_to_pred = dist_pred[gt_surface]
        gt_within_tolerance = np.sum(distances_gt_to_pred <= tolerance)
        
        surface_dice = (pred_within_tolerance + gt_within_tolerance) / (len(output_surface_coords) + len(gt_surface_coords))
        
        return float(surface_dice)



    def compute_ncc(self, output, volume_threshold=500):
        output_np = output.cpu().numpy().astype(np.uint8)
        
        if output_np.sum() == 0:
            return 0
        
        labeled_array, num_components = label(output_np)
        
        large_volume_count = 0
        for region_label in range(1, num_components + 1):
            region_volume = np.sum(labeled_array == region_label)
            if region_volume >= volume_threshold:
                large_volume_count += 1
        
        return large_volume_count



    def compute_hausdorff_distance(self, output, gt):
        pred_coords = torch.nonzero(output, as_tuple=False).float()
        gt_coords = torch.nonzero(gt, as_tuple=False).float()

        if pred_coords.size(0)==0 or gt_coords.size(0)==0:
            return float('inf')

        d_matrix = torch.cdist(pred_coords, gt_coords)
        hd_forward = d_matrix.min(dim=1)[0].max()
        hd_backward = d_matrix.min(dim=0)[0].max()
        hausdorff_distance = torch.max(hd_forward, hd_backward)
        return hausdorff_distance.item()



    def compute_hausdorff_95_distance(self, output, gt):
        pred_coords = torch.nonzero(output, as_tuple=False).float()
        gt_coords = torch.nonzero(gt, as_tuple=False).float()

        if pred_coords.size(0)==0 or gt_coords.size(0)==0:
            return float('inf')

        d_matrix = torch.cdist(pred_coords, gt_coords)
        min_dists, _ = torch.min(d_matrix, dim=1)

        weights = torch.tensor([0.25, 0.5, 0.25], device=output.device)
        percentiles = torch.tensor([90.0, 95.0, 99.0], device=output.device)

        hausdorff_percentiles = torch.quantile(min_dists, percentiles / 100.0)
        weighted_hd95 = torch.sum(weights * hausdorff_percentiles)

        return weighted_hd95.item()

    def compute_smoothness(self, output):
        device = output.device
        output = output.float().unsqueeze(0).unsqueeze(0)

        sobel_kernel_x = torch.tensor([
            [[[-1,0,1],[-2,0,2],[-1,0,1]],
            [[-2,0,2],[-4,0,4],[-2,0,2]],
            [[-1,0,1],[-2,0,2],[-1,0,1]]]], device=device).unsqueeze(0).float()/32.0

        sobel_kernel_y = torch.tensor([
            [[[-1,-2,-1],[0,0,0],[1,2,1]],
            [[-2,-4,-2],[0,0,0],[2,4,2]],
            [[-1,-2,-1],[0,0,0],[1,2,1]]]], device=device).unsqueeze(0).float()/32.0

        sobel_kernel_z = torch.tensor([
            [[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]],
            [[0,0,0],[0,0,0],[0,0,0]],
            [[1,2,1],[2,4,2],[1,2,1]]]], device=device).unsqueeze(0).float()/32.0

        pad = (1,1,1)

        grad_x = F.conv3d(output, sobel_kernel_x, padding=pad)
        grad_y = F.conv3d(output, sobel_kernel_y, padding=pad)
        grad_z = F.conv3d(output, sobel_kernel_z, padding=pad)

        norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        normal_vectors = torch.cat([grad_x/norm, grad_y/norm, grad_z/norm], dim=1)

        diff_x = normal_vectors[:,:, :-1, :-1, :-1] - normal_vectors[:,:, 1:, :-1, :-1]
        diff_y = normal_vectors[:,:, :-1, :-1, :-1] - normal_vectors[:,:, :-1, 1:, :-1]
        diff_z = normal_vectors[:,:, :-1, :-1, :-1] - normal_vectors[:,:, :-1, :-1, 1:]

        normal_diff = diff_x**2 + diff_y**2 + diff_z**2
        smoothness = torch.mean(torch.sqrt(torch.sum(normal_diff, dim=1) + 1e-8))

        return smoothness.item() * 1000
    


    def compute_small_volume_detection(self, output, volume_threshold = 500):
        output_np = output.cpu().numpy()

        labeled_array, num_features = label(output_np)

        small_volume_count = 0
        for region_label in range(1, num_features + 1):
            region_volume = np.sum(labeled_array == region_label)
            if region_volume < volume_threshold:
                small_volume_count += 1

        return small_volume_count



    def compute_iou_and_dice(self, pred, gt):
        intersection = torch.logical_and(pred, gt).sum().float()
        union = torch.logical_or(pred, gt).sum().float()

        iou = intersection / (union + 1e-8)
        dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)

        return iou.item(), dice.item()
    


    def compute_left_right_metrics(self, output, gt):
        x_center = output.shape[2] // 2

        output_left = output[:, :, x_center:]
        gt_left = gt[:, :, x_center:]

        output_right = output[:, :, :x_center]
        gt_right = gt[:, :, :x_center]

        left_iou, left_dice = self.compute_iou_and_dice(output_left, gt_left)
        left_hausdorff = self.compute_hausdorff_distance(output_left, gt_left)
        left_hausdorff_95 = self.compute_hausdorff_95_distance(output_left, gt_left)
        left_smoothness = self.compute_smoothness(output_left)
        left_small_vol = self.compute_small_volume_detection(output_left)
        left_surface_dice = self.compute_surface_dice(output_left, gt_left, tolerance=1.0)
        left_ncc = self.compute_ncc(output_left)

        right_iou, right_dice = self.compute_iou_and_dice(output_right, gt_right)
        right_hausdorff = self.compute_hausdorff_distance(output_right, gt_right)
        right_hausdorff_95 = self.compute_hausdorff_95_distance(output_right, gt_right)
        right_smoothness = self.compute_smoothness(output_right)
        right_small_vol = self.compute_small_volume_detection(output_right)
        right_surface_dice = self.compute_surface_dice(output_right, gt_right, tolerance=1.0)
        right_ncc = self.compute_ncc(output_right)

        left_metrics = (left_iou, left_dice, left_hausdorff, left_hausdorff_95, left_smoothness, left_small_vol, left_surface_dice, left_ncc)
        right_metrics = (right_iou, right_dice, right_hausdorff, right_hausdorff_95, right_smoothness, right_small_vol, right_surface_dice, right_ncc)

        return left_metrics, right_metrics


    def test(self, phase, alpha=2, save_individual_results=False):
            self.model.eval()

            # test_start_time = time.time()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     torch.cuda.reset_peak_memory_stats()
            #     start_gpu_memory = torch.cuda.memory_allocated() / 1024**2

            with torch.inference_mode():
                self.evaluator.reset_eval()
                losses = []
                hausdorff_distances, hausdorff_distances_95, smoothnesses, small_volume_detections = [], [], [], []
                surface_dices, nccs = [], []
                left_metrics_list, right_metrics_list = [], []
                
                individual_results = []

                if phase == 'Test':
                    dataset = self.test_dataset
                    patch_overlap = 36
                elif phase == 'Validation':
                    dataset = self.val_dataset
                    patch_overlap = 0

                for subject_idx, subject in enumerate(tqdm(dataset, desc=f'{phase} epoch {str(self.epoch)}')):
                    sampler = tio.inference.GridSampler(subject, self.config.data_loader.patch_shape, patch_overlap)
                    loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                    aggregator = tio.inference.GridAggregator(sampler)
                    gt_aggregator = tio.inference.GridAggregator(sampler)

                    for patch in loader:
                        images, gt_patch, emb_codes = self.extract_data_from_patch(patch)
                        preds = self.model(images, emb_codes)
                        aggregator.add_batch(preds, patch[tio.LOCATION])
                        gt_aggregator.add_batch(gt_patch, patch[tio.LOCATION])

                    output = aggregator.get_output_tensor()
                    gt = gt_aggregator.get_output_tensor()

                    partition_weights = 1
                    gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                    if torch.sum(gt_count) != 0:
                        partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))

                    loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
                    losses.append(loss.item())

                    output = output.squeeze(0)
                    output_bin = (output > 0.5)
                    
                    self.evaluator.compute_metrics(output_bin, gt)
                    groundtruth = gt.squeeze(0)
                    
                    if save_individual_results and phase == 'Test':
                        current_iou, current_dice = self.compute_iou_and_dice(output_bin, gt)
                        
                        sample_result = {
                            'subject_id': subject_idx,
                            'subject_name': getattr(subject, 'name', f'subject_{subject_idx}'),
                            'iou': current_iou,
                            'dice': current_dice
                        }

                    if phase == 'Test':
                        if alpha >= 1:
                            hd = self.compute_hausdorff_distance(output_bin, groundtruth)
                            hd95 = self.compute_hausdorff_95_distance(output_bin, groundtruth)
                            smooth = self.compute_smoothness(output_bin)
                            small_vol = self.compute_small_volume_detection(output_bin)
                            surf_dice = self.compute_surface_dice(output_bin, groundtruth, tolerance=1.0)
                            ncc = self.compute_ncc(output_bin)
                            
                            hausdorff_distances.append(hd)
                            hausdorff_distances_95.append(hd95)
                            smoothnesses.append(smooth)
                            small_volume_detections.append(small_vol)
                            surface_dices.append(surf_dice)
                            nccs.append(ncc)
                            
                            sample_result.update({
                                'hausdorff_distance': hd,
                                'hausdorff_distance_95': hd95,
                                'smoothness': smooth,
                                'small_volume_detection': small_vol,
                                'surface_dice': surf_dice,
                                'ncc': ncc,
                            })

                        if alpha == 2:
                            left_metrics, right_metrics = self.compute_left_right_metrics(output_bin, groundtruth)
                            left_metrics_list.append(left_metrics)
                            right_metrics_list.append(right_metrics)
                            
                            sample_result.update({
                                'right_iou': right_metrics[0],
                                'right_dice': right_metrics[1],
                                'right_hausdorff': right_metrics[2],
                                'right_hausdorff_95': right_metrics[3],
                                'right_smoothness': right_metrics[4],
                                'right_small_vol': right_metrics[5],
                                'right_surface_dice': right_metrics[6],
                                'right_ncc': right_metrics[7],
                                'left_iou': left_metrics[0],
                                'left_dice': left_metrics[1],
                                'left_hausdorff': left_metrics[2],
                                'left_hausdorff_95': left_metrics[3],
                                'left_smoothness': left_metrics[4],
                                'left_small_vol': left_metrics[5],
                                'left_surface_dice': left_metrics[6],
                                'left_ncc': left_metrics[7],
                            })
                    
                        individual_results.append(sample_result)

            # test_end_time = time.time()
            # total_test_time = test_end_time - test_start_time
            
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            #     end_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            #     gpu_memory_used = peak_gpu_memory - start_gpu_memory
            # else:
            #     peak_gpu_memory = 0.0
            #     gpu_memory_used = 0.0
            
            # num_samples = len(individual_results)
            # avg_time_per_sample = total_test_time / num_samples if num_samples > 0 else 0.0
            
            # if phase == 'Test':
            #     logging.info(f"\n=== Performance Statistics ({phase}) ===")
            #     logging.info(f"Total {phase.lower()} time: {total_test_time:.2f}s")
            #     logging.info(f"Number of samples: {num_samples}")
            #     logging.info(f"Average time per sample: {avg_time_per_sample:.3f}s")
            #     if torch.cuda.is_available():
            #         logging.info(f"Peak GPU memory usage: {peak_gpu_memory:.2f}MB")
            #         logging.info(f"GPU memory increase: {gpu_memory_used:.2f}MB")
            #     logging.info(f"Throughput: {num_samples/total_test_time:.2f} samples/sec")

            if save_individual_results and phase == 'Test':
                import pandas as pd
                import json
                
                df = pd.DataFrame(individual_results)
                csv_path = os.path.join(self.config.project_dir, self.config.title, 
                                    f'individual_test_results_epoch_{self.epoch}.csv')
                df.to_csv(csv_path, index=False)
                logging.info(f'Saved individual test results to {csv_path}')
                
                json_path = os.path.join(self.config.project_dir, self.config.title, 
                                        f'individual_test_results_epoch_{self.epoch}.json')
                with open(json_path, 'w') as f:
                    json.dump(individual_results, f, indent=4)
                logging.info(f'Saved individual test results to {json_path}')
                
                # performance_summary = {
                #     'total_test_time_seconds': total_test_time,
                #     'num_samples': num_samples,
                #     'avg_time_per_sample_seconds': avg_time_per_sample,
                #     'throughput_samples_per_sec': num_samples/total_test_time if total_test_time > 0 else 0,
                #     'peak_gpu_memory_mb': peak_gpu_memory,
                #     'gpu_memory_increase_mb': gpu_memory_used,
                #     'epoch': self.epoch,
                #     'phase': phase
                # }
                
                # perf_path = os.path.join(self.config.project_dir, self.config.title, 
                #                     f'performance_summary_{phase.lower()}_epoch_{self.epoch}.json')
                # with open(perf_path, 'w') as f:
                #     json.dump(performance_summary, f, indent=4)
                # logging.info(f'Saved performance summary to {perf_path}')

            epoch_loss = sum(losses) / len(losses)
            epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)

            if phase == 'Test':
                if alpha == 0:
                    if save_individual_results:
                        return epoch_iou, epoch_dice, individual_results
                    return epoch_iou, epoch_dice

                epoch_hausdorff_distance = np.mean(hausdorff_distances)
                epoch_hausdorff_distance_95 = np.mean(hausdorff_distances_95)
                epoch_smoothness = np.mean(smoothnesses)
                epoch_small_volume_detection = np.mean(small_volume_detections)
                epoch_surface_dice = np.mean(surface_dices) 
                epoch_ncc = np.mean(nccs)

                if alpha == 1:
                    if save_individual_results:
                        return (
                            epoch_iou, epoch_dice, epoch_hausdorff_distance, epoch_hausdorff_distance_95,
                            epoch_smoothness, epoch_small_volume_detection, epoch_surface_dice, epoch_ncc,
                            individual_results
                        )
                    return (
                        epoch_iou, epoch_dice, epoch_hausdorff_distance, epoch_hausdorff_distance_95,
                        epoch_smoothness, epoch_small_volume_detection, epoch_surface_dice, epoch_ncc
                    )

                elif alpha == 2:
                    left_avg_metrics = np.mean(left_metrics_list, axis=0)
                    right_avg_metrics = np.mean(right_metrics_list, axis=0)
                    
                    results = (
                        epoch_iou, epoch_dice, epoch_hausdorff_distance, epoch_hausdorff_distance_95,
                        epoch_smoothness, epoch_small_volume_detection, epoch_surface_dice, epoch_ncc,
                        right_avg_metrics[0], right_avg_metrics[1], right_avg_metrics[2], 
                        right_avg_metrics[3], right_avg_metrics[4], right_avg_metrics[5],
                        right_avg_metrics[6], right_avg_metrics[7],
                        left_avg_metrics[0], left_avg_metrics[1], left_avg_metrics[2], 
                        left_avg_metrics[3], left_avg_metrics[4], left_avg_metrics[5],
                        left_avg_metrics[6], left_avg_metrics[7],
                    )
                    
                    if save_individual_results:
                        return results + (individual_results,)
                    return results

            elif phase == 'Validation':
                if save_individual_results:
                    return epoch_iou, epoch_dice, individual_results
                return epoch_iou, epoch_dice