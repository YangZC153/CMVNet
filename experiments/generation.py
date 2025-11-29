import os
import logging
import numpy as np
import torch
from tqdm import tqdm
import torchio as tio
from torch.utils.data import DataLoader
from experiments.experiment import Experiment
from dataloader.AugFactory import *
from dataloader.Maxillo import Maxillo
import SimpleITK as sitk

class Generation(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)

    def convert_and_save_nifti(self, npy_file_path, output_base_folder):
        patient_id = os.path.basename(os.path.dirname(npy_file_path))
        
        npy_data = np.load(npy_file_path)
        
        sitk_image = sitk.GetImageFromArray(npy_data)
        
        output_path = os.path.join(output_base_folder, f"{patient_id}.nii.gz")
        
        sitk.WriteImage(sitk_image, output_path)
        
        print(f"File saved as {output_path}")

    def inference(self, output_path):

        self.model.eval()

        with torch.no_grad():
            dataset = Maxillo(
                    self.config.data_loader.dataset,
                    'splits.json',
                    splits=['inference'],
                    transform=self.config.data_loader.preprocessing,
                    dist_map=['sparse', 'dense']
            )
            crop_or_pad_transform = tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0)
            for i, subject in tqdm(enumerate(dataset), total=len(dataset)):
                directory = os.path.join(output_path, f'{subject.patient}')
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, 'generated.npy')

                if os.path.exists(file_path) and False:
                    logging.info(f'skipping {subject.patient}...')
                    continue

                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        36,
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')

                logging.info(f'patient {subject.patient}...')
                for j, patch in enumerate(loader):
                    images = patch['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                    sparse = patch['sparse'][tio.DATA].float().cuda()
                    emb_codes = patch[tio.LOCATION].float().cuda()

                    # join sparse + data
                    images = torch.cat([images, sparse], dim=1)
                    output = self.model(images, emb_codes)  # BS, Classes, Z, H, W
                    aggregator.add_batch(output, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                # output = tio.CropOrPad(original_shape, padding_mode=0)(output)
                output = output.squeeze(0)
                output = (output > 0.5).int()
                output = output.detach().cpu().numpy()  # BS, Z, H, W

                np.save(file_path, output)
                self.convert_and_save_nifti(file_path, "/home/admil/YZC/Inference_results/ZJPPH/CMV_Net/")

                logging.info(f'patient {subject.patient} completed, {file_path}.')
