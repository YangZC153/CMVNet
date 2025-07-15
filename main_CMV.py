import sys
import os
import argparse
import logging
import shutil
import yaml
import random
import time
import numpy as np
import torch
from hashlib import shake_256
from munch import Munch, munchify, unmunchify
from os import path
from experiments.ExperimentFactory import ExperimentFactory
from dataloader.AugFactory import AugFactory


def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)
    return h.upper()

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="/home/yangzhencheng/CMV/configs/CMV/seg-CMV_test.yaml", required=False)  
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    logging.info(f'setup to be deterministic')
    setup(config.seed)

    os.environ['WANDB_DISABLED'] = 'true'

    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preproc = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preproc = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preproc_file:
            preproc = yaml.load(preproc_file, yaml.FullLoader)
    config.data_loader.preprocessing = AugFactory(preproc).get_transform()

    logging.info(f'loading augmentations')
    if config.data_loader.augmentations is None:
        aug = []
    elif not os.path.exists(config.data_loader.augmentations):
        logging.warning(f'Augmentations file does not exist: {config.augmentations}')
        aug = []
    else:
        with open(config.data_loader.augmentations) as aug_file:
            aug = yaml.load(aug_file, yaml.FullLoader)
    config.data_loader.augmentations = AugFactory(aug).get_transform()

    config.title = f'{config.title}_{timehash()}'


    logging.info(f'Instantiation of the experiment')
    experiment = ExperimentFactory(config, args.debug).get()
    logging.info(f'experiment title: {experiment.config.title}')

    project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.title)
    os.makedirs(project_dir_title, exist_ok=True)
    logging.info(f'project directory: {project_dir_title}')

    file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    copy_config_path = os.path.join(project_dir_title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(experiment.config.data_loader.dataset):
        logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
        raise SystemExit

    checkpoints_path = path.join(project_dir_title, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
        raise SystemExit

    best_val = float('-inf')
    best_test = {
            'value': float('-inf'),
            'epoch': -1
            }

    if config.trainer.do_train:
        logging.info('Training...')
        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs+1):
            experiment.train()

            val_iou, val_dice = experiment.test(phase="Validation")
            logging.info(f'Epoch {epoch} Val IoU: {val_iou}')
            logging.info(f'Epoch {epoch} Val Dice: {val_dice}')

            if val_iou < 1e-05 and experiment.epoch > 15:
                logging.warning('WARNING: drop in performances detected.')

            optim_name = experiment.optimizer.name
            sched_name = experiment.scheduler.name

            if experiment.scheduler is not None:
                if optim_name == 'SGD' and sched_name == 'Plateau':
                    experiment.scheduler.step(val_iou)
                else:
                    experiment.scheduler.step()

            experiment.save('last.pth')

            if val_iou > best_val:
                best_val = val_iou
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best test IoU found: {best_test['value']} at epoch: {best_test['epoch']}
                ''')

    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        (test_iou, test_dice, test_hausdorff_distance, test_hausdorff_distance_95, test_smoothness, test_small_volume_detection,
        right_iou, right_dice, right_hausdorff_distance, right_hausdorff_distance_95, right_smoothness, right_small_volume_detection,
        left_iou, left_dice, left_hausdorff_distance, left_hausdorff_distance_95, left_smoothness, left_small_volume_detection) = experiment.test(phase="Test")

        logging.info(f'\nTest results (Overall) - IoU: {test_iou}, Dice: {test_dice}, Hausdorff Distance: {test_hausdorff_distance}, '
                    f'Hausdorff Distance 95: {test_hausdorff_distance_95}, Smoothness: {test_smoothness}, Small Volume Detection: {test_small_volume_detection}')

        logging.info(f'\nTest results (Right) - IoU: {right_iou}, Dice: {right_dice}, Hausdorff Distance: {right_hausdorff_distance}, '
                    f'Hausdorff Distance 95: {right_hausdorff_distance_95}, Smoothness: {right_smoothness}, Small Volume Detection: {right_small_volume_detection}')

        logging.info(f'\nTest results (Left) - IoU: {left_iou}, Dice: {left_dice}, Hausdorff Distance: {left_hausdorff_distance}, '
                    f'Hausdorff Distance 95: {left_hausdorff_distance_95}, Smoothness: {left_smoothness}, Small Volume Detection: {left_small_volume_detection}')

    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        experiment.inference(os.path.join(config.data_loader.dataset,'SPARSE'))