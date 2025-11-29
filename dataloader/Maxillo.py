import json
from pathlib import Path

import numpy as np
import torch
import torchio as tio

from torch.utils.data import DataLoader

class Maxillo(tio.SubjectsDataset):
    """
    Maxillo dataset
    TODO: Add more information about the dataset
    """

    def __init__(self, root, filename, splits, transform=None, dist_map=None, **kwargs):
        if type(dist_map) == str:
            dist_map = [dist_map]

        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]

        subjects_list = self._get_subjects_list(root, filename, splits, dist_map)
        super().__init__(subjects_list, transform, **kwargs)


    def _numpy_reader(self, path):
        data = np.load(path)
        if data.dtype == np.uint32:
            data = data.astype(np.float32)
        data = torch.from_numpy(data).float()
        affine = torch.eye(4, requires_grad=False)
        return data, affine


    def _get_subjects_list(self, root, filename, splits, dist_map=None):
        base_dir = root
        dense_dir = root / 'DENSE'
        splits_path = root / filename

        with open(splits_path) as splits_file:
            json_splits = json.load(splits_file)
        
        if dist_map is None:
            dist_map = []

        subjects = []
        for split in splits:
            for patient in json_splits[split]:
                data_path = base_dir / patient / 'img.npy'
                sparse_path = base_dir / patient / 'line.npy'
                dense_path = base_dir / patient / 'label.npy'

                if not data_path.is_file():
                    raise ValueError(f'Missing data file for patient {patient} ({data_path})')

                subject_dict = {
                        'partition': split,
                        'patient': patient,
                        'data': tio.ScalarImage(data_path, reader=self._numpy_reader),
                        'sparse': tio.LabelMap(sparse_path, reader=self._numpy_reader),
                        'dense': tio.LabelMap(dense_path, reader=self._numpy_reader),  # []
                        }
                subjects.append(tio.Subject(**subject_dict))
            print(f"Loaded {len(subjects)} patients for split {split}")
        return subjects

    def get_loader(self, config, aggr=None):
        samples_per_volume = [np.round(i / (j-config.grid_overlap)) for i, j in zip(config.resize_shape, config.patch_shape)]
        samples_per_volume = int(np.prod(samples_per_volume))
        sampler = tio.UniformSampler(patch_size=config.patch_shape)
        queue = tio.Queue(
                subjects_dataset=self,
                max_length=100,
                samples_per_volume=10,
                sampler=sampler,
                num_workers=config.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,
                start_background=False,
        )
        loader = DataLoader(queue, batch_size=config.batch_size, num_workers=0, pin_memory=True)
        return loader

    def get_aggregator(self, config, aggr=None):
        samplers = [ tio.GridSampler(sj, patch_size=config.patch_shape, patch_overlap=0) for sj in self._subjects ]
        return [ (test_p, DataLoader(test_p, 2, num_workers=4)) for test_p in samplers ]

