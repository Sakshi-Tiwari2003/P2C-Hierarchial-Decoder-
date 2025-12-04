import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import json
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class EPNDataset(data.Dataset):
    def __init__(self, config, default_args=None):
        if default_args is None:
            default_args = {}

        cfg = {**default_args, **config}

        self.npoints = cfg['N_POINTS']
        self.category_file = cfg['CATEGORY_FILE_PATH']
        self.partial_points_path = cfg['PARTIAL_POINTS_PATH']
        self.complete_points_path = cfg['COMPLETE_POINTS_PATH']
        self.class_choice = cfg['class_choice']
        self.subset = cfg['subset']

        with open(self.category_file) as f:
            self.dataset_categories = json.load(f)
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.class_choice]

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([
                {'callback': 'RandomSamplePoints', 'parameters': {'n_points': self.npoints}, 'objects': ['partial', 'complete']},
                {'callback': 'RandomMirrorPoints', 'objects': ['partial', 'complete']},
                {'callback': 'ToTensor', 'objects': ['partial', 'complete']}
            ])
        else:
            return data_transforms.Compose([
                {'callback': 'RandomSamplePoints', 'parameters': {'n_points': self.npoints}, 'objects': ['partial']},
                {'callback': 'ToTensor', 'objects': ['partial', 'complete']}
            ])

    def _get_file_list(self, subset):
        file_list = {'taxonomy_id': [], 'model_id': [], 'partial_path': [], 'gt_path': []}
        missing_samples = 0

        for dc in self.dataset_categories:
            print_log(f'Collecting files of Taxonomy [ID={dc["taxonomy_id"]}, Name={dc["taxonomy_name"]}]', logger='EPN3DDATASET')
            category = dc['taxonomy_name']

            if subset not in dc:
                print_log(f"Warning: Subset '{subset}' not found for class '{category}'", logger='EPN3DDATASET')
                continue

            partial_samples = dc[subset].get('partial', [])
            complete_samples = dc[subset].get('complete', [])

            for p, c in zip(partial_samples, complete_samples):
                partial_path = self.partial_points_path % (category, p)
                gt_path = self.complete_points_path % (category, c)

                if not (os.path.exists(partial_path) and os.path.exists(gt_path)):
                    missing_samples += 1
                    continue

                file_list['taxonomy_id'].append(dc['taxonomy_id'])
                file_list['model_id'].append(c)
                file_list['partial_path'].append(partial_path)
                file_list['gt_path'].append(gt_path)

        file_list['shuffled_gt_path'] = file_list['gt_path'].copy()
        print_log(f'Complete collecting files of the dataset. Total files: {len(file_list["partial_path"])} (Skipped {missing_samples} missing files)', logger='EPN3DDATASET')
        return file_list

    def shuffle_gt(self):
        random.shuffle(self.file_list['shuffled_gt_path'])

    def __getitem__(self, idx):
        sample = {
            'taxonomy_id': self.file_list['taxonomy_id'][idx],
            'model_id': self.file_list['model_id'][idx]
        }
        data = {
            'partial': IO.get(self.file_list['partial_path'][idx]).astype(np.float32),
            'complete': IO.get(
                self.file_list['shuffled_gt_path'][idx] if self.subset == 'train' else self.file_list['gt_path'][idx]
            ).astype(np.float32)
        }
        if self.transforms:
            data = self.transforms(data)
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['complete'])

    def __len__(self):
        return len(self.file_list['partial_path'])

@DATASETS.register_module(name='EPN3D')
class EPN3D(EPNDataset):
    pass
