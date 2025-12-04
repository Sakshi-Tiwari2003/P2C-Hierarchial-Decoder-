import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCNCompleteDataset(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.others.subset
        self.categories = config.others.categories

        # Load the dataset indexing file
        with open(self.category_file) as f:
            self.dataset_categories = json.load(f)
            self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.categories]

        self.file_list = self._get_file_list()
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        return data_transforms.Compose([
            {
                'callback': 'RandomSamplePoints',
                'parameters': { 'n_points': self.npoints },
                'objects': ['partial', 'gt']
            },
            {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },
            {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }
        ])

    def _get_file_list(self):
        file_list = {
         'taxonomy_id': [],
         'model_id': [],
         'partial_path': [],
         'gt_path': []
        }

        for dc in self.dataset_categories:
            print_log(f"Collecting files of Taxonomy [ID={dc['taxonomy_id']}, Name={dc['taxonomy_name']}]", logger='PCNCompleteDATASET')
            for s in dc[self.subset]:
                partial_path = self.partial_points_path % (dc['taxonomy_id'], s, 0)
                gt_path = self.complete_points_path % (dc['taxonomy_id'], s)

                # Only add sample if both partial and complete files exist
                if os.path.exists(partial_path) and os.path.exists(gt_path):
                   file_list['taxonomy_id'].append(dc['taxonomy_id'])
                   file_list['model_id'].append(s)
                   file_list['partial_path'].append(partial_path)
                   file_list['gt_path'].append(gt_path)
                else:
                   missing = []
                   if not os.path.exists(partial_path):
                        missing.append("partial")
                   if not os.path.exists(gt_path):
                        missing.append("gt")
                   print_log(f"[WARNING] Skipping {s} due to missing {', '.join(missing)} file(s).", logger='PCNCompleteDATASET')

        print_log(f"Complete collecting files of the dataset. Total files: {len(file_list['partial_path'])}", logger='PCNCompleteDATASET')
        return file_list



    def __getitem__(self, idx):
        sample = {}
        data = {}

        sample['taxonomy_id'] = self.file_list['taxonomy_id'][idx]
        sample['model_id'] = self.file_list['model_id'][idx]

        data['partial'] = IO.get(self.file_list['partial_path'][idx]).astype(np.float32)
        data['gt'] = IO.get(self.file_list['gt_path'][idx]).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list['model_id'])
