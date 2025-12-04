import os
import numpy as np
from torch.utils.data import Dataset

class ShapeNetDataset(Dataset):
    def __init__(self, partial_points_path, complete_points_path, subset, class_choice):
        self.partial_points_path = partial_points_path
        self.complete_points_path = complete_points_path
        self.subset = subset
        self.class_choice = class_choice

        # Load file paths based on class choice and subset
        self.partial_files = self._get_files(partial_points_path, class_choice)
        self.complete_files = self._get_files(complete_points_path, class_choice)

    def _get_files(self, path_template, class_choice):
        files = []
        for class_id in class_choice:
            class_path = path_template % class_id
            files += [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.npy')]
        return files

    def __len__(self):
        return len(self.partial_files)

    def __getitem__(self, idx):
        partial = np.load(self.partial_files[idx])
        complete = np.load(self.complete_files[idx])
        return partial, complete
