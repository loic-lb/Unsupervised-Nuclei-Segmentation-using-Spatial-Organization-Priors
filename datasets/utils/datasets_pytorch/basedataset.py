import os
import random
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self, data_dir, transform):
        self.dataset = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def apply_transform(self, data):
        if self.transform:
            return self.transform(data)
        else:
            return data


class BaseGtDataset(ABC, Dataset):
    def __init__(self, data_dir, data_transform, gt_transform):
        self.dataset = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
        self.data_transform = data_transform
        self.gt_transform = gt_transform

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def apply_transform(self, data, gt):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.data_transform:
            transformed_data = self.data_transform(data)
        else:
            transformed_data = data
        random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform:
            transformed_gt = self.gt_transform(gt)
        else:
            transformed_gt = gt
        return transformed_data, transformed_gt
