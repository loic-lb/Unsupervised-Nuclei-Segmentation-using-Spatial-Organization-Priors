import os
import torch
from PIL import Image
from .basedataset import BaseImgMaskDataset


class BenchmarkDatasetDeepLiif(BaseImgMaskDataset):
    def __init__(self, mask_dir, gt_dir, mask_transform=None, gt_transform=None):
        super().__init__(mask_dir, mask_transform, gt_transform)
        self.gt_masks = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.images[idx]
        mask = Image.open(mask)
        gt_mask = self.gt_masks[idx]
        gt_mask = Image.open(gt_mask)
        if self.img_transform:
            mask = self.img_transform(mask)
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)
        return mask, gt_mask
