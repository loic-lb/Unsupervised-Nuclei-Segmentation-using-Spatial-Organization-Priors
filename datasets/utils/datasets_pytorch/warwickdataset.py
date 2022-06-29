import os
import torch
from PIL import Image
from .basedataset import BaseDataset, BaseGtDataset


class WarwickImgDataset(BaseDataset):

    def __init__(self, img_path, transform=None):
        super().__init__(img_path, transform)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx]
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        return img


class WarwickImgMaskDataset(BaseGtDataset):

    def __init__(self, img_path, mask_path, img_transform=None, mask_transform=None):
        super().__init__(img_path, img_transform, mask_transform)
        self.masks = sorted([os.path.join(mask_path, file) for file in os.listdir(mask_path)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx]
        img = Image.open(img)
        mask = self.masks[idx]
        mask = Image.open(mask).convert('L')
        img, mask = self.apply_transform(img, mask)
        mask = (mask > 0.5).float()
        return img, mask


class WarwickBenchmarkDataset(BaseGtDataset):

    def __init__(self, mask_dir, gt_dir, mask_transform=None, gt_transform=None):
        super().__init__(mask_dir, mask_transform, gt_transform)
        self.gt = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.dataset[idx]
        mask = Image.open(mask)
        gt_mask = self.gt[idx]
        gt_mask = Image.open(gt_mask).convert('L')
        mask, gt_mask = self.apply_transform(mask, gt_mask)
        mask = mask.float()
        gt_mask = (gt_mask > 0.5).float()
        return mask, gt_mask
