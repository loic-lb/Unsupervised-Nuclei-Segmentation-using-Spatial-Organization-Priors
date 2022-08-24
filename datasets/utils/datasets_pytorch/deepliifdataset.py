import os
import cv2
import numpy as np
from PIL import Image
import torch
from .basedataset import BaseDataset, BaseGtDataset


class DeepLIIFImgDataset(BaseDataset):

    def __init__(self, img_dir, transform=None):
        super().__init__(img_dir, transform)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx]
        base_image = Image.open(img)
        image = base_image.crop((0, 0, 512, 512))
        return self.apply_transform(image)


class DeepLIIFImgMaskDataset(BaseGtDataset):

    def __init__(self, img_dir, img_transform=None, mask_transform=None):
        super().__init__(img_dir, img_transform, mask_transform)

    @staticmethod
    def process_target(img):
        img = np.array(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_range = np.array([110, 50, 50])
        upper_range = np.array([130, 255, 255])
        mask2 = cv2.inRange(img_hsv, lower_range, upper_range)

        mask = mask0 + mask1 + mask2

        return Image.fromarray(((mask > 0) * 255.).astype(np.uint8))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx]
        base_image = Image.open(img)
        image = base_image.crop((0, 0, 512, 512))
        target = base_image.crop((5 * 512, 0, 6 * 512, 512))
        target = self.process_target(target)
        return self.apply_transform(image, target)


class DeepLIIFBenchmarkDataset(BaseGtDataset):

    def __init__(self, mask_dir, gt_dir, mask_transform=None, gt_transform=None):
        super().__init__(mask_dir, mask_transform, gt_transform)
        self.gt = sorted([os.path.join(gt_dir, file) for file in os.listdir(gt_dir)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.dataset[idx]
        mask = Image.open(mask)
        base_image = self.gt[idx]
        base_image = Image.open(base_image)
        gt_mask = base_image.crop((5 * 512, 0, 6 * 512, 512))
        gt_mask = DeepLIIFImgMaskDataset.process_target(gt_mask)
        return self.apply_transform(mask, gt_mask)

