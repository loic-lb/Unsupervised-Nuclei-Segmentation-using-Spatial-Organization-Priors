import os
import h5py
import torch
import numpy as np
from PIL import Image
from .basedataset import BaseDataset


class BCDataImgDataset(BaseDataset):

    def __init__(self, img_dir, gt_points_dir, center_crop=False, transform=None):
        super().__init__(img_dir, transform)
        self.gt_points_negative = sorted([os.path.join(gt_points_dir, "negative", file) for file in
                                          os.listdir(os.path.join(gt_points_dir, "negative"))])
        self.gt_points_positive = sorted([os.path.join(gt_points_dir, "positive", file) for file in
                                          os.listdir(os.path.join(gt_points_dir, "positive"))])
        self.center_crop = center_crop
        self.gt_points = self.merge_gt_points()
        self.transform = transform

    def merge_gt_points(self):
        merged_result = []
        for i in range(len(self)):
            pts_im = []
            pos_dset = h5py.File(self.gt_points_positive[i], 'r')
            for pts in pos_dset["coordinates"]:
                pts_im.append((pts[1], pts[0]))
            neg_dset = h5py.File(self.gt_points_negative[i], 'r')
            for pts in neg_dset["coordinates"]:
                pts_im.append((pts[1], pts[0]))
            if self.center_crop:
                indxin = np.logical_and([np.logical_and(k[0], k[1]) for k in pts_im < np.array([564, 564])],
                                        [np.logical_and(k[0], k[1]) for k in np.array([64, 64]) < pts_im])
                merged_result.append(np.array(pts_im)[indxin] - 64)
            else:
                merged_result.append(np.array(pts_im))
        return merged_result

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.dataset[idx]
        img = Image.open(img)
        return self.apply_transform(img)
