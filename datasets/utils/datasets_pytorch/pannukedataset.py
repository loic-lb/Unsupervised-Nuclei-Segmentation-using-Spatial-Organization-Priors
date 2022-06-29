import torch
from PIL import Image
from .basedataset import BaseDataset


class PannukeMasksDataset(BaseDataset):

    def __init__(self, mask_dir, transform=None):
        super().__init__(mask_dir, transform)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.dataset[idx]
        mask = Image.open(mask)
        if self.transform:
            mask = self.transform(mask)
        return mask
