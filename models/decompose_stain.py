import torch
import numpy as np
import skimage.color as skcolor
from datasets.utils.data_augmentation import inv_tensor_operations, apply_tensor_operations

def decompose_stain(batch, mean, std):
    batch_h = []
    batch_d = []
    for i in range(len(batch)):
        img = inv_tensor_operations(batch[i], mean, std).numpy()
        hdx = skcolor.separate_stains(img, skcolor.hdx_from_rgb)
        null = np.zeros_like(hdx[:, :, 0])
        img_h = torch.tensor(
            skcolor.combine_stains(np.stack((hdx[:, :, 0], null, null), axis=-1), skcolor.rgb_from_hdx).astype(
                np.float32))
        img_d = torch.tensor(
            skcolor.combine_stains(np.stack((null, hdx[:, :, 1], null), axis=-1), skcolor.rgb_from_hdx).astype(
                np.float32))
        batch_h.append(apply_tensor_operations(img_h, mean, std).unsqueeze(0))
        batch_d.append(apply_tensor_operations(img_d, mean, std).unsqueeze(0))
    return torch.cat(batch_h), torch.cat(batch_d)
