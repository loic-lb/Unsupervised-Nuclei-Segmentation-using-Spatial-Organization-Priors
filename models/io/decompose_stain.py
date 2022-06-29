import torch
import numpy as np
import skimage.color as skcolor

def inv_tensor_operations(img):
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)
    return img


def apply_tensor_operations(img):
    img = img / 0.5 - 0.5
    img = img.permute(2, 0, 1)
    return img


def decompose_stain(batch):
    batch_h = []
    batch_d = []
    for i in range(len(batch)):
        img = inv_tensor_operations(batch[i]).numpy()
        hdx = skcolor.separate_stains(img, skcolor.hdx_from_rgb)
        null = np.zeros_like(hdx[:, :, 0])
        img_h = torch.tensor(
            skcolor.combine_stains(np.stack((hdx[:, :, 0], null, null), axis=-1), skcolor.rgb_from_hdx).astype(
                np.float32))
        img_d = torch.tensor(
            skcolor.combine_stains(np.stack((null, hdx[:, :, 1], null), axis=-1), skcolor.rgb_from_hdx).astype(
                np.float32))
        batch_h.append(apply_tensor_operations(img_h).unsqueeze(0))
        batch_d.append(apply_tensor_operations(img_d).unsqueeze(0))
    return torch.cat(batch_h), torch.cat(batch_d)
