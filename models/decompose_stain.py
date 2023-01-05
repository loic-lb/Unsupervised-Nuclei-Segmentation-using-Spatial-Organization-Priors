import torch
import numpy as np
import skimage.color as skcolor
from datasets.utils import inv_tensor_operations, apply_tensor_operations

# Stain deconvolution matrix created with color_unmixing
conv_matrix = [[1.06640743, 1.081867, 0.85172556],
               [0.71290748, 0.95542728, 1.33166524],
               [-0.62692285, 0.81289619, -0.24760368]]


def decompose_stain(batch, mean, std):
    batch_h = []
    batch_d = []
    for i in range(len(batch)):
        img = inv_tensor_operations(batch[i], mean, std).numpy()
        hdx = skcolor.separate_stains(img, np.linalg.inv(conv_matrix))
        null = np.zeros_like(hdx[:, :, 0])
        img_h = torch.tensor(
            skcolor.combine_stains(np.stack((hdx[:, :, 0], null, null), axis=-1), conv_matrix).astype(
                np.float32))
        img_d = torch.tensor(
            skcolor.combine_stains(np.stack((null, hdx[:, :, 1], null), axis=-1), conv_matrix).astype(
                np.float32))
        batch_h.append(apply_tensor_operations(img_h, mean, std).unsqueeze(0))
        batch_d.append(apply_tensor_operations(img_d, mean, std).unsqueeze(0))
    return torch.cat(batch_h), torch.cat(batch_d)
