import torch
import numpy as np
import skimage.color as skcolor

# Stain deconvolution matrix created with color_unmixing
conv_matrix = [[1.06640743, 1.081867, 0.85172556],
               [0.71290748, 0.95542728, 1.33166524],
               [-0.62692285, 0.81289619, -0.24760368]]


def inv_tensor_operations(img, mean, std):
    img = img * std + mean
    img = img.permute(1, 2, 0)
    return img


def apply_tensor_operations(img, mean, std):
    img = (img - mean) / std
    img = img.permute(2, 0, 1)
    return img


def create_stain_aug_transform(mean=1.0, std=0.03):
    def transform(img):
        img = np.asarray(img)
        hdx = skcolor.separate_stains(img, np.linalg.inv(conv_matrix))
        alphas = np.clip(np.random.normal(size=(1, 1, 3), loc=mean, scale=std), 0, 1)
        hdx = hdx * alphas
        img = skcolor.combine_stains(hdx, conv_matrix).astype(np.float32)
        return torch.tensor(img)

    return transform


def create_data_ihc_aug(data, transform, mean, std):
    result = []
    for image in data:
        image = inv_tensor_operations(image, mean, std)
        image = transform(image.numpy())
        result.append(apply_tensor_operations(image, mean, std).unsqueeze(0))
    return torch.cat(result, axis=0)
