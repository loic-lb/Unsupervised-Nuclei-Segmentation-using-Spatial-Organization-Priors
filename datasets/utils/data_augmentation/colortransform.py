import torch
import numpy as np
import skimage.color as skcolor


def inv_tensor_operations(img, mean, std):
    img = img * std + mean
    img = img.permute(1, 2, 0)
    return img


def apply_tensor_operations(img, mean, std):
    img = (img - mean) / std
    img = img.permute(2, 0, 1)
    return img


def create_color_transform(mean=1.0, std=0.03):
    def transform(img):
        img = np.asarray(img)
        hdx = skcolor.separate_stains(img, skcolor.hdx_from_rgb)
        alphas = np.random.normal(size=(1, 1, 3), loc=mean, scale=std)
        hdx = np.clip(hdx * alphas, 0, 1)
        img = skcolor.combine_stains(hdx, skcolor.rgb_from_hdx).astype(np.float32)
        return torch.tensor(img)

    return transform


def create_data_ihc_aug(data, transform, mean, std):
    result = []
    for image in data:
        image = inv_tensor_operations(image, mean, std)
        image = transform(image.numpy())
        result.append(apply_tensor_operations(image, mean, std).unsqueeze(0))
    return torch.cat(result, axis=0)
