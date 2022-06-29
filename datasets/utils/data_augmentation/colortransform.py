import torch
import numpy as np
import skimage.color as skcolor


def create_color_transform(mean=1.0, std=0.03):
    def transform(img):
        img = np.asarray(img)
        hdx = skcolor.separate_stains(img, skcolor.hdx_from_rgb)
        alphas = np.random.normal(size=(1, 1, 3), loc=mean, scale=std)
        hdx = hdx * alphas
        img = skcolor.combine_stains(hdx, skcolor.rgb_from_hdx).astype(np.float32)
        return torch.tensor(img)

    return transform


def create_data_ihc_aug(data, transform):
    result = []
    for image in data:
        image = image * 0.5 + 0.5
        image = image.permute(1, 2, 0)
        image = transform(image.numpy())
        result.append((image.permute(2, 0, 1) / 0.5 - 0.5).unsqueeze(0))
    return torch.cat(result, axis=0)
