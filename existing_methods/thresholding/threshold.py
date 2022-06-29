import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import gaussian

import histomicstk as htk


input_crude_image_folder = '....'
output_post_processed_image_folder = os.path.join(os.path.dirname(__file__), 'threshold_loic')
if not os.path.exists(output_post_processed_image_folder):
    os.makedirs(output_post_processed_image_folder)
is_done_post_processing = os.path.join(output_post_processed_image_folder, 'done')

if not os.path.exists(is_done_post_processing):
    # List all crude input images
    crude_images = os.listdir(input_crude_image_folder)
    crude_images = list(map(lambda f: os.path.join(input_crude_image_folder, f), crude_images))

    for crude_image in tqdm(crude_images):
        npy_img = np.asarray(Image.open(crude_image))
        cut_npy_img = npy_img[:, :npy_img.shape[1] // 6, :]
        np.save(os.path.join(output_post_processed_image_folder, os.path.basename(crude_image)[:-4] + '.npy'),
                cut_npy_img)
    open(is_done_post_processing, 'w').close()

# List all npy images
npy_imgs_paths = os.listdir(output_post_processed_image_folder)
npy_imgs_paths = list(filter(lambda f: f.endswith('.npy'), npy_imgs_paths))
npy_imgs_paths = list(map(lambda f: os.path.join(output_post_processed_image_folder, f), npy_imgs_paths))


def get_otsu(grayscale_img):
    blur = cv2.GaussianBlur(grayscale_img, (9, 9), 0)
    _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


npy_imgs = [np.load(f) for f in npy_imgs_paths[:50]]
# w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(np.vstack(npy_imgs[:50]), 255)
w_est = np.asarray([[0.38419878, 0.78642162, -0.45024397],
                    [0.47867977, 0.52978205, 0.84366279],
                    [0.78946626, 0.31760355, -0.29242685]])
print('Estimated stain colors (rows):', w_est.T[:2] * 255, sep='\n')


def get_otsu_by_color_deconvolution(img):
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(img, w_est, 255)

    otsu_1 = get_otsu(deconv_result.Stains[:, :, 0])
    otsu_2 = get_otsu(deconv_result.Stains[:, :, 1])
    return np.logical_and(otsu_1 == 255, otsu_2 == 255) * 255


output_dir = os.path.join(output_post_processed_image_folder, 'predicted_masks_by_threshold')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'otsu_on_grayscale'))
    os.makedirs(os.path.join(output_dir, 'merged', 'otsu_on_grayscale'))
    os.makedirs(os.path.join(output_dir, 'otsu_on_deconvolved_colors'))
    os.makedirs(os.path.join(output_dir, 'merged', 'otsu_on_deconvolved_colors'))
for npy_img_path in tqdm(npy_imgs_paths):
    output_filename = os.path.basename(npy_img_path[:-4] + '.png')
    npy_img = np.load(npy_img_path)
    # Otsu on grayscale
    otsu_grayscale = 255 - get_otsu(cv2.cvtColor(npy_img, cv2.COLOR_BGR2GRAY))
    # plt.imshow(otsu_grayscale)
    # plt.show()
    Image.fromarray(otsu_grayscale).save(os.path.join(output_dir, 'otsu_on_grayscale', output_filename))
    Image.fromarray(np.hstack((npy_img, np.repeat(otsu_grayscale[..., np.newaxis], 3, -1)))).save(
        os.path.join(output_dir, 'merged', 'otsu_on_grayscale', output_filename))
    # Otsu on color deconvolution
    otsu_color_deconvolution = 255 - get_otsu_by_color_deconvolution(npy_img)
    # plt.imshow(otsu_color_deconvolution)
    # plt.show()
    Image.fromarray(otsu_color_deconvolution.astype(np.uint8)).save(
        os.path.join(output_dir, 'otsu_on_deconvolved_colors', output_filename))
    Image.fromarray(np.hstack((npy_img, np.repeat(otsu_color_deconvolution.astype(np.uint8)[..., np.newaxis], 3, -1)))).save(
        os.path.join(output_dir, 'merged', 'otsu_on_deconvolved_colors', output_filename))