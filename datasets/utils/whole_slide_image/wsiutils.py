import numpy as np
import cv2
from PIL import Image
from scipy.signal import oaconvolve
from itertools import product

class isInContourV3_Easy:
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (int(pt[0] + self.patch_size // 2), int(pt[1] + self.patch_size // 2))
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) >= 0:
                return 1
        return 0


class isInContourV3_Hard:
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (int(pt[0] + self.patch_size // 2), int(pt[1] + self.patch_size // 2))
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) < 0:
                return 0
        return 1

def isWhitePatch_HLS(patch, lightThresh=210, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    patch_hls = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2HLS)
    return True if (patch_hls[:, :, 1]>lightThresh).sum() > num_pixels * percentage else False

def remove_black_areas(slide):
    slidearr = np.asarray(slide)
    indices = np.where(np.all(slidearr == (0, 0, 0), axis=-1))
    slidearr[indices] = [255, 255, 255]
    return Image.fromarray(slidearr.astype("uint8"))


def local_average(img, window_size, keep_grayscale):
    window = np.ones((window_size, window_size)) / (window_size ** 2)
    img_grayscaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keep_grayscale:
        return img_grayscaled
    else:
        return img_grayscaled - oaconvolve(img_grayscaled, window, mode='same')


def compute_law_feats(img, window_size):
    def get_img_energy(name):
        return imgs_energy[int(np.where(filters_name == name)[0])]

    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])

    vectors = [L5, E5, S5, R5]
    vectors_name = ["L5", "E5", "S5", "R5"]
    filters = [np.expand_dims(vectors[i], -1).dot(np.expand_dims(vectors[j], -1).T) for i, j in
               product(range(len(vectors)), range(len(vectors)))]
    filters_name = np.array([vectors_name[i] + vectors_name[j] for i, j in
                             product(range(len(vectors)), range(len(vectors)))])

    imgs_filtered = []
    for filt in filters:
        imgs_filtered.append(oaconvolve(img, filt, mode="same"))

    window = np.ones((window_size, window_size))
    imgs_energy = []
    for img_filtered in imgs_filtered:
        imgs_energy.append(oaconvolve(np.abs(img_filtered), window, mode="same"))

    imgs_feats = [np.mean(np.array([get_img_energy("L5E5"), get_img_energy("E5L5")]), axis=0),
                  np.mean(np.array([get_img_energy("L5R5"), get_img_energy("R5L5")]), axis=0),
                  np.mean(np.array([get_img_energy("E5S5"), get_img_energy("S5E5")]), axis=0),
                  get_img_energy("S5S5"),
                  get_img_energy("R5R5"),
                  np.mean(np.array([get_img_energy("L5S5"), get_img_energy("S5L5")]), axis=0),
                  get_img_energy("E5E5"),
                  np.mean(np.array([get_img_energy("E5R5"), get_img_energy("R5E5")]), axis=0),
                  np.mean(np.array([get_img_energy("S5R5"), get_img_energy("R5S5")]), axis=0)]

    return np.stack(imgs_feats, axis=-1)


def filter_ROI(roi):
    img = np.asarray(roi)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.bilateralFilter(img, 3, 3 * 2, 3 / 2)
    return img_filtered


def thresh_ROI(roi, thresh, inv):
    if inv:
        _, img_thresh = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, img_thresh = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    return img_thresh


def floodfill_ROI(roi, start):
    im_floodfill = roi.copy()
    h, w = roi.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, start, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = roi | im_floodfill_inv
    return im_out


def contour_ROI(roi):
    contours = cv2.findContours(roi, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours