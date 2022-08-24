from skimage import morphology as skmorphology
from scipy import ndimage as nd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def compute_morphological_operations(img, erosion=False):
    img = nd.median_filter(img, size=5)
    if erosion:
        img = skmorphology.binary_erosion(img, skmorphology.disk(5))
    img = nd.binary_opening(img, structure=skmorphology.disk(2)).astype(int)
    img = nd.binary_closing(img, structure=skmorphology.disk(2)).astype(int)
    img = nd.binary_opening(img, structure=skmorphology.disk(1)).astype(int)
    img = nd.binary_closing(img, structure=skmorphology.disk(1)).astype(int)
    return img


def compute_watershed(img, sigma=1, disk_size=12):
    distance = nd.morphology.distance_transform_edt(img)
    smoothed_distance = nd.gaussian_filter(distance, sigma=sigma)
    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=skmorphology.disk(disk_size),
                                labels=img)  # morphology.disk(10)
    markers = nd.label(local_maxi)[0]
    img_post = watershed(-smoothed_distance, markers, mask=img)
    return img_post
