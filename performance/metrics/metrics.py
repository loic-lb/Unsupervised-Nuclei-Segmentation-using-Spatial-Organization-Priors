from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import hausdorff_distance

eps = 1e-10


def recall_m(pred, targs):
    true_positives = (pred * targs).sum(axis=[1, 2, 3])
    possible_positives = targs.sum(axis=[1, 2, 3])
    recall = (true_positives / (possible_positives + eps)).mean(axis=0)
    return recall


def precision_m(pred, targs):
    true_positives = (pred * targs).sum(axis=[1, 2, 3])
    predicted_positives = pred.sum(axis=[1, 2, 3])
    precision = (true_positives / (predicted_positives + eps)).mean(axis=0)
    return precision


def f1_m(pred, targs):
    true_positives = (pred * targs).sum(axis=[1, 2, 3])
    predicted_positives = pred.sum(axis=[1, 2, 3])
    precision = true_positives / (predicted_positives + eps)
    possible_positives = targs.sum(axis=[1, 2, 3])
    recall = true_positives / (possible_positives + eps)
    return (2 * (precision * recall) / (precision + recall + eps)).mean(axis=0)


def dice_by_sample(pred, targs):
    intersection = (pred * targs).sum(axis=[1, 2, 3])
    union = pred.sum(axis=[1, 2, 3]) + targs.sum(axis=[1, 2, 3])
    return (2 * intersection / (union + eps)).mean(axis=0)


def dice(Gi, Si):
    return 2. * (Gi * Si).sum() / (Gi.sum() + Si.sum() + eps)


def hausdorff(Gi, Si):
    return hausdorff_distance(Gi, Si)
    # return max(directed_hausdorff(Gi, Si)[0],directed_hausdorff(Si, Gi)[0])
