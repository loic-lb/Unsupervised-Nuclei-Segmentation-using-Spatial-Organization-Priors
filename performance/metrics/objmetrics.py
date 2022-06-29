import numpy as np
from statistics import mode
from skimage.measure import label as sklabel
from postprocessing import compute_watershed
from performance.metrics import dice, hausdorff

eps = 1e-10


def compute_temp(labelA, labelledA, labelledB, metric):
    temp = 0
    total_area = (labelledA > 0).sum()
    for label in labelA:
        Ai = labelledA == label
        intersect = list(labelledB[Ai])
        intersect = [i for i in intersect if i != 0]
        if len(intersect) > 0:
            indexBi = mode(intersect)
            Bi = labelledB == indexBi
        else:
            if metric == hausdorff:
                continue
            else:
                Bi = np.zeros(labelledB.shape, dtype=bool)

        omegai = Ai.sum() / total_area
        temp = temp + omegai * metric(Bi, Ai)
    return temp


def obj_metric(S, G, metric, watershed=False, already_labelled=False):
    scores = []
    for i in range(len(G)):
        if watershed:
            labelled_S = compute_watershed(S[i].numpy().squeeze().astype(np.uint8))
        elif already_labelled:
            labelled_S = S[i].numpy().squeeze(0)
        else:
            labelled_S = sklabel(S[i].numpy().squeeze().astype(np.uint8))

        labelled_G = sklabel(np.array(G[i]).squeeze())

        labels_S = list(np.unique(labelled_S))
        labels_S.remove(0)
        numS = len(labels_S)
        labels_G = list(np.unique(labelled_G))
        labels_G.remove(0)
        numG = len(labels_G)

        if numS == 0 and numG == 0:
            if metric == dice:
                scores.append(1)
                continue
            elif metric == hausdorff:
                scores.append(0)
                continue
        elif numS == 0 or numG == 0:
            if metric == dice:
                scores.append(0)
                continue
            elif metric == hausdorff:
                continue

        temp1 = compute_temp(labels_S, labelled_S, labelled_G, metric)
        temp2 = compute_temp(labels_G, labelled_G, labelled_S, metric)

        scores.append((temp1 + temp2) / 2)
    return np.mean(scores)


def aji(gt_map, predicted_map):
    gt_list = np.unique(gt_map)
    gt_list = gt_list[1:]
    pr_list = np.unique(predicted_map)
    pr_list = pr_list[1:]
    pr_list = np.asarray([pr_list, np.zeros(len(pr_list))]).T

    overall_correct_count = 0
    union_pixel_count = 0

    i = len(gt_list) - 1
    while len(gt_list) > 0:
        gt = 1 * (gt_map == gt_list[i])
        predicted_match = gt * predicted_map

        if np.sum(predicted_match) == 0:
            union_pixel_count += np.sum(gt)
            gt_list = np.delete(gt_list, i)
            i = len(gt_list) - 1
        else:
            predicted_nuc_index = np.unique(predicted_match)
            predicted_nuc_index = predicted_nuc_index[1:]
            JI = 0
            for j in np.unique(predicted_nuc_index):
                matched = 1 * (predicted_map == j)
                nJI = np.sum(np.logical_and(gt, matched)) / np.sum(np.logical_or(gt, matched))
                if nJI > JI:
                    best_match = j
                    JI = nJI

            predicted_nuclei = 1 * (predicted_map == best_match)
            overall_correct_count += np.sum(np.sum(np.logical_and(gt, predicted_nuclei)))
            union_pixel_count += np.sum(np.sum(np.logical_or(gt, predicted_nuclei)))
            gt_list = np.delete(gt_list, i)
            i = len(gt_list) - 1

            index = np.where(pr_list == best_match)[0]
            pr_list[index, 1] = pr_list[index, 1] + 1

    unused_nuclei_list = np.where(pr_list[:, 1] == 0)[0]
    for k in unused_nuclei_list:
        unused_nuclei = 1 * (predicted_map == pr_list[k, 0])
        union_pixel_count = union_pixel_count + np.sum(unused_nuclei)

    aji = overall_correct_count / (union_pixel_count + eps)
    return aji


def aji_score(S, G, watershed=False, already_labelled=False):
    score = []
    for i in range(len(S)):
        labelled_G = sklabel(G[i].numpy().squeeze())
        if watershed:
            labelled_S = compute_watershed(S[i].numpy().squeeze().astype(np.uint8))
        elif already_labelled:
            labelled_S = np.array(S[i]).squeeze()
        else:
            labelled_S = sklabel(S[i].numpy().squeeze().astype(np.uint8))
        score.append(aji(labelled_G, labelled_S))
    return score
