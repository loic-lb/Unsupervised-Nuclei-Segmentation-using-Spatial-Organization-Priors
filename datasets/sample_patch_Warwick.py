import os
import argparse
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import IsolationForest


def save_patches(output_dir, case_id, selected_patches):
    work_dir = os.path.join(output_dir, case_id)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    for i, patch in enumerate(selected_patches):
        shutil.copy(patch, os.path.join(work_dir, f"{case_id}-roi_{i}.png"))


def main():
    parser = argparse.ArgumentParser(description='Sample patches from Warwick dataset')
    parser.add_argument('patches_path', type=str, default=None,
                        help='Extracted patches location')
    parser.add_argument('sampled_path', type=str, default=None,
                        help='Sampled patches location')
    parser.add_argument('--number_patches', type=int, default=306,
                        help='Number of patches to sample')
    parser.add_argument('--feature_type', type=str, choices=["resnet", "rgb_hist"], default="resnet",
                        help="Features used to perform the clustering")
    parser.add_argument('--testing_set', action="store_true",
                        help="Sample patches from testing set (else sample patches from training set)")

    args = parser.parse_args()
    Path(args.sampled_path).mkdir(parents=True, exist_ok=True)
    df_contours = pd.read_csv(os.path.join(args.patches_path, "contours.csv"), dtype={'case_id': str})
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity()
    resnet18.eval()
    case_ids = df_contours[df_contours.testing_set == args.testing_set].case_id.unique()
    number_patches_to_extract = np.round(args.number_patches / len(case_ids)).astype(int)
    print(f"Sampling {number_patches_to_extract} patches from {len(case_ids)} slides ...")
    for case_id in tqdm(case_ids):
        files = np.array([os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(args.patches_path,
                                                                                            case_id))
                          for f in filenames if "roi" in f])
        X = []
        for file in files:
            img_patch = imread(file)
            if args.feature_type == "rgb_hist":
                r_histogram, g_histogram, b_histogram = [np.histogram(img_patch[:, :, i], range(256))[0] for i in
                                                         range(3)]
                feature_vector = np.concatenate((r_histogram, g_histogram, b_histogram))
            elif args.feature_type == "resnet":
                feature_vector = resnet18(transforms.ToTensor()(img_patch).unsqueeze(0)).detach().numpy()
            else:
                raise NotImplementedError
            X.append(feature_vector)
        X = np.concatenate(X, axis=0)
        IF = IsolationForest(random_state=0)
        predictions = IF.fit_predict(X)
        X_selected = X[predictions == 1]
        km = KMeans(n_clusters=number_patches_to_extract, random_state=0).fit(X_selected)
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
        selected_patches = files[closest]
        save_patches(args.sampled_path, str(case_id), selected_patches)
    print("... done")


if __name__ == '__main__':
    main()
