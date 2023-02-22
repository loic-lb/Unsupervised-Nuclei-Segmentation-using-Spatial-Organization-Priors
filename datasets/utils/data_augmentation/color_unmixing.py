import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

eps = 1e-6


def retrieve_centroids(image):
    image = image.astype("float64")
    np.maximum(image, eps, out=image)
    OD = - np.log(image / 255)
    ODrgb = np.mean(OD, axis=-1)
    cx = OD[..., 0] / (ODrgb + eps) - 1
    cy = (OD[..., 1] - OD[..., 2]) / (ODrgb * np.sqrt(3) + eps)
    criteria = OD.mean(axis=-1) > 0.3
    kmeans = KMeans(n_clusters=2, random_state=0)
    if criteria.sum() == 0:
        return
    else:
        kmeans.fit(np.concatenate([np.expand_dims(cx[criteria].flatten(), -1),
                                   np.expand_dims(cy[criteria].flatten(), -1)], axis=-1))
    return kmeans.cluster_centers_


def create_conv_matrix(cx, cy):
    cx_h, cx_d = cx
    cy_h, cy_d = cy
    OD_h_r = 1 + cx_h
    OD_h_g = 0.5 * (2 - cx_h + np.sqrt(3) * cy_h)
    OD_h_b = 0.5 * (2 - cx_h - np.sqrt(3) * cy_h)
    stain_vector_h = np.array([OD_h_r, OD_h_g, OD_h_b])
    OD_d_r = 1 + cx_d
    OD_d_g = 0.5 * (2 - cx_d + np.sqrt(3) * cy_d)
    OD_d_b = 0.5 * (2 - cx_d - np.sqrt(3) * cy_d)
    stain_vector_d = np.array([OD_d_r, OD_d_g, OD_d_b])
    stain_vector_x = np.cross(stain_vector_h, stain_vector_d)
    conv_matrix = np.stack([stain_vector_h, stain_vector_d, stain_vector_x])
    return conv_matrix


images = [Image.open(os.path.join("/path/to/patch_256x256_HER2_train", filepath))
          for filepath in os.listdir("/path/to/patch_256x256_HER2_train") if filepath.endswith(".png")]

dists = []
list_centroids = []
for img in images:
    centroids = retrieve_centroids(np.array(img))
    if centroids is None:
        continue
    else:
        dists.append(np.linalg.norm(centroids[0] - centroids[1]))
        list_centroids.append(centroids)

l = np.percentile(dists, 99)
selected_centroids = np.array(list_centroids)[np.array(dists) > l]
avg_centroid = np.median(selected_centroids, axis=0)
conv_matrix = create_conv_matrix(avg_centroid[:, 0], avg_centroid[:, 1])

print(len(selected_centroids))
print(conv_matrix)
