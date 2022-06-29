# Unsupervised Nuclei Segmentation using Spatial Organization Priors - Datasets

## DeepLIIF

Set of 1667 registered 512x512 IHC images built to train the DEEPLIIF algorithm; only the original Ki67 stained 
and the nuclei segmentation images were used - see [here](https://zenodo.org/record/4751737#.YlWRiN86-Ul) for the data.

The data are divided into 709 samples for training, 303 samples for validation and 598 samples for testing. This
decomposition was used to train and validate the results on this dataset.

As the nuclei segmentation masks are identifying positive and negative cells (w.r.t. Ki67) with different colors, these
masks were binarized to discard classification considerations - see [here](utils/datasets_pytorch/deepliifdataset.py) 
for more details regarding the implementation.

## Warwick

Set of 86 HER2 stained WSIs - see [here](https://warwick.ac.uk/fac/cross_fac/tia/data/her2contest) 
for more details and request access to the data.

The data are divided into 53 WSIs fir training and 34 WSIs for testing. This decomposition was used to train
and validate the results on this dataset.

The 53 WSIs were transformed into a training set of 700 512x512 patches, and the 34 WSIs into a testing set of 68 256x256
patches using the following scripts:

```extract_patch_Warwick.py``` retrieves tissue contours and extract patches from slides:
* ```data_path``` describes warwick train (or test) set data location.
* ```result_path``` describes contours (and/or patches) save location.
* ```--segment_contour``` performs contours segmentation, saved as pickle files. It also produces a csv file with 3
columns : 'case_id', 'contours', 'included'. 'Included' can then be modified to remove some segmented tissues from
the patch extraction by setting value to 0 (remaining artifacts, control tissue, ...).
* ```--contour_snapchot``` produces an image of each segmented contours along the pickle files.
* ```--extract_patches``` performs patch extraction after contours segmentation (needs contours.csv and pickle files
resulting from the contour segmentation).

To reproduce the datasets used in the paper, please first segment the contours, and then use the provided 
```contours.csv``` file to extract patches.

```sample_patch_Warwick.py``` samples extracted patches using KMeans partition to extract representatives elements:
* ```patches_path``` describes location of patches extraction results.
* ```sampled_path``` describes sampled patches save location.
* ```--feature_type``` designs the features to use to perform the clustering: either features extracted from a Resnet18 
model pretrained on ImageNet, or rgb histogram.
* ```--testing_set``` to sample patch from the testing set (else, sample patch from training set).

The ground truth masks are available [here](./Warwick_test_set).
## BCDataset

Set of 1338 640x640 IHC images stained for Ki67 with cell center annotations - 
see [here](https://sites.google.com/view/bcdataset) for more details and retrieve the data.

The results were assessed qualitatively by visualizing models predictions and cell centers on original images. As for 
DeepLIIF, positive and negative cells annotations were merged together to discard classification considerations.


