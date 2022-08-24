# Unsupervised Nuclei Segmentation using Spatial Organization Priors - Existing methods

## Nuclick

Method proposed by N.A. Koohbanani, et al. - see [here](https://arxiv.org/abs/2005.14511) for full paper on arxiv.

The code of Nuclick can be retrieved [here](https://github.com/navidstuv/NuClick). The provided code can then
be used by placing the files at the root of the Nuclick project to reproduce the results of this paper.

## Stardist

Method proposed by U. Schmidt, et al. - see [here](https://arxiv.org/abs/1806.03535) for full paper on arxiv.

The scripts have been developped for the Qupath version of this algorithm (see [here](https://github.com/qupath/qupath-extension-stardist) for installation). They should
be used in the following order:

* ```save_annotation``` saves an annotation. A square should first be drawn with Qupath toolbox englobing the whole patch (for one image).
* ```reproduced_saved_annotations``` reproduces a saved annotations. Using this script, the square annotation can be applied to all the images
  (without drawing it each time by hand)
* ```script_stardist``` performs segmentation using Stardist algorithm (requiring prior installation).
* ```script_save_masks``` saves nuclei masks under binary png images.

## Thresholding

Masks produced from grayscale transformation were used for DeepLIIF and from the hematoxylin deconvolved images for Warwick.

