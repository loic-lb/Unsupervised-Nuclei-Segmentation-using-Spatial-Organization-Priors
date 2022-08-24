# Unsupervised Nuclei Segmentation using Spatial Organization Priors

Code for training and benchmarking unsupervised and supervised methods on several dataset of H-DAB stained immunohistochemistry images for nuclei segmentation. 
* ```train_GAN.py``` trains the proposed cycle GAN model for a given IHC dataset and ground truth masks of 
nuclei segmentation from any source. 
* ```train_Unet.py``` trains the generator of the proposed method (based on an Unet architecture) in a supervised
fashion 
* ```tuning_Unet.py``` retrieves best set of hyperparameters for training the generator in a supervised fashion.
* ```test.py``` computes every metrics for all the benchmarked methods.

## Installation
The packages needed to run the code are provided in ```requirements.txt```. The environment can be recreated using:

```
conda create --name <env_name> --file requirements.txt
```

## Data

4 datasets were used in this paper:
* **Deepliif**: Set of 1667 registered 512x512 IHC images built to train the DEEPLIIF algorithm; only the original Ki67 stained 
and the nuclei segmentation images were used - see [here](https://zenodo.org/record/4751737#.YlWRiN86-Ul) for the data.
* **Warwick**: Set of 86 HER2 stained WSIs - see [here](https://warwick.ac.uk/fac/cross_fac/tia/data/her2contest) for more details and request access to the data.
* **BCDataset**: Set of 1338 640x640 IHC images stained for Ki67 with cell center annotations - see [here](https://sites.google.com/view/bcdataset) for more details and retrieve the data.
* **Pannuke**: Set of more than 7000 256x256 H&E stained images with corresponding nuclei segmentation - see [here](https://jgamper.github.io/PanNukeDataset/) for more details and retrieve the data.

Please see the directory [dataset](datasets/README.md) for more information about how was used each dataset.

## Proposed method
```train_GAN.py``` requires 3 arguments to train the GAN model: 
* ```dataihcroot_images``` describes the location of H-DAB stained training data.
* ```dataroot_masks``` describes the location of nuclei segmentation masks from another database (Pannuke here).
* ```dataset_name``` specifies which dataset is used among Deepliif and Warwick (only these 2 were used in this paper for training).

and some optional arguments are available to specify the number of epochs, batch_size, number of worker nodes, ...

Example command line:

```
python train.py /path/to/DeepLiif/ /path/to/PanNuke/ deepliif
```

```test.py``` requires 4 arguments to compute performance metrics:
* ```data_path``` describes the location of H-DAB stained testing data (H-DAB stained images and their corresponding nuclei segmentation)
* ```method``` specifies the method to benchmark ("GAN", "Unet", "Stardist", "Nuclick", "Thresholding")
*  ```dataset_name``` specifies which dataset is used among Deepliif and Warwick (only these 2 were used in this paper to compute metrics).
* Depending on the chosen method, ```--benchmark_path``` or ```--checkpoint_path``` must be provided:
  * ```--checkpoint_path``` describes the location of best saved models for "GAN" and "Unet" methods (must contain model checkpoints and 
training losses for "GAN" and optimal hyperparameters for "Unet")
  * ```--benchmark_path``` describes the location of the mask computed with existing methods ("Stardist", "Nuclick", "Thresholding")

Example command line:

```
python test.py /path/to/DeepLiif/ GAN deepliif --checkpoint_path /path/to/training_results --batch_size=46
```

```
python test.py /path/to/Warwick/ Nuclick warwick --benchmark_path /path/to/Nuclick_results
```

The pretrained models can be retrieved [here](https://drive.google.com/drive/folders/1qSBd6_m5omPAGijiDa2BRhZDAtxcaRxL?usp=sharing).
## Existing methods

3 methods were used to benchmark the proposed method:
* Qupath software with Stardist plugin. The scripts are available [here](existing_methods/stardist).
* Nuclick algorithm to extract mask segmentation from nuclei center annotations. The code of Nuclick can be retrieved [here](https://github.com/navidstuv/NuClick), and the code for this paper [here](existing_methods/nuclick).
* A baseline thresholding approach, using color deconvolution for membranous staining. The code is available [here](existing_methods/thresholding)

## Acknowledgement

Many thanks to Zhu.et.al for providing the code of their cycleGAN architectures - see [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference

If you find this code useful in your research then please cite:

```
@inproceedings{lebescond:hal-03644463,
  TITLE = {{Unsupervised Nuclei Segmentation using Spatial Organization Priors}},
  AUTHOR = {Le Bescond, Lo{\"i}c and Lerousseau, Marvin and Garberis, Ingrid and Andr{\'e}, Fabrice and Christodoulidis, Stergios and Vakalopoulou, Maria and Talbot, Hugues},
  BOOKTITLE = {{MICCAI 2022 - 25th International Conference on Medical Image Computing and Computer Assisted Intervention}},
  YEAR = {2022},
}
```

## License

This code is MIT licensed, as found in the LICENSE file.
