import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from datasets.utils import DeepLIIFImgMaskDataset, WarwickImgMaskDataset, DeepLIIFBenchmarkDataset, \
    WarwickBenchmarkDataset
from models import Generator, CustomUnet, load_optimal_hyperparameters, load_pickle_file
from postprocessing import compute_morphological_operations
from performance import dice_by_sample, dice, hausdorff, f1_m, precision_m, recall_m, obj_metric, aji_score


def main():
    parser = argparse.ArgumentParser(description='Train GAN Model for IHC nuclei segmentation from spatial organization'
                                                 'prior')
    parser.add_argument('data_path', type=str, default=None,
                        help='IHC images and/or ground truth masks root path')
    parser.add_argument('method', type=str, choices=["GAN", "Unet", "Stardist", "Nuclick", "Thresholding"],
                        default=None, help='Choose the method to evaluate among ["GAN", "Unet", "Stardist",'
                                           ' "Nuclick", "Thresholding"]')
    parser.add_argument('dataset_name', type=str, choices=["deepliif", "warwick"], default=None,
                        help='Choose between deepliif, bcdataset (Ki67) or warwick (HER2) dataset')
    parser.add_argument('--benchmark_path', type=str, default=None,
                        help='Predicted masks path with benchmark methods ("Stardist", "Nuclick", "Thresholding")')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained_models",
                        help='Best model .pt file path (should also contain G_losses_GAN.pickle for GAN method, '
                             'and optimal_parameters.pickle for Unet method)')
    parser.add_argument('--start_epoch_checkpoint', type=int, default=250,
                        help='Number of epoch to skip before assessing the minimum of G_losses_GAN (for GAN method)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of worker nodes (default: 5)')
    args = parser.parse_args()

    if args.method in ["GAN", "Unet"] and args.checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path when evaluating GAN or Unet approach")
    elif args.method in ["Stardist", "Nuclick", "Thresholding"] and args.benchmark_path is None:
        raise ValueError("Please provide a benchmark path when evaluating existing methods")

    if args.method in ["GAN", "Unet"]:
        if args.dataset_name == "deepliif":
            dataset_ihc_test_path = os.path.join(args.data_path, "DeepLIIF_Testing_Set")
            dataset_ihc_test = DeepLIIFImgMaskDataset(dataset_ihc_test_path,
                                                      img_transform=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize(
                                                                                            (0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5,
                                                                                             0.5))]),
                                                      mask_transform=transforms.ToTensor())
        elif args.dataset_name == "warwick":
            dataset_ihc_test_images_path = os.path.join(args.data_path, "images")
            dataset_ihc_test_masks_path = os.path.join(args.data_path, "ground_truths_corrected")
            dataset_ihc_test = WarwickImgMaskDataset(dataset_ihc_test_images_path, dataset_ihc_test_masks_path,
                                                     img_transform=transforms.Compose([transforms.ToTensor(),
                                                                                       transforms.Normalize(
                                                                                           (0.5, 0.5, 0.5),
                                                                                           (0.5, 0.5,
                                                                                            0.5))]),
                                                     mask_transform=transforms.ToTensor())
        else:
            raise NotImplementedError
    else:
        if args.dataset_name == "deepliif":
            dataset_ihc_test_path = os.path.join(args.data_path, "DeepLIIF_Testing_Set")
            dataset_ihc_test = DeepLIIFBenchmarkDataset(args.benchmark_path, dataset_ihc_test_path,
                                                        mask_transform=transforms.ToTensor(),
                                                        gt_transform=transforms.ToTensor())
        elif args.dataset_name == "warwick":
            dataset_ihc_test_masks_path = os.path.join(args.data_path, "ground_truths")
            dataset_ihc_test = WarwickBenchmarkDataset(args.benchmark_path, dataset_ihc_test_masks_path,
                                                       mask_transform=transforms.ToTensor(),
                                                       gt_transform=transforms.ToTensor())
        else:
            raise NotImplementedError

    print(f"Testing {args.method} method on {len(dataset_ihc_test)} data...")
    if args.batch_size:
        dataloader_ihc_test = DataLoader(dataset_ihc_test, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers)
    else:
        dataloader_ihc_test = DataLoader(dataset_ihc_test, batch_size=len(dataset_ihc_test), shuffle=False,
                                         num_workers=args.workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.method == "GAN":
        if os.path.isfile(os.path.join(args.checkpoint_path, "G_losses_system.pickle")):
            G_losses_system = load_pickle_file(os.path.join(args.checkpoint_path, "G_losses_GAN.pickle"))
            start_idx = min(args.start_epoch_checkpoint, len(G_losses_system) - 1)
            best_idx = np.argmin(G_losses_system[start_idx:]) + start_idx
            checkpoint = torch.load(os.path.join(args.checkpoint_path, f"model_epoch{best_idx}.pt"))
            print(f"Best model found for epoch {best_idx}.")
        else:
            checkpoint = torch.load(os.path.join(args.checkpoint_path, f"proposed_{args.dataset_name}.pt"))
        model = Generator().to(device)
        model.load_state_dict(checkpoint["generator_ihc_to_mask_state_dict"])
        model.eval()
    elif args.method == "Unet":
        optimal_parameters = load_optimal_hyperparameters(os.path.join(args.checkpoint_path,
                                                                       "optimal_parameters.pickle"))
        ngf = optimal_parameters["ngf"]
        dropout_value = optimal_parameters["dropout_value"]
        checkpoint = torch.load(os.path.join(args.checkpoint_path, "best_model.pt"))
        model = CustomUnet(ngf=ngf, dropout_value=dropout_value).to(device)
        model.load_state_dict(checkpoint)
        model.eval()

    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    dice_scores = []
    dice_object = []
    aji_scores = []
    hausdorff_scores = []

    already_labelled = args.method == "Nuclick"
    watershed = args.method not in ["Stardist", "Nuclick"]
    with torch.no_grad():
        for data, target in dataloader_ihc_test:
            if args.method in ["GAN", "Unet"]:
                data = data.to(device)
                pred = model(data)
                pred = (pred > 0.5).float().cpu()
                all_pred = []
                for img in pred:
                    img_postprocessed = compute_morphological_operations(img.squeeze().numpy(),
                                                                         erosion=args.dataset_name == "warwick")
                    all_pred.append(torch.tensor(img_postprocessed).unsqueeze().float())
                pred = torch.stack(all_pred)
                pred_obj = pred
            elif args.method == "Thresholding":
                pred = data
                all_pred = []
                for img in pred:
                    img_postprocessed = compute_morphological_operations(img.squeeze().numpy(),
                                                                         erosion=args.dataset_name == "warwick")
                    all_pred.append(torch.tensor(img_postprocessed).unsqueeze().float())
                pred = torch.stack(all_pred)
                pred_obj = pred
            elif args.method == "Nuclick":
                pred = (data > 0.).float()
                pred_obj = data
            else:
                pred = data
                pred_obj = data
            accuracies.append(
                np.mean([balanced_accuracy_score(target[i, ...].flatten(), pred[i, ...].flatten()) for i in
                         range(target.shape[0])]))
            f1_scores.append(f1_m(pred, target).item())
            precisions.append(precision_m(pred, target).item())
            recalls.append(recall_m(pred, target).item())
            dice_scores.append(dice_by_sample(pred, target).item())
            dice_object.append(obj_metric(pred_obj, target, dice, watershed, already_labelled))
            hausdorff_scores.append(obj_metric(pred_obj, target, hausdorff, watershed, already_labelled))
            aji_scores.append(aji_score(pred_obj, target, watershed, already_labelled))
    print(f"Averaged test : acc = {np.mean(accuracies)}, Dice = {np.mean(dice_scores)}, "
          f"Dice object = {np.mean(dice_object)} F1 score = {np.mean(f1_scores)}, "
          f"precision = {np.mean(precisions)}, recall = {np.mean(recalls)},"
          f"AJI = {np.mean(aji_scores)}, Hausdorff = {np.mean(hausdorff_scores)}")


if __name__ == '__main__':
    main()
