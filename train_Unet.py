import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from time import sleep
from datasets.utils import DeepLIIFImgMaskDataset, create_color_transform, create_data_ihc_aug
from models import CustomUnet, load_optimal_hyperparameters
from performance import dice, f1_m, precision_m, recall_m


def dice_coef_loss(pred, targs):
    return 1 - dice(pred, targs)


def main():
    parser = argparse.ArgumentParser(
        description='Train Unet Model for IHC nuclei segmentation for benchmarking generator'
                    ' trained in a supervised setting')
    parser.add_argument('dataihcroot_images', type=str, default=None,
                        help='IHC images root path')
    parser.add_argument('--dataset_name', type=str, choices=["deepliif", "warwick"], default="deepliif",
                        help='Choose between deepliif (KI67) or warwick (HER2) dataset (not implemented)')
    parser.add_argument('--exp_save_path', type=str, default="./",
                        help='Result save path (must contain optimal set of parameters produced by tuning_Unet.py)')
    parser.add_argument('--num_epochs', type=int, default=600,
                        help='Number of epochs for training (default: 200)')
    parser.add_argument('--batch_size_validation', type=int, default=34,
                        help='Batch size for validation (default: 34)')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of worker nodes (default: 5)')
    args = parser.parse_args()

    optimal_parameters_file_path = os.path.join(args.exp_save_path, "optimal_parameters.pickle")
    if not os.path.isfile(optimal_parameters_file_path):
        raise ValueError("Please provide a valid path for experiment results")

    if args.dataset_name == "deepliif":
        dataset_ihc_train_path = os.path.join(args.dataihcroot_images, "DeepLIIF_Training_Set")
        dataset_ihc_val_path = os.path.join(args.dataihcroot_images, "DeepLIIF_Validation_Set")
        dataset_ihc_train = DeepLIIFImgMaskDataset(dataset_ihc_train_path,
                                                   img_transform=transforms.Compose([transforms.RandomChoice(
                                                       [transforms.RandomRotation((0, 0)),
                                                        transforms.RandomRotation((90, 90)),
                                                        transforms.RandomRotation((2 * 90, 2 * 90)),
                                                        transforms.RandomRotation((3 * 90, 3 * 90))]),
                                                       transforms.RandomVerticalFlip(),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.RandomCrop((256, 256)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                                            (0.5, 0.5, 0.5))]),
                                                   mask_transform=transforms.Compose([transforms.RandomChoice(
                                                       [transforms.RandomRotation((0, 0)),
                                                        transforms.RandomRotation((90, 90)),
                                                        transforms.RandomRotation((2 * 90, 2 * 90)),
                                                        transforms.RandomRotation((3 * 90, 3 * 90))]),
                                                       transforms.RandomVerticalFlip(),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.RandomCrop((256, 256)),
                                                       transforms.ToTensor()])
                                                   )
        dataset_ihc_val = DeepLIIFImgMaskDataset(dataset_ihc_val_path,
                                                 img_transform=transforms.Compose([transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5,
                                                                                        0.5))]),
                                                 mask_transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    optimal_parameters = load_optimal_hyperparameters(optimal_parameters_file_path)

    dataloader_ihc_train = torch.utils.data.DataLoader(dataset_ihc_train,
                                                       batch_size=int(optimal_parameters["batch_size"]),
                                                       shuffle=True, num_workers=args.workers)
    dataloader_ihc_val = torch.utils.data.DataLoader(dataset_ihc_val, batch_size=args.batch_size_validation,
                                                     shuffle=False, num_workers=args.workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    color_transform = create_color_transform(1, 0.75)
    criterion_consistency = torch.nn.L1Loss()

    model = CustomUnet(ngf=int(optimal_parameters["ngf"]), dropout_value=optimal_parameters["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=optimal_parameters["learning_rate"],
                                 betas=(0.9, 0.999), weight_decay=optimal_parameters["decay"])
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_perf = []
    val_perf = []
    best_score = 0

    for epoch in range(args.num_epochs):
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        dice_scores = []
        train_loss = 0
        cumul_perf = 0
        with tqdm(dataloader_ihc_train, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}, training :")
            for data, target in tepoch:
                data_aug = create_data_ihc_aug(data, color_transform).to(device)
                data, target = data.to(device), target.to(device)
                model.train()
                optimizer.zero_grad()
                pred = model(data)
                pred_aug = model(data_aug)
                loss_consistency = criterion_consistency(pred, pred_aug)
                err_train = dice_coef_loss(pred, target) + loss_consistency
                train_loss += err_train.item()
                err_train.backward()
                optimizer.step()
                pred = (pred > 0.5).float()
                perf = f1_m(pred, target).cpu().item()
                cumul_perf += perf
                tepoch.set_postfix(loss=err_train.item(), f1_score=perf)
                sleep(0.1)
        train_losses.append(train_loss / len(dataloader_ihc_train))
        train_perf.append(cumul_perf / len(dataloader_ihc_train))

        val_loss = 0
        cumul_perf = 0
        with torch.no_grad():
            for data, target in dataloader_ihc_val:
                data_aug = create_data_ihc_aug(data, color_transform).to(device)
                data, target = data.to(device), target.to(device)
                model.eval()
                pred = model(data)
                pred_aug = model(data_aug)
                loss_consistency = criterion_consistency(pred, pred_aug)
                err_val = dice_coef_loss(pred, target) + loss_consistency
                val_loss += err_val.item()
                pred = (pred > 0.5).float()
                perf = f1_m(pred, target).cpu().item()
                cumul_perf += perf
                accuracies.append(((pred == target).sum(axis=(1, 2, 3)) / target[0].numel()).mean().cpu().item())
                f1_scores.append(f1_m(pred, target).cpu().item())
                precisions.append(precision_m(pred, target).cpu().item())
                recalls.append(recall_m(pred, target).cpu().item())
                dice_scores.append(dice(pred, target).cpu().item())

        val_losses.append(val_loss / len(dataloader_ihc_val))
        val_perf.append(cumul_perf / len(dataloader_ihc_val))
        score = np.mean(f1_scores)
        print(f"Averaged validation : acc = {np.mean(accuracies)}, Dice = {np.mean(dice_scores)}, "
              f"F1 score = {score}, precision = {np.mean(precisions)}, recall = {np.mean(recalls)}")
        if score > best_score:
            best_score = score
            best_perf = {"acc": np.mean(accuracies), "dice": np.mean(dice_scores), "f1_score": score,
                         "precision": np.mean(precisions), "recall": np.mean(recalls)}
            PATH = os.path.join(args.exp_save_path, "best_model.pt")
            torch.save(model.state_dict(), PATH)

    print(f"Best performance on validation set: {best_perf}")


if __name__ == '__main__':
    main()
