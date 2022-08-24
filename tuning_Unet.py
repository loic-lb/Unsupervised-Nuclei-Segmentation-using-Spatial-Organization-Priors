import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical
from datasets.utils import DeepLIIFImgMaskDataset, create_color_transform, create_data_ihc_aug
from models import CustomUnet, export_optimal_hyperparameters
from performance import dice, f1_m, precision_m, recall_m


def create_model(ngf, dropout, learning_rate, decay):
    model = CustomUnet(ngf=ngf, dropout_value=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=decay)
    return model, optimizer


def dice_coef_loss(pred, targs):
    return 1 - dice(pred, targs)


def main():
    parser = argparse.ArgumentParser(description='Tuning Unet model before training using Bayesian Optimisation')

    parser.add_argument('dataihcroot_images', type=str, default=None,
                        help='IHC images root path')
    parser.add_argument('--dataset_name', type=str, choices=["deepliif", "warwick"], default="deepliif",
                        help='Choose between deepliif (KI67) or warwick (HER2) dataset (not implemented yet)')
    parser.add_argument('--exp_save_path', type=str, default="./",
                        help='Result save path')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs for training each model (default: 10)')
    parser.add_argument('--n_calls', type=int, default=50,
                        help='Number of iterations for gaussian process algorithm (default: 50)')
    parser.add_argument('--batch_size_validation', type=int, default=34,
                        help='Batch size for validation (default: 34)')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of worker nodes (default: 5)')
    args = parser.parse_args()

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
                                                                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                        (0.5, 0.5,
                                                                                                         0.5))]),
                                                 mask_transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    dim_ngf = Categorical([64, 128], name="ngf")
    dim_dropout = Real(0.3, 0.5, name="dropout")
    dim_learning_rate = Real(low=1e-5, high=1e-2, prior="log-uniform", name="learning_rate")
    dim_decay = Real(low=1e-10, high=1e-3, prior="log-uniform", name="decay")
    dim_batch_size = Categorical([10, 30, 60, 120, 140], name="batch_size")

    dimensions = [dim_ngf,
                  dim_dropout,
                  dim_learning_rate,
                  dim_decay,
                  dim_batch_size]

    parameters_list = ["ngf", "dropout", "learning_rate", "decay", "batch_size"]
    default_parameters = [64, 0.5, 1e-4, 1e-10, 30]

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    color_transform = create_color_transform(1, 0.75)
    criterion_consistency = torch.nn.L1Loss()

    @use_named_args(dimensions=dimensions)
    def fit_opt(ngf, dropout, learning_rate, decay, batch_size):
        best_score = 0
        model, optimizer = create_model(ngf, dropout, learning_rate, decay)
        model = model.to(device)
        dataloader_ihc_train = torch.utils.data.DataLoader(dataset_ihc_train, batch_size=int(batch_size),
                                                           shuffle=True, num_workers=args.workers)
        dataloader_ihc_val = torch.utils.data.DataLoader(dataset_ihc_val, batch_size=int(args.batch_size_validation),
                                                         shuffle=False, num_workers=args.workers)
        for epoch in range(args.num_epochs):
            accuracies = []
            f1_scores = []
            precisions = []
            recalls = []
            model.train()
            for data, target in dataloader_ihc_train:
                data_aug = create_data_ihc_aug(data, color_transform).to(device)
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                pred = model(data)
                pred_aug = model(data_aug)
                loss_consistency = criterion_consistency(pred, pred_aug)
                err = dice_coef_loss(pred, target) + loss_consistency
                err.backward()
                optimizer.step()
            model.eval()
            for data, target in dataloader_ihc_val:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                pred = (pred > 0.5).float()
                accuracies.append(((pred == target).sum(axis=(1, 2, 3)) / target[0].numel()).mean().cpu().item())
                f1_scores.append(f1_m(pred, target).cpu().item())
                precisions.append(precision_m(pred, target).cpu().item())
                recalls.append(recall_m(pred, target).cpu().item())

            score = np.mean(f1_scores)
            print(f"Averaged validation : acc = {np.mean(accuracies)}, F1 score = {score}, "
                  f"precision = {np.mean(precisions)}, recall = {np.mean(recalls)}")
            if score > best_score:
                best_score = score
        return -best_score

    print(f"Tuning parameters for {args.n_calls} iterations...")
    gp_result = gp_minimize(func=fit_opt,
                            dimensions=dimensions,
                            acq_func="EI",
                            n_calls=args.n_calls,
                            noise="gaussian",
                            n_jobs=-1,
                            x0=default_parameters,
                            verbose=True,
                            callback=lambda x: torch.cuda.empty_cache())

    print(f"Optimal set of parameters found at iteration {np.argmin(gp_result.func_vals)}, now exporting ...")
    export_optimal_hyperparameters(gp_result.x, parameters_list, args.exp_save_path)


if __name__ == '__main__':
    main()
