import os
import argparse
import itertools
import torch
import pickle
import torchvision.utils as vutils
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.utils import DeepLIIFImgDataset, WarwickImgDataset, PannukeMasksDataset, create_color_transform, \
    create_data_ihc_aug
from models import Generator, Discriminator, decompose_stain
from performance import generate_examples, save_generated_examples


def main():
    parser = argparse.ArgumentParser(description='Train GAN Model for IHC nuclei segmentation from spatial organization'
                                                 ' prior')
    parser.add_argument('dataihcroot_images', type=str, default=None,
                        help='IHC images root path')
    parser.add_argument('dataroot_masks', type=str, default=None,
                        help='Pannuke H&E masks root path')
    parser.add_argument('dataset_name', type=str, choices=["deepliif", "warwick"], default=None,
                        help='Choose between deepliif (KI67) or warwick (HER2) dataset')
    parser.add_argument('--exp_save_path', type=str, default="./",
                        help='Result save path')
    parser.add_argument('--num_epochs', type=int, default=600,
                        help='Number of epochs for training (default: 600)')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch_size for training (default: 30)')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of worker nodes (default: 5)')
    parser.add_argument('--remove_loss', type=str, choices=["cycle", "consistency", "r=1"], default=None,
                        help='Loss to remove (for ablation study)')
    args = parser.parse_args()

    Path(args.exp_save_path).mkdir(parents=True, exist_ok=True)
    if args.dataset_name == "deepliif":
        dataset_ihc_train_path = os.path.join(args.dataihcroot_images, "DeepLIIF_Training_Set")
        dataset_ihc = DeepLIIFImgDataset(dataset_ihc_train_path,
                                         transform=transforms.Compose([transforms.RandomChoice(
                                             [transforms.RandomRotation((0, 0)), transforms.RandomRotation((90, 90)),
                                              transforms.RandomRotation((2 * 90, 2 * 90)),
                                              transforms.RandomRotation((3 * 90, 3 * 90))]),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomChoice(
                                                 [transforms.RandomCrop((256, 256)),
                                                  transforms.Resize((256, 256))]),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))]))
    elif args.dataset_name == "warwick":
        dataset_ihc_train_path = os.path.join(args.dataihcroot_images, "Warwick_Training_Set")
        dataset_ihc = WarwickImgDataset(dataset_ihc_train_path,
                                        transform=transforms.Compose([transforms.RandomChoice(
                                            [transforms.RandomRotation((0, 0)), transforms.RandomRotation((90, 90)),
                                             transforms.RandomRotation((2 * 90, 2 * 90)),
                                             transforms.RandomRotation((3 * 90, 3 * 90))]),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomChoice(
                                                [transforms.RandomCrop((256, 256)),
                                                 transforms.Resize((256, 256))]),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5))]))
    else:
        raise NotImplementedError

    dataset_masks = PannukeMasksDataset(args.dataroot_masks, transform=transforms.Compose([transforms.RandomChoice(
        [transforms.RandomRotation((0, 0)), transforms.RandomRotation((90, 90)),
         transforms.RandomRotation((2 * 90, 2 * 90)), transforms.RandomRotation((3 * 90, 3 * 90))]),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]))

    dataloader_ihc = DataLoader(dataset_ihc, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloader_masks = DataLoader(dataset_masks, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    fixed_images = next(iter(dataloader_ihc)).to(device)
    save_generated_examples(args.exp_save_path, vutils.make_grid(fixed_images.cpu(), padding=2, normalize=True), suffix="input_images")
    color_transform = create_color_transform(1, 0.75)

    if args.remove_loss == "r=1":
        netG_ihc_to_mask = Generator(r=1).to(device)
    else:
        netG_ihc_to_mask = Generator().to(device)

    netG_mask_to_ihc = Generator(input_nc=1, output_nc=3, segmentation=False).to(device)
    netD_ihc = Discriminator().to(device)
    netD_mask = Discriminator(input_nc=1).to(device)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(netG_ihc_to_mask.parameters(), netG_mask_to_ihc.parameters()),
                                   lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_ihc = torch.optim.Adam(netD_ihc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_mask = torch.optim.Adam(netD_mask.parameters(), lr=2e-4, betas=(0.5, 0.999))

    real_label = 1.
    fake_label = 0.

    G_losses_system = []
    G_losses_ihc_to_mask = []
    G_losses_mask_to_ihc = []
    G_losses_consistency = []
    G_losses_cycle = []
    D_losses_ihc = []
    D_losses_mask = []

    print("Starting Training Loop...")

    for epoch in range(args.num_epochs):

        dataloader_iterator = iter(dataloader_masks)

        loss_G_epoch = 0
        loss_G_ihc_to_mask_epoch = 0
        loss_G_mask_to_ihc_epoch = 0
        loss_cycle_epoch = 0
        loss_consistency_epoch = 0
        loss_D_ihc_epoch = 0
        loss_D_mask_epoch = 0

        for i, data_ihc in enumerate(dataloader_ihc):

            try:
                data_mask = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader_masks)
                data_mask = next(dataloader_iterator)

            data_ihc_aug = create_data_ihc_aug(data_ihc, color_transform).to(device)

            if args.dataset_name == "warwick":
                data_ihc_h = decompose_stain(data_ihc)[0].to(device)

            data_ihc = data_ihc.to(device)
            data_mask = data_mask.to(device)

            netG_ihc_to_mask.train()
            netG_mask_to_ihc.train()

            # Generator training ...

            optimizer_G.zero_grad()

            fake_ihc_to_mask = netG_ihc_to_mask(data_ihc)
            fake_ihc_to_mask_aug = netG_ihc_to_mask(data_ihc_aug)
            fake_mask_to_ihc = netG_mask_to_ihc(data_mask)

            # Consistency loss

            loss_consistency = 1. * criterion_identity(fake_ihc_to_mask, fake_ihc_to_mask_aug)

            true_ihc = netD_ihc(fake_mask_to_ihc)
            true_mask = netD_mask(fake_ihc_to_mask)

            loss_GAN_ihc_to_mask = criterion_GAN(true_mask, torch.tensor(real_label).expand_as(true_mask).to(device))
            loss_GAN_mask_to_ihc = criterion_GAN(true_ihc, torch.tensor(real_label).expand_as(true_ihc).to(device))

            # Generator loss

            loss_GAN = 0.5 * (loss_GAN_ihc_to_mask + loss_GAN_mask_to_ihc)

            # Cycle loss

            recov_ihc = netG_mask_to_ihc(fake_ihc_to_mask)
            if args.dataset_name == "warwick":
                loss_cycle_ihc = criterion_cycle(recov_ihc, data_ihc_h)
            else:
                loss_cycle_ihc = criterion_cycle(recov_ihc, data_ihc)
            recov_mask = netG_ihc_to_mask(fake_mask_to_ihc)
            loss_cycle_mask = criterion_cycle(recov_mask, data_mask)

            loss_cycle = 10. * (loss_cycle_ihc + loss_cycle_mask) / 2

            if args.remove_loss == "cycle":
                loss_G = loss_GAN + loss_consistency
            elif args.remove_loss == "consistency":
                loss_G = loss_GAN + loss_cycle
            else:
                loss_G = loss_GAN + loss_cycle + loss_consistency

            loss_G.backward()
            optimizer_G.step()

            # Discriminator training ...

            optimizer_D_ihc.zero_grad()

            if args.dataset_name == "warwick":
                output = netD_ihc(data_ihc_h)
            else:
                output = netD_ihc(data_ihc)
            loss_real = criterion_GAN(output, torch.tensor(real_label).expand_as(output).to(device))
            output = netD_ihc(fake_mask_to_ihc.detach())
            loss_fake = criterion_GAN(output, torch.tensor(fake_label).expand_as(output).to(device))
            loss_D_ihc = (loss_real + loss_fake) / 2

            loss_D_ihc.backward()
            optimizer_D_ihc.step()

            optimizer_D_mask.zero_grad()

            output = netD_mask(data_mask)
            loss_real = criterion_GAN(output, torch.tensor(real_label).expand_as(output).to(device))
            output = netD_mask(fake_ihc_to_mask.detach())
            loss_fake = criterion_GAN(output, torch.tensor(fake_label).expand_as(output).to(device))
            loss_D_mask = (loss_real + loss_fake) / 2

            loss_D_mask.backward()
            optimizer_D_mask.step()

            if i % 5 == 0:
                print(
                    f'[{epoch}/{args.num_epochs}][{i:02d}/{len(dataloader_ihc)}]\tLoss_D_ihc: {loss_D_ihc.item():.3f}'
                    f'\tLoss_D_mask: {loss_D_mask.item():.3f}\tLoss_G_ihc_to_mask: {loss_GAN_ihc_to_mask.item():.3f}'
                    f'\tLoss_G_mask_to_ihc: {loss_GAN_mask_to_ihc.item():.3f}\tLoss_G_cycle :{loss_cycle:.3f}'
                    f'\tLoss_G_consistency : {loss_consistency.item():.3f}\tLoss_G : {loss_G.item():.3f}')

            loss_G_epoch += loss_G.item()
            loss_G_ihc_to_mask_epoch += loss_GAN_ihc_to_mask.item()
            loss_G_mask_to_ihc_epoch += loss_GAN_mask_to_ihc.item()
            loss_cycle_epoch += loss_cycle.item()
            loss_consistency_epoch += loss_consistency.item()
            loss_D_ihc_epoch += loss_D_ihc.item()
            loss_D_mask_epoch += loss_D_mask.item()

        loss_G_epoch /= len(dataloader_ihc)
        loss_G_ihc_to_mask_epoch /= len(dataloader_ihc)
        loss_G_mask_to_ihc_epoch /= len(dataloader_ihc)
        loss_cycle_epoch /= len(dataloader_ihc)
        loss_consistency_epoch /= len(dataloader_ihc)
        loss_D_ihc_epoch /= len(dataloader_ihc)
        loss_D_mask_epoch /= len(dataloader_ihc)

        G_losses_system.append(loss_G_epoch)
        G_losses_ihc_to_mask.append(loss_G_ihc_to_mask_epoch)
        G_losses_mask_to_ihc.append(loss_G_ihc_to_mask_epoch)
        G_losses_cycle.append(loss_cycle_epoch)
        G_losses_consistency.append(loss_consistency_epoch)
        D_losses_ihc.append(loss_D_ihc_epoch)
        D_losses_mask.append(loss_D_mask_epoch)

        with torch.no_grad():
            model_save_path = os.path.join(args.exp_save_path, f"model_epoch{epoch}.pt")
            torch.save({
                'generator_ihc_to_mask_state_dict': netG_ihc_to_mask.state_dict(),
                'discriminator_ihc_state_dict': netD_ihc.state_dict(),
                'generator_mask_to_ihc_state_dict': netG_mask_to_ihc.state_dict(),
                'discriminator_mask_state_dict': netD_mask.state_dict(),
            }, model_save_path)
            netG_ihc_to_mask.eval()
            netG_mask_to_ihc.eval()
            generate_examples(args.exp_save_path, netG_ihc_to_mask, netG_mask_to_ihc, fixed_images, epoch)

    with open(os.path.join(args.exp_save_path, "G_losses_system.pickle"), "wb") as fp:
        pickle.dump(G_losses_system, fp)
    with open(os.path.join(args.exp_save_path, "G_losses_ihc_to_mask.pickle"), "wb") as fp:
        pickle.dump(G_losses_ihc_to_mask, fp)
    with open(os.path.join(args.exp_save_path, "G_losses_mask_to_ihc.pickle"), "wb") as fp:
        pickle.dump(G_losses_mask_to_ihc, fp)
    with open(os.path.join(args.exp_save_path, "G_losses_cycle.pickle"), "wb") as fp:
        pickle.dump(G_losses_cycle, fp)
    with open(os.path.join(args.exp_save_path, "G_losses_consistency.pickle"), "wb") as fp:
        pickle.dump(G_losses_consistency, fp)
    with open(os.path.join(args.exp_save_path, "D_losses_ihc.pickle"), "wb") as fp:
        pickle.dump(D_losses_ihc, fp)
    with open(os.path.join(args.exp_save_path, "D_losses_mask.pickle"), "wb") as fp:
        pickle.dump(D_losses_mask, fp)


if __name__ == '__main__':
    main()
