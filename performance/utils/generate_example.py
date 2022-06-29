import os
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path


def save_generated_examples(path, grid, num_epoch=None, suffix=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    im = np.transpose(grid.numpy(), (1, 2, 0))
    im = Image.fromarray((im * 255).astype(np.uint8))
    if suffix and type(num_epoch) == int:
        img_name = os.path.join(path, f"generated_{suffix}_epoch_{num_epoch}.png")
    elif type(num_epoch) == int and not suffix:
        img_name = os.path.join(path, f"generated_example_epoch_{num_epoch}.png")
    elif suffix and not type(num_epoch) == int:
        img_name = os.path.join(path, f"generated_example_{suffix}.png")
    else:
        img_name = os.path.join(path, f"generated_example.png")
    im.save(img_name)


def generate_examples(save_path, netG_segmentation, netG_reconstruction, fixed_images, epoch):
    fake = netG_segmentation(fixed_images).detach()
    grid_fake_segmentation = vutils.make_grid((fake > 0.5).float().cpu(), padding=2, normalize=False)
    save_generated_examples(os.path.join(save_path, "segmentation"), grid_fake_segmentation, epoch, "segmentation")
    input_fake = netG_reconstruction(fake).detach()
    grid_fake_reconstruction = vutils.make_grid(input_fake.cpu(), padding=2, normalize=True)
    save_generated_examples(os.path.join(save_path, "reconstruction"), grid_fake_reconstruction, epoch,
                            "reconstruction")
