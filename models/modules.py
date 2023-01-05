import torch
import torch.nn as nn
from .architectures import define_D, define_G


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=64, netG="unet_256", use_dropout=True, dropout_value=0.5,
                 norm='instance', r=60, segmentation=True):
        super(Generator, self).__init__()
        self.unet_model = define_G(input_nc=input_nc, output_nc=output_nc, ngf=ngf, netG=netG, use_dropout=use_dropout,
                                   dropout_value=dropout_value, norm=norm, init_type="normal", init_gain=0.02,
                                   bias_last_conv=True)
        # Compression of the sigmoid.
        self.r = r
        self.sigmoid = lambda x: torch.sigmoid(self.r * x)
        self.segmentation = segmentation

    def forward(self, input):
        mask = self.unet_model(input)
        if self.segmentation:
            mask = self.sigmoid(mask)
        return mask


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, netD="n_layers", n_layers_D=3, norm='instance'):
        super(Discriminator, self).__init__()
        self.model = define_D(input_nc=input_nc, ndf=ndf, netD=netD, n_layers_D=n_layers_D, norm=norm,
                              init_type='normal', init_gain=0.02)

    def forward(self, input):
        return self.model(input)


class CustomUnet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=64, netG="unet_256", use_dropout=True, dropout_value=0.5):
        super(CustomUnet, self).__init__()
        self.unet_model = define_G(input_nc=input_nc, output_nc=output_nc, ngf=ngf, netG=netG, use_dropout=use_dropout,
                                   dropout_value=dropout_value, norm="batch", init_type="normal", init_gain=0.02)

    def forward(self, input):
        mask = self.unet_model(input)
        mask = torch.sigmoid(mask)
        return mask

