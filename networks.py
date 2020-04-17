import torch
import torch.nn as nn
from util import swap_axis


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        # First layer - down sampling
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], padding=struct[0]//2, stride=int(1 / conf.scale_factor), bias=False)
        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], padding=struct[layer]//2, bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], bias=False)
        self.final_act = nn.Tanh()
        self.downsample = nn.AvgPool2d(2, ceil_mode=True)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size


    def forward(self, input_tensor):
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features) + self.downsample(input_tensor)
        return swap_axis(output)


class Upsampler(nn.Module):
    def __init__(self):
        super(Upsampler, self).__init__()
        self.first_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, 7):
            feature_block += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_act = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = input_tensor
        input_tensor = self.upsample(input_tensor)
        x = self.first_layer(input_tensor.detach())
        x = self.feature_block(x)
        x = self.final_layer(x)
        return x + input_tensor

def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def weights_init_U(m):
    """ initialize weights of the upsampler  """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)

