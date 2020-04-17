import torch
import networks
from torch import nn
from util import im2tensor, tensor2im
from PIL import Image
import numpy as np
import os
from scipy.io import savemat


class DBPISR:
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = networks.Generator(conf).cuda()
        self.U = networks.Upsampler().cuda()

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()

        # Define loss function
        self.L1 = nn.L1Loss()

        # Initialize networks weightsZ
        self.G.apply(networks.weights_init_G)
        self.U.apply(networks.weights_init_U)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_U = torch.optim.Adam(self.U.parameters(), lr=conf.u_lr, betas=(conf.beta1, 0.999))

        print('*' * 60 + '\nSTARTED DBPI-BlindSR on: \"%s\"...' % conf.input_image_path)
        self.ds = nn.AvgPool2d(2, ceil_mode=True)

    def train(self, g_input):
        self.set_input(g_input)
        self.train_g()

    def set_input(self, g_input):
        self.g_input = g_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        self.optimizer_U.zero_grad()

        # Generator forward pass
        g_pred = self.G(self.g_input)
        g_downup = self.U(torch.clamp(g_pred.detach(), -1, 1))
        g_up = self.U(self.g_input)
        g_updown = self.G(torch.clamp(g_up.detach() + 1e-1 * torch.randn_like(g_up), -1, 1))

        rand_input = torch.rand(3).cuda().view(1, 3, 1, 1,) * 2 - 1
        rand_l = rand_input * torch.ones_like(g_pred)
        rand_h = rand_input * torch.ones_like(self.g_input)
        g_rand = self.G(rand_h)
        u_rand = self.U(rand_l)

        loss_ud = self.L1(g_updown, self.g_input)
        loss_du = self.L1(g_downup, self.g_input)

        total_loss = loss_du + loss_ud + self.L1(g_rand, rand_l) + .1 * self.L1(u_rand, rand_h)

        # Calculate gradients
        total_loss.backward()

        # Update weights
        self.optimizer_G.step()
        self.optimizer_U.step()

    def finish(self, image):

        with torch.no_grad():
            image = im2tensor(image)

            sr = self.U(image)
            if self.conf.X4:
                sr = im2tensor(tensor2im(sr))
                sr = self.U(sr)
            sr = tensor2im(sr)

            def save_np_as_img(arr, path):
                Image.fromarray(np.uint8(arr)).save(path)
            save_np_as_img(sr, os.path.join(self.conf.output_dir_path, 'image sr.png'))
            print('FINISHED RUN (see --%s-- folder)\n' % self.conf.output_dir_path + '*' * 60 + '\n\n')
