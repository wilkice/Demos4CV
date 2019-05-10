"""DCGAN implement by PyTorch
time: 2019/5/8
"""

import os
import random
from IPython.display import HTML

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO: beta 1

# set random seed for reproducibility
manualseed = 999
print('Random seed: ', manualseed)
random.seed(manualseed)
torch.manual_seed(manualseed)

"""Hyperparamater setting
"""
dataroot = 'gdrive/My\ Drive/data/'
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 2e-4
beta1 = 0.5
ngpu = 1


"""Prepare data
"""
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = dset.ImageFolder(dataroot, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
device = torch.device('cuda: 0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')


"""weight initialization
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # state size (ngf*8,4,4)

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size (ngf*4,8,8)

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size(ngf*2, 16,16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size(ngf, 32,32)

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size(nc, 64,64)
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input (nc,64,64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf, 32,32)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf * 2, 16,16)

            nn.Conv2d(ndf * 2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf *4 , 8,8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf * 8, 4,4)

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size(1)
        )

    def forward(self, input):
        return self.main(input)


netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)


"""Loss Functions and optimizers
"""

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


"""Training"""
img_list = []
G_losses = []
D_losses = []
iters = 0

print('Start training')

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # (1) update D network

        # (1.1) train with all_real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ), real_label, device=device)

        # forward pass real batch D
        output = netD(real_cpu).view(-1)
        # calculate loss on real batch
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # (1.2) train with all_fake batch
        noise = torch.randn(b_size, nz, 1,1,device=device)
        fake = netG(noise)
        label.fill_(fake_label)

        # forward pass fake batch D
        output = netG(fake.detach()).view(-1)
        # calculate loss on fake batch
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_fake + errD_real
        optimizerD.step()

        # (2) update G network
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[{}/{}] [{}/{}] Loss_D:{:.4f} Loss_G:{:.4f} D(x):{:.4f} D(G(z)):{:.4f}/{:.4f}'.
            format(epoch, num_epochs, i, len(dataloader),errD.item(),errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs -1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters +=1





