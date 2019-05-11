"""DCGAN demo to generate imgs like mnist
version: 0.1
software: PyTorch
"""

import random
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import argparse
from IPython.display import HTML


# set seed to reproducibility
manualseed = 999
print('Random seed is {}'.format(manualseed))
random.seed(manualseed)
torch.manual_seed(manualseed)

# set hyperparameters
workers = 2 
batch_size = 128
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 2e-4
beta1 = 0.5
ngpu = 1
device = torch.device(
    'cuda:0' if torch.cuda.is_available() and ngpu>0 else 'cpu')

# load data
transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = dset.MNIST(
    'files', train=True, transform=transfrom, download=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# weight initialization
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 7, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8 ),
            nn.ReLU(True),
            # size(ngf*8, 7, 7)

            nn.ConvTranspose2d(ngf *8, ngf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # size(ngf*4, 14, 14 )

            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # size (ngf * 2, 16, 16)

            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2,1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # size (ngf, 28, 28

            nn.ConvTranspose2d(ngf*4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # size(nc, 28, 28)
        )

    def forward(self, input):
        return self.main(input)


# instantiate the generator
netG = Generator(ngpu).to(device)    
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weight_init)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size(nc, 28, 28)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size (ndf, 14, 14)

            nn.Conv2d(ndf, ndf, 4,2,1,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # size(ndf *2, 7, 7)

            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf* 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # size (ndf *4, 8, 8 )

            # nn.Conv2d(ndf *4, ndf *8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf*8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # size(ndf * 8, 4, 4)

            nn.Conv2d(ndf, 1, 7,1, 0,bias=False),
            nn.Sigmoid()
            # size(1,)
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)
if device.type=='cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weight_init)

# loss function
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz,1,1,device=device)

real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# train

img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting Training Loop...')
for epoch in range(num_epochs):
    since = time()
    for i, data in enumerate(dataloader, 0):
        # 1.update discriminator
        netD.zero_grad()
        # train with real data
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake data
        noise = torch.randn(b_size,nz,1,1,device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1) # detach
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_x_z1 = output.mean().item()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 2. update generator 
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1) # dont detach
        errG = criterion(output, label)
        errG.backward()
        D_g = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print(
                '[{}/{}][{}/{}] D(x):{} D(G(z)):{}'.format(epoch, num_epochs, i,len(dataloader), D_x, D_g)
            )
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if iters % 500 == 0 or (epoch == num_epochs -1 and i == len(dataloader)-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1
        interval = time() - since
    print('Time: {:.2f}'.format(interval))

print('end')
torch.save({
    'generator':netG.state_dict(),
    'optimizer_generator':optimizerG.state_dict(),
    'discriminator':netD.state_dict(),
    'optimizer_discriminator':optimizerD.state_dict()
},'./first_train.tar')




