import torch
import torch.nn as nn
import torch.optim as optim
from PIL import  Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# current_device = torch.cuda.get_device_name(torch.cuda.current_device())
print('You are using {}'.format(device))

root_folder = 'data/face/lfw-deepfunneled'
workers = 2 
batch_size = 128
nc = 3
nz = 100
ngf = 64
ndf = 64
image_size = 64
num_epochs = 10
lr = 2e-4
beta1 = 0.5
ngpu = 1


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8 ),
            nn.ReLU(True),
           

            nn.ConvTranspose2d(ngf *8, ngf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # size(ngf*4, 14, 14 )

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # size (ngf * 2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2,1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # size (ngf, 28, 28

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # size(nc, 28, 28)
        )

    def forward(self, input):
        return self.main(input)


# instantiate the generator
netG = Generator(ngpu).to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# load parameters
checkpoint = torch.load('face_train.tar')
netG.load_state_dict(checkpoint['generator'])
optimizerG.load_state_dict(checkpoint['optimizer_generator'])
netG.eval()

noise = torch.randn(1,100,1,1,device=device)
output = netG(noise)

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imsave(tensor, name):
    tensor = (tensor * 127.5 + 127.5)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    print(image) # remove the fake batch dimension
    image = unloader(image)
    image.save(name)
imsave(output, 'face.jpg')