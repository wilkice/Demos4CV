"""
Neural transfer using PyTorch
"""

import copy
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models

# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# TODO: tensorboard

# use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    imsize = 512
else:
    device = torch.device('cpu')
    imsize = 128

# pic preprocessing
# writer = SummaryWriter()


def image_loader(img_name):
    image = Image.open(img_name)
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    """
    torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.

    For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.

    If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
    """
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader('style.jpg')
content_img = image_loader('content.jpg')
assert style_img.size() == content_img.size()

"""Loss function

"""

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


"""use pretrained model
"""

cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalize our img for vgg
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img-self.mean) / self.std


# create our model
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_model(cnn, mean, std, style_img, content_img):

    content_losses = []
    style_losses = []

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(mean, std).to(device)
    model = nn.Sequential(normalization)

    i=0  # layer num
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]
    return model, style_losses, content_losses


"""Gradient Descent"""

input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])

def run_style_transfer(cnn=cnn, mean=mean, std=std, content_img=content_img, style_img=style_img, epochs=500, style_weight=1000000, content_weight=1):
    print('Building the style transfer model...')
    
    model, style_losses, content_losses = get_model(
        cnn, mean, std, style_img, content_img)
    print('Optimizing...')
    run = [0]
    while run[0] <= epochs:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            run[0] +=1
            if run[0] % 50 == 0:
                print('run {}'.format(run))
                print('Style loss: {:.4f}  Content loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img


output = run_style_transfer()
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save('transfer.jpg')
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
