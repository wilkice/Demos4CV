import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import one_hot_encoding
import torchvision.transforms as transforms
import numpy as np
from captcha_cnn_model import Net


class Data(Dataset):
    def __init__(self, folder):
        super(Data, self).__init__()
        self.train_img_paths = [os.path.join(folder, img_name) for img_name in os.listdir(folder)]

    def __len__(self):
        return len(self.train_img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.train_img_paths[idx])
       
        # image = np.array(np.transpose(image, (2,0,1)), dtype='f')
        label = self.train_img_paths[idx][-8:-4]
        return image, one_hot_encoding.encoding(label)

if __name__ == "__main__":
    epochs = 10
    batch_size = 4
    learning_rate = 1e-3

    net = Net()
    net.train()
    
    trainset = Data('cap_img')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print('Start training!')
    for epoch in range(epochs):
        runing_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs=net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runing_loss +=loss.item()
            if i % 100 ==99:
                print('[{}, {}] loss:{}'.format(epoch+1, i+1, runing_loss/100))
                runing_loss=0.0
    print('finished')








