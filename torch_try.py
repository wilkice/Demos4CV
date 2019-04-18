import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import one_hot_encoding
import torchvision.transforms as transforms
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


class Data(Dataset):
    def __init__(self, folder):
        super(Data, self).__init__()
        self.train_img_paths = [os.path.join(folder, img_name) for img_name in os.listdir(folder)]

    def __len__(self):
        return len(self.train_img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.train_img_paths[idx])
        image = np.array(np.transpose(image, (2,0,1)), dtype='f')

        # Attention: Under windows, the path will be 'cap_img\\xxxx.jpg',
        # so if u use split('/), u will get nothing!!!

        label = self.train_img_paths[idx][-8:-4]
        return image, one_hot_encoding.encoding(label)

if __name__ == "__main__":
    epochs = 10
    learning_rate = 1e-3
    net = Net()
    trainset = Data('cap_img')
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
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








