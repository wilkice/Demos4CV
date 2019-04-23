from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import data_preprocess
from cnn_model import Net
import setting
import one_hot_encoding as ohe
import numpy as np


num_epochs = 10
batch_size = 16
learning_rate = 1e-3


def main():
    net = Net().to(setting.device)
    net.train()
    print('The modle has been initialized.')
    # define loss and optimizer
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_dataloader = data_preprocess.get_train_dataloader()
    for epoch in range(num_epochs):
        start = time()
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            # imgs = Variable(imgs)
            # label = Variable(labels)
            imgs, labels = imgs.to(setting.device), labels.to(setting.device)
            labels_ohe_predict = net(imgs)
            loss = criterion(labels_ohe_predict, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}      loss: {:.04}'.format(epoch, loss.item()))

    torch.save(net.state_dict(), './model.pkl')
    end = time()
    print('{:.2f}s'.format(end-start))
    print('*'*30)
    print('model has been saved to ./model.pkl')
    print('*'*30)


def valid():
    net = Net().to(setting.device)
    correct = 0
    total = 0
    valid_dataloader = data_preprocess.get_valid_dataloader()
    with torch.no_grad():
        for batch_size, (imgs, labels) in enumerate(valid_dataloader):
            imgs, labels = imgs.to(setting.device), labels.to(setting.device)
            labels_ohe_predict = net(imgs)
            total += len(imgs)
            # for each img in one batch
            for single in range(labels_ohe_predict.shape[0]):
                single_labels_ohe_predict = labels_ohe_predict[single, :]
                predict_label = ''
                # get predict_label
                for slice in range(setting.char_num):
                    char = setting.charset[np.argmax(
                        single_labels_ohe_predict[slice*setting.length_charset:(slice+1)*setting.length_charset].data.numpy())]
                    predict_label += char
                # get true label
                true_label = ohe.decode(labels[single, :].numpy())
            # print(true_label,predict_label)
                if predict_label == true_label:
                    correct += 1
                print('True label:', true_label,
                      '    Predict label:', predict_label)
            print('accuracy: {:.4f}'.format(correct/total))


if __name__ == "__main__":
    main()
    valid()
