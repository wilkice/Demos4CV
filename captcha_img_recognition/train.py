from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import data_preprocess
from cnn_model import Net
import setting
import one_hot_encoding as ohe
import numpy as np

# cpu() convert data to cpu

num_epochs = 30
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
        end = time()
        print('epoch: {}, time: {:.2f}s     loss: {:.04}'.format(
            epoch, end-start, loss.item()))

    torch.save(net.state_dict(), './model.pkl')

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
                        single_labels_ohe_predict[slice*setting.length_charset:(slice+1)*setting.length_charset].cpu().data.numpy())]
                    predict_label += char
                # get true label
                true_label = ohe.decode(labels[single, :].cpu().numpy())
            # print(true_label,predict_label)
                if predict_label == true_label:
                    correct += 1
            print('accuracy: {:.4f}'.format(correct/total))
            print('total accurate:{}'.format(correct))


def didi():
    single_labels_ohe_predict = np.zeros(248,dtype=float)

    single_labels_ohe_predict[2]=0.3
    single_labels_ohe_predict[3]=0.5
    single_labels_ohe_predict[5]=0.2
    single_labels_ohe_predict[64]=0.7
    single_labels_ohe_predict[65]=0.3
    single_labels_ohe_predict[126]=0.7
    single_labels_ohe_predict[127]=0.3
    single_labels_ohe_predict[188]=0.7
    single_labels_ohe_predict[189]=0.3
    predict_label = ''
    # get predict_label
    for slice in range(setting.char_num):
        char = setting.charset[np.argmax(
            # single_labels_ohe_predict[slice*setting.length_charset:(slice+1)*setting.length_charset].data.numpy())]
            single_labels_ohe_predict[slice*setting.length_charset:(slice+1)*setting.length_charset])]
        predict_label += char
    print(predict_label)


if __name__ == "__main__":
    main()
    valid()
    # didi()
