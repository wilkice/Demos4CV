"""
TODO: add accuracy to this program
"""
from time import time

import torch
import torch.nn as nn
from tqdm import tqdm

import data_preprocess
from cnn_model import Net
import setting
# cpu() convert data to cpu

num_epochs = 30
learning_rate = 1e-3


def main():
    net = Net().to(setting.device)
    net.train()
    print('The modle has been initialized.')
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    train_dataloader = data_preprocess.get_train_dataloader()
    loss = 0
    for epoch in tqdm(range(num_epochs)):
        start = time()
        # for train accuracy
        # total = setting.train_img_nums
        # correct = 0
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            # imgs = Variable(imgs)
            # label = Variable(labels)
            imgs, labels = imgs.to(setting.device), labels.to(setting.device)
            labels = labels.long()
            labels_ohe_predict = net(imgs)

            
            for i in range(setting.char_num):
                one_label = labels[:, i * setting.pool_length:(i+1) * setting.pool_length]
                one_class = one_label.argmax(dim=1)
                one_predict_label = labels_ohe_predict[:, i * setting.pool_length:(i+1) * setting.pool_length]
                one_loss = criterion(one_predict_label, one_class)
                loss += one_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for single in range(labels_ohe_predict.shape[0]):
  
            #     single_labels_ohe_predict = labels_ohe_predict[single, :]
                
            #     predict_label = ''
            #     # get predict_label
            #     for slice in range(setting.char_num):
            #         char = ohe.num2char[np.argmax(
            #             single_labels_ohe_predict[slice*setting.pool_length:(slice+1)*setting.pool_length].cpu().data.numpy())]
            #         predict_label += char
            #     # get true label
            #     true_label = ohe.decode(labels[single, :].cpu().numpy())           
            #     if predict_label == true_label:
            #         correct += 1
        end = time()
        print('epoch: {}, time: {:.2f}s   loss: {:.04}'.format(
            epoch, end-start, loss.item()))
        # print('epoch: {}, time: {:.2f}s   loss: {:.04}  accuracy: {}/{} -- {:.4f}'.format(
        #     epoch, end-start, loss.item(), correct, total, correct/total))

    torch.save(net.state_dict(), './model.pt')


if __name__ == "__main__":
    main()
