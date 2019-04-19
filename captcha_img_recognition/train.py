import torch
import torch.nn as nn
from torch.autograd import Variable
import data_preprocess
from cnn_model import Net


num_epochs = 30
batch_size = 5
learning_rate = 1e-3

def main():
    net = Net()
    net.train()
    print('The modle has been initialized.')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


    train_dataloader = data_preprocess.get_train_dataloader()
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs = Variable(imgs)
            label = Variable(labels)
            label_predict = net(imgs)

            loss = criterion(label_predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}      loss: {:.04}'.format(epoch, loss.item()))
        
    torch.save(net.state_dict(), './model.pkl')
    print('*'*30)
    print('model has been saved to ./model.pkl')
    print('*'*30)

if __name__ == "__main__":
    main()