import os

import numpy as np
import torch
from torch.autograd import Variable
import setting
import data_preprocess
from cnn_model import Net
import one_hot_encoding as ohe


def main():
    net = Net()
    net.eval()
    net.load_state_dict(torch.load('model.pkl'))
    print('model has been loaded.')

    valid_dataloader = data_preprocess.get_valid_dataloader()

    correct = 0
    total = 0
    print('true_label: predict_label\n')
    for batch_times, (imgs, labels) in enumerate(valid_dataloader):
        img = Variable(imgs)
        valid_output = net(img)
        # for every img in batch
        for single in range(valid_output.shape[0]):
            single_valid_output = valid_output[single, :]
            predict_label = ''
            for slice in range(setting.char_num):
                char = setting.charset[np.argmax(
                    single_valid_output[slice*setting.length_charset:(slice+1)*setting.length_charset].data.numpy())]
                predict_label += char

            true_label = ohe.decode(labels[single, :].numpy())
            # print(true_label,predict_label)
            if predict_label == true_label:
                correct += 1
        total += 2  # 2 is from data_process , valid's batch_size
        print('accuracy: {:.4}'.format(correct/total))


if __name__ == "__main__":
    main()
