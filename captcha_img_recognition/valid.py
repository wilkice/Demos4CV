import os
import collections

import numpy as np
import torch
from tqdm import tqdm

import setting
import data_preprocess
from cnn_model import Net
import one_hot_encoding as ohe


def main():
    net = Net().to(setting.device)
    net.eval()
    net.load_state_dict(torch.load('model.pt'))
    print('model has been loaded.')
    correct = 0
    valid_dataloader = data_preprocess.get_valid_dataloader()
    total = len(os.listdir(setting.valid_folder_path))
    with torch.no_grad():
        
        miss_character = {}
        for (imgs, labels) in tqdm((valid_dataloader)):
            imgs, labels = imgs.to(setting.device), labels.to(setting.device)
            labels_ohe_predict = net(imgs)
            # for each img in one batch
            for single in range(labels_ohe_predict.shape[0]):              
                single_labels_ohe_predict = labels_ohe_predict[single, :]
                predict_label = ''
                # get predict_label
                for slice in range(setting.char_num):
                    char = ohe.num2char[np.argmax(
                        single_labels_ohe_predict[slice*setting.pool_length:(slice+1)*setting.pool_length].cpu().data.numpy())]
                    predict_label += char
                # get true label
                true_label = ohe.decode(labels[single, :].cpu().numpy())
                # print('true label:', true_label, '   predict label:', predict_label)
                if predict_label == true_label:
                    correct += 1
                else:
                    
                    for i in range(setting.char_num):
                        if predict_label[i] != true_label[i]:
                            error_info = '{} -> {}'.format(true_label[i], predict_label[i])
                            if error_info in miss_character:
                                miss_character[error_info] +=1
                            else:
                                miss_character[error_info] =1
    sorted_miss = sorted(miss_character.items(), key=lambda kv:kv[1], reverse=True)
    sorted_miss=collections.OrderedDict(sorted_miss)            
    with open('miss_character.txt','w') as f:
        for i in sorted_miss:
            f.write('{} : {}\n'.format(i, sorted_miss[i]))
    print('accuracy: {}/{} -- {:.4f}'.format(correct, total, correct/total))


if __name__ == "__main__":
    main()
