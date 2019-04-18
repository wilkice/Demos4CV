"""get one-hot encoding for label
"""
import torch

def label2num():
    """convert label to category nums, eg q:0, w:1, e:2"""
    label_string = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    num_list = list(range(62))
    dict_num = dict(zip(label_string, num_list))
    return dict_num

def encoding(label):
    one_hot_vector = torch.empty(4, 62)
    for i in range(4):
        tmp = torch.zeros(62)
        character = label[i]
        num = dict_num[character]
        tmp[num]=1
        one_hot_vector[i,:] = tmp
    return one_hot_vector

def decoding(vector):
    pass

dict_num = label2num()

    