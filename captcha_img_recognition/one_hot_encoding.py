"""one hot encoding for labels and decode
"""
import numpy as np

import setting


def encode(label):
    """Encode label to one-hot encoding
    Arguments:
        label -- captcha string, eg, "huY6"
    Return:
        vector -- one-hot encoding vector ,size:(setting.char_num * setting.pool_length)
    """
    vector = np.zeros(
        (setting.char_num * setting.pool_length), dtype=np.float32)
    for i, char in enumerate(label):
        num = char2num[char]
        vector[i*setting.pool_length+num] = 1.0
    return vector


def decode(vector):
    """Decode one-hot encoding vector to label
    Arguments:
        vector -- one-hot encoding vector
    Return:
        label -- captcha string, eg "huY6"
    """
    nums = vector.nonzero()[0]
    label = ''
    for i in range(setting.char_num):
        char = num2char[nums[i]-i*setting.pool_length]
        label += char
    return label


# use dict to map character and it's position
charset = setting.char_pool
num_list = list(range(setting.pool_length))
char2num = dict(zip(charset, num_list))
num2char = {item[1]: item[0] for item in char2num.items()}


if __name__ == "__main__":
    vector = encode('nnnn')
    print('After encoding and decoding, the final value of nnnn is ----', decode(vector))
