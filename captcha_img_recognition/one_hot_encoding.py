
import numpy as np
import setting


def encode(label):
    vector = [0.0 for _ in range(setting.char_num*setting.length_charset)]
    for i, char in enumerate(label):
        num = char2num[char]
        vector[i*setting.length_charset+num] = 1.0
    return np.array(vector, dtype=np.float32)


def decode(vector):
    nums = vector.nonzero()[0]
    label =''
    for i in range(setting.char_num):
        char = num2char[nums[i]-i*setting.length_charset]
        label += char
    return label


charset = setting.charset
num_list = range(setting.length_charset)
char2num = dict(zip(charset, num_list))
num2char = {item[1]: item[0]for item in char2num.items()}


if __name__ == "__main__":
    vector = encode('JoK1')
    print(decode(vector))
