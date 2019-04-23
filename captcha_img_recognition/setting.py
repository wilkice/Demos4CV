"""
settings for captcha image
"""
import os
import torch

number = '1234567890'
upper_case = 'QWERTYUIOPASDFGHJKLZXCVBNM'
lower_case = 'qwertyuiopasdfghjklzxcvbnm'

# if u don't need one of them, just delte
charset = number+upper_case+lower_case
length_charset = len(charset)

# change this num if u need more or less character
char_num = 4

# height and width for generated imgs
image_height = 60
image_width = 160

# specify img path and img nums
train_img_path = 'dataset/train'
train_img_nums = 100000

valid_img_path = 'dataset/valid'
valid_img_nums = 10000

test_img_path = 'dataset/test'
test_img_nums = 10

# specify GPU or CPU, we only talk about at most 1 GPU card here
use_cuda=torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if __name__ == "__main__":
    if use_cuda:
        print('You are using GPU:', device)
    else:
        print('You are using CPU')