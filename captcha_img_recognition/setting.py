"""
Settings for captcha image generator
"""

import torch

number = '0123456789'
upper_case = 'QWERTYUIOPASDFGHJKLZXCVBNM'
lower_case = 'qwertyuiopasdfghjklzxcvbnm'

# delete what you don't need
char_pool = number+upper_case+lower_case
pool_length = len(char_pool)

# change this num if u need more or less character
char_num = 4

# height and width for generated imgs
image_height = 60
image_width = 160

# specify img path and img nums
train_folder_path = 'dataset/train'
train_img_nums = 10000

valid_folder_path = 'dataset/valid'
valid_img_nums = 1000

test_folder_path = 'dataset/test'
test_img_nums = 0

# specify GPU or CPU, we only talk about at most 1 GPU card here
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('You are using', device)