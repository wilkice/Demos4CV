"""
settings for captcha image
"""
import os

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
train_img_nums = 5000

valid_img_path = 'dataset/valid'
valid_img_nums = 200

test_img_path = 'dataset/test'
test_img_nums = 10
