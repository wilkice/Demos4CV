"""
Generate captcha images using python module-captcha
Github: https://github.com/lepture/captcha
Install by `pip install captcha`
"""

import os
import random

from captcha.image import ImageCaptcha
import setting


def generate_img(imgfolder, img_num):
    """Save img to folder

    Arguments:
        imgfolder -- directory to save imgs
        img_num -- num of imgs to save
    """
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)
    image = ImageCaptcha(setting.image_width, setting.image_height)
    for i in range(img_num):
        label = ''.join(random.choices(setting.char_pool, k=setting.char_num))
        img_name = os.path.join(imgfolder, label + '.png')
        image.write(label, img_name)


if __name__ == "__main__":
    generate_img(setting.train_folder_path, setting.train_img_nums)
    generate_img(setting.valid_folder_path, setting.valid_img_nums)
    generate_img(setting.test_folder_path, setting.test_img_nums)
