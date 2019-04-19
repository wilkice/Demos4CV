"""
Generate captcha images using python module-captcha
github: https://github.com/lepture/captcha
install by `pip install capacha`
"""

import os
from PIL import Image
import random

from captcha.image import ImageCaptcha
import setting


def generate_img():
    label = ''.join(random.choices(setting.charset, k=4))
    img = ImageCaptcha()
    image_data = Image.open(img.generate(label))
    return label, image_data

def save_img2folder(folder_path, img_nums):
    for _ in range(img_nums):
        label, image_data = generate_img()
        filename = label + '.png'
        img_path = os.path.join(folder_path, filename)

        image_data.save(img_path)
    print('  {} imgs saved to {}.'.format(img_nums, folder_path) )

if __name__ == "__main__":
    save_img2folder(setting.train_img_path, setting.train_img_nums)
    save_img2folder(setting.valid_img_path, setting.valid_img_nums)
    save_img2folder(setting.test_img_path, setting.test_img_nums)



