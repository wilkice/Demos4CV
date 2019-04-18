"""Get captcha img by using captcha.img
github: https://github.com/lepture/captcha
img contains 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789'
if you want more, change this string,
default is 4 characters
"""

import os
import random

from captcha.image import ImageCaptcha
import pandas as pd

def img_generate():
    cap_list = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    cap_length = 4
    num_img = int(input('Pls tell me how many imgs u want to get: '))
    if  isinstance(num_img, int):
        try:
            os.mkdir('cap_img')
        except:
            pass
        finally:
            image = ImageCaptcha(80, 30)
            for _ in range(num_img):
                label = ''.join(random.choices(cap_list, k=cap_length))
                image.write(label, 'cap_img/'+label+'.jpg')

def csv_generate():
    """generate a csv file which contains filename, labels and convert labels \
        to nums
    """
    # dict{key: character, value: num}
    char_list = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    num_list = list(range(62))
    char2num = dict(zip(char_list, num_list))
    file_list = os.listdir('cap_img')
    label_list = [filename.split('.')[0] for filename in file_list]
    num_list = [[char2num[char] for char in label] for label in label_list]
    data = {'img_name': file_list, 'label': label_list, 'char2num': num_list}
    df=pd.DataFrame(data)
    df.to_csv('char2num.csv')

if __name__ == "__main__":
    img_generate()
    csv_generate()
