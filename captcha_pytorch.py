
import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
import cv2

def init():
    file_path = './cap_img'
    batch_size = 16
    epoch = 10

def data_get(csv_path,img_folder):
    df = pd.read_csv(csv_path)

    x=[]
    y=[]
    for i in range(10):
        img_path = os.path.join(img_folder, df.iloc[i]['img_name'])
        img = cv2.imread(img_path, )
if __name__ == "__main__":
    init()
    data_get('char2num.csv', 'cap_img')
