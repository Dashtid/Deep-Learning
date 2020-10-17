import os
import re
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add, AveragePooling2D, Reshape, ConvLSTM2D, Dense, Flatten, ZeroPadding2D, \
    MaxPooling2D, Conv2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM


def load_img_array(path_list, size):
    img_data = np.zeros((len(path_list), size, size, 1), dtype='float')
    for i in range(len(path_list)):
        img_arr = sitk.ReadImage(path_list[i])
        img_arr = sitk.GetArrayFromImage(img_arr)
        img_arr = img_arr / np.max(img_arr)
        img_data[i, :, :, 0] = img_arr
    return img_data


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def get_file_list(data_path, msk_name):
    img_list = []
    msk_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file == 'image.nii.gz':
                mypath = os.path.join(root, file)
                img_list.append(mypath)
            if msk_name in file:
                mypath = os.path.join(root, file)
                msk_list.append(mypath)

    img_list.sort(key=natural_sort_key)
    msk_list.sort(key=natural_sort_key)

    return img_list, msk_list
