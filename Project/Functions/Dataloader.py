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


def load_img_array(path_list, size, img_ch, window_level):
    img_data = np.zeros((len(path_list), size, size, img_ch), dtype='float')
    zero_pad_img = np.zeros((size, size, img_ch))

    for i in range(len(path_list)):
        img_arr = sitk.ReadImage(path_list[i])

        if window_level:
            img_arr = sitk.IntensityWindowing(img_arr)
        img_arr = sitk.RescaleIntensity(img_arr, 0, 1)
        img_arr = sitk.GetArrayFromImage(img_arr)

        if np.ndim(img_arr) > 2:
            img_arr = np.rollaxis(img_arr, 0, 3)
            img_size = len(img_arr)
            margin = size - img_size

            if img_size > size:

                if img_size < len(img_arr[0]):
                    y_len = len(img_arr[0])
                    y_shrink = y_len - size

                else:
                    y_shrink = img_size - size
                    y_len = img_size

                x_shrink = img_size - size
                img_arr = img_arr[x_shrink // 2:img_size - x_shrink // 2, y_shrink // 2:y_len - y_shrink // 2, :]
                img_size = len(img_arr)
                margin = size - img_size

            # Zero-padding the images
            for j in range(img_ch):
                zero_pad_img[margin // 2:img_size + margin // 2, margin // 2:img_size + margin // 2, j] = img_arr[:, :, j]
            img_data[i, :, :, :] = zero_pad_img

        else:
            img_size = len(img_arr)
            margin = size - img_size
            img_arr = np.expand_dims(img_arr, 2)
            zero_pad_img[margin // 2:img_size + margin // 2, margin // 2:img_size + margin // 2] = img_arr
            img_data[i, :, :, :] = zero_pad_img

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
