import os
import re
import copy
import warnings
import numpy as np
import tensorflow as tf

from random import shuffle
from skimage import exposure
from skimage.io import imread
from skimage.transform import resize, rescale, rotate
from matplotlib import pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Convolution2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from functions.dataloader4 import shuffle_n_split_data, get_file_list, train_gen_weight
from functions.networks4 import get_unet_weight
from functions.training_tools4 import recall, precision, dice_coef, plotting

# --------------- Task 2 ------------ #

if __name__ == "__main__":
    img_dir = '/DL_course_data/Lab3/MRI/Image/'
    msk_dir = '/DL_course_data/Lab3/MRI/Mask/'

    img_pathlist = get_file_list(img_dir)
    msk_pathlist = get_file_list(msk_dir)

    base = 8
    img_w = 240
    img_h = 240
    img_ch = 1
    bs = 8
    lr = 0.0001
    batch_norm = 1
    dropout = 1
    n_ep = 150
    Metric = [dice_coef, precision, recall]
    weight_strength = 1

    x_train_list, y_train_list, x_val_list, y_val_list = shuffle_n_split_data(img_pathlist, msk_pathlist, 0.8)

    n_train_sample = len(x_train_list)
    n_val_sample = len(x_val_list)

    train_generator = train_gen_weight(x_train_list,
                                       y_train_list,
                                       n_train_sample, bs, img_w)

    val_generator = train_gen_weight(x_val_list,
                                     y_val_list,
                                     n_val_sample, bs, img_w)

    network_task2 = get_unet_weight(base, img_w, img_h, img_ch, batch_norm, dropout, weight_strength, lr, Metric)

    network_task2_hist = network_task2.fit_generator(train_generator,
                                                     steps_per_epoch=int(n_train_sample) / bs,
                                                     validation_data=val_generator,
                                                     validation_steps=int(n_val_sample) / bs,
                                                     epochs=n_ep)
    plotting(network_task2_hist)
