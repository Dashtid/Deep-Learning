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

from functions.Lab5.dataloader import k_split, get_file_list, train_gen
from functions.Lab5.networks import get_unet
from functions.Lab5.training_tools import recall, precision, dice_coef, dice_loss, plotting

# --------------- Task 1 ------------ #

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
    n_ep = 50
    folds = 3
    Metric = [dice_coef, precision, recall]

    split_list = k_split(img_pathlist, msk_pathlist, folds)

    for i in range(folds):
        x_train_list = split_list[i][0]
        y_train_list = split_list[i][1]
        x_val_list = split_list[i][2]
        y_val_list = split_list[i][3]

        n_train_sample = len(x_train_list)
        n_val_sample = len(x_val_list)

        train_generator = train_gen(x_train_list,
                                    y_train_list,
                                    n_train_sample, bs, img_w)

        val_generator = train_gen(x_val_list,
                                  y_val_list,
                                  n_val_sample, bs, img_w)

        network_task1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)

        network_task1.compile(loss=dice_loss,
                              optimizer=Adam(lr=lr),
                              metrics=Metric)

        network_task1_hist = network_task1.fit_generator(train_generator,
                                                         steps_per_epoch=n_train_sample/bs,
                                                         validation_data=val_generator,
                                                         validation_steps=n_val_sample/bs,
                                                         epochs=n_ep)
        plotting(network_task1_hist)
