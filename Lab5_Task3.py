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

from functions.dataloader4 import k_split, get_file_list, load_img
from functions.networks4 import get_unet
from functions.training_tools4 import recall, precision, dice_coef, dice_loss, plotting, get_autocontext_fold

# ------------ Task 3 ------------ #

if __name__ == "__main__":
    img_dir = '/DL_course_data/Lab3/MRI/Image/'
    msk_dir = '/DL_course_data/Lab3/MRI/Mask/'

    img_pathlist = get_file_list(img_dir)
    msk_pathlist = get_file_list(msk_dir)

    base = 8
    img_w = 240
    img_h = 240
    img_ch = 2
    bs = 8
    lr = 0.0001
    batch_norm = 1
    dropout = 1
    n_ep = 100
    folds = 3
    Metric = [dice_coef, precision, recall]

    split_list = k_split(img_pathlist[:1800], msk_pathlist[:1800], folds)
    model_predictions = np.zeros((len(img_pathlist[:1800]), img_h, img_w, 1))

    for s in range(2):
        for i in range(folds):
            x_train = load_img(split_list[i][0], 240, 0)
            y_train = load_img(split_list[i][1], 240, 1)
            x_val = load_img(split_list[i][2], 240, 0)
            y_val = load_img(split_list[i][3], 240, 1)

            img_per_fold = len(x_val)

            if s == 0:
                autocontext_train = np.zeros_like(x_train) + 0.5
                x_train = np.concatenate((x_train, autocontext_train), axis=-1)
                autocontext_val = np.zeros_like(x_val) + 0.5
                x_val = np.concatenate((x_val, autocontext_val), axis=-1)
            else:
                y_pred = np.load('step' + str(s - 1) + '.npy')
                autocontext_train, autocontext_val = get_autocontext_fold(y_pred, i, folds, img_per_fold, img_h)

                # Concatenate image data and posterior probabilities:
                x_train = np.concatenate((x_train, autocontext_train), axis=-1)
                x_val = np.concatenate((x_val, autocontext_val), axis=-1)

            network_task3 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)

            network_task3.compile(loss=dice_loss,
                                  optimizer=Adam(lr=lr),
                                  metrics=Metric)

            network_task3_hist = network_task3.fit(x_train, y_train,
                                                   batch_size=bs, epochs=n_ep,
                                                   validation_data=(x_val, y_val))

            val_predictions = network_task3.predict(x_val, batch_size=int(bs / 2))
            model_predictions[(i * img_per_fold):((i + 1) * img_per_fold)] = val_predictions
            np.save('step' + str(s) + '.npy', model_predictions)

            del x_train
            del y_train
            del x_val
            del y_val

            plotting(network_task3_hist)

            K.clear_session
