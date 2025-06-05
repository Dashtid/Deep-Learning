import os
import re
import cv2
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, ConvLSTM2D, Dense, Flatten, MaxPooling2D, Conv2D, Activation, \
    Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM, Bidirectional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from functions.dataloader5 import get_file_list, shuffle_n_split_data
from functions.training_tools5 import train_gen, dice_loss, dice_coef, precision, recall
from functions.networks5 import get_unet, plot_history

if __name__ == "__main__":

    # --------------- Task 3 ------------ #

    img_dir = '/DL_course_data/Lab3/MRI/Image/'
    msk_dir = '/DL_course_data/Lab3/MRI/Mask/'

    img_pathlist = get_file_list(img_dir)
    msk_pathlist = get_file_list(msk_dir)

    x_train_list, y_train_list, x_val_list, y_val_list = shuffle_n_split_data(img_pathlist, msk_pathlist, 0.8)

    base = 8           # Number of feature maps
    img_w = 240        # Image width
    img_h = 240        # Image height
    img_ch = 1         # Number of image channels
    bs = 8             # Batch size
    lr = 0.0001        # Learning rate
    batch_norm = 1     # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 1        # On/Off switch for dropout layer, 0 = False, 1 = True
    n_ep = 50          # Number of epochs

    Metric = [dice_coef, precision, recall]  # Evaluation metrics

    n_train_sample = len(x_train_list)
    n_val_sample = len(x_val_list)

    train_generator = train_gen(x_train_list,
                                y_train_list,
                                n_train_sample, bs, img_w)

    val_generator = train_gen(x_val_list,
                              y_val_list,
                              n_val_sample, bs, img_w)

    model = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)

    model.compile(loss=dice_loss,
                  optimizer=Adam(lr=lr),
                  metrics=Metric)

    model_hist = model.fit_generator(train_generator,
                                     steps_per_epoch=int(n_train_sample) / bs,
                                     validation_data=val_generator,
                                     validation_steps=int(n_val_sample) / bs,
                                     epochs=n_ep)

    plot_history(model_hist, "figure_t3_l6")


