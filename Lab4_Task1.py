import os
import re
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

from functions.networks3 import get_unet
from functions.dataloader3 import get_file_list, shuffle_n_split_data, load_img
from functions.training_tools3 import train_with_adam, train_with_generator

# --- Task 1 --- #

if __name__ == "__main__":
    # ----------------------------------- PREPARATION ----------------------------------- #

    train_dir = '/DL_course_data/Lab3/X_ray/Image/'
    val_dir = '/DL_course_data/Lab3/X_ray/Mask/'

    # Loading in paths and splitting them according to the task, 80/20 split
    img_pathlist = get_file_list(train_dir)
    msk_pathlist = get_file_list(val_dir)
    train_img, train_msk, val_img, val_msk = shuffle_n_split_data(img_pathlist, msk_pathlist, 0.8)

    # Loading in the actual images
    x_train = load_img(train_img, 256, 0)
    y_train = load_img(train_msk, 256, 1)
    x_val = load_img(val_img, 256, 0)
    y_val = load_img(val_msk, 256, 1)
    val_data = (x_val, y_val)

    # -------------------------------------- Task 1A -------------------------------------- #

    # Setting parameters
    base = 8  # Number of feature maps in convolutional layer
    img_w = 256  # Image width
    img_h = 256  # Image height
    img_ch = 1  # Number of image channels
    bs = 8  # Batch size
    lr = 0.0001  # Learning rate
    batch_norm = 0  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 0  # On/Off switch for dropout layer, 0 = False, 1 = True
    dice = 0  # On/Off switch for DICE-loss function, 0 = False, 1 = True
    n_ep = 150  # Number of epochs to be run

    # Training the network and plotting results
    networktask1a = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1a, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1B -------------------------------------- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    networktask1b = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1b, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1C -------------------------------------- #

    dropout = 1  # Turning on dropout layers

    # --- Without DICE --- #

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    networktask1c1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1c1, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # --- With DICE --- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    networktask1c2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1c2, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1D -------------------------------------- #

    base = 32  # Increasing number of feature maps in convolutional layer

    # --- Without DICE --- #

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    networktask1d1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1d1, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # --- With DICE --- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    networktask1d2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(networktask1d2, lr, bs, n_ep, dice, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1E -------------------------------------- #

    base = 16  # Increasing number of feature maps in convolutional layer
    batch_norm = 1  # Turning on batch-normalization layer
    dropout = 1  # Turning on dropout layer
    categorical = 0  # On/Off switch for categorical cross-entropy function, 0 = False, 1 = True
    metrics = 0  # On/Off switch for precision and recall, 0 = False, 1 = True

    train_datagenerator = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
    train_generator = train_datagenerator.flow(x_train, y_train, batch_size=8)

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    networktask1e1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_generator(networktask1e1, lr, n_ep, metrics, categorical, train_generator, val_data)

    # --- With DICE --- #

    metrics = 1  # Using the dice loss function

    # Training the network and plotting results
    networktask1e2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_generator(networktask1e2, lr, n_ep, metrics, categorical, train_generator, val_data)
