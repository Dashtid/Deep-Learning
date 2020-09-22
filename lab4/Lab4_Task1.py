import os
import numpy as np

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Activation, Dropout, \
    BatchNormalization, Conv2D, Conv2DTranspose, concatenate
from tensorflow.python.keras.api._v2.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from functions.Lab4.networks import get_unet
from functions.Lab4.dataloader import get_pathlist_from_file, shuffle_n_split_data, load_img
from functions.Lab4.training_tools import train_with_adam, train_with_generator, plotting

# --- Task 1 --- #

if __name__ == "__main__":

    # ----------------------------------- PREPARATION ----------------------------------- #

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
    n_ep = 150

    train_dir = '/DL_course_data/Lab3/X_ray/Image/'
    val_dir = '/DL_course_data/Lab3/X_ray/Mask/'

    # Loading in paths and splitting them according to the task, 80/20 split
    img_pathlist = get_pathlist_from_file(train_dir)
    msk_pathlist = get_pathlist_from_file(val_dir)
    train_img, train_msk, val_img, val_msk = shuffle_n_split_data(img_pathlist, msk_pathlist, 0.8)

    # Loading in the actual images
    x_train = load_img(train_img, 256, 0)
    y_train = load_img(train_msk, 256, 1)
    x_val = load_img(val_img, 256, 0)
    y_val = load_img(val_msk, 256, 1)

    # -------------------------------------- Task 1A -------------------------------------- #

    # Training the network and plotting results
    network_task1a = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    hist_task1a = train_with_adam(network_task1a, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)
    plotting(hist_task1a)

    # -------------------------------------- Task 1B -------------------------------------- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    network_task1b = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    hist_task1b = train_with_adam(network_task1b, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)
    plotting(hist_task1b)

    # -------------------------------------- Task 1C -------------------------------------- #

    dropout = 1  # Turning on dropout layers

    # --- Without DICE --- #

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    network_task1c1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(network_task1c1, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)

    # --- With DICE --- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    network_task1c2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(network_task1c2, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1D -------------------------------------- #

    base = 32  # Increasing number of feature maps in convolutional layer

    # --- Without DICE --- #

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    network_task1d1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(network_task1d1, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)

    # --- With DICE --- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    network_task1d2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_adam(network_task1d2, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)

    # -------------------------------------- Task 1E -------------------------------------- #

    base = 16  # Increasing number of feature maps in convolutional layer

    train_datagenerator = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=0.2,
                                             horizontal_flip=True)

    train_generator = train_datagenerator.flow(x_train, y_train, batch_size=8)

    dice = 0  # Turning off the dice loss function

    # Training the network and plotting results
    network_task1e1 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_generator(network_task1e1, lr, bs, n_ep, train_generator, x_train, y_train, x_val, y_val)

    # --- With DICE --- #

    dice = 1  # Using the dice loss function

    # Training the network and plotting results
    network_task1e2 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)
    train_with_generator(network_task1e2, lr, bs, dice, n_ep, x_train, y_train, x_val, y_val)

