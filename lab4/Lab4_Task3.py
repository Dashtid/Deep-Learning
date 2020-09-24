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

from functions.Lab4.networks import get_unet
from functions.Lab4.dataloader import shuffle_n_split_data, load_img, get_file_list, load_msk
from functions.Lab4.training_tools import train_with_generator

# --- Task 3 --- #

if __name__ == "__main__":

    # --------- PREPARATION ----- #

    img_dir = '/DL_course_data/Lab3/CT/Image/'
    msk_dir = '/DL_course_data/Lab3/CT/Mask/'

    img_pathlist = get_file_list(img_dir)
    msk_pathlist = get_file_list(msk_dir)

    train_img, train_msk, val_img, val_msk = shuffle_n_split_data(img_pathlist[0:4000], msk_pathlist[0:4000], 0.8)

    # Loading in the actual images
    x_train = load_img(train_img, 256, 0)
    y_train = load_msk(train_msk, 256)
    x_val = load_img(val_img, 256, 0)
    y_val = load_msk(val_msk, 256)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

    val_data = (x_val, y_val)

    # -------------------------------------- Task 3 -------------------------------------- #

    # Setting parameters
    base = 8  # Number of feature maps in convolutional layer
    img_w = 256  # Image width
    img_h = 256  # Image height
    img_ch = 1  # Number of image channels
    bs = 8  # Batch size
    lr = 0.00001  # Learning rate
    batch_norm = 1  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    dice = 1  # On/Off switch for DICE-loss function, 0 = False, 1 = True
    categorical = 1  # On/Off switch for categorical cross-entropy function, 0 = False, 1 = True
    metrics = 1  # On/Off switch for precision and recall, 0 = False, 1 = True
    n_ep = 80  # Number of epochs to be run

    train_datagenerator = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
    train_generator = train_datagenerator.flow(x_train, y_train, batch_size=8)

    network_task3 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout, task3=3)
    train_with_generator(network_task3, lr, n_ep, metrics, categorical, train_generator, val_data)
