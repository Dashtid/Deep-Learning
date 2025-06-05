import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure

from functions.networks2 import alexnet
from functions.training_tools2 import train_with_adam
from functions.dataloader import get_train_test_arrays

# --- Task 2 --- #

if __name__ == "__main__":

    # Setting paths and lists to be able to load in data
    skin_labels_string_list = ['Mel', 'Nev']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/Lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w, skin_labels_string_list)

    # -------------------------------------- Task 2A -------------------------------------- #

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 32  # Number of feature maps in first layer
    learning_rate = 0.0001  # Learning rate
    bs = 8  # Batch size
    n_ep = 150  # Number of epochs
    dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    batch_norm = 0  # On/Off switch for batch-normalization dropout layer, 0 = False, 1 = True
    spat_dropout = 1  # On/Off switch for spatial dropout layer, 0 = False, 1 = True

    # --- CNN w/ spatial dropout --- #
    alexnet2a1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet2a1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- CNN wo/ spatial dropout --- #
    spat_dropout = 0  # Turning spatial dropout layer off
    alexnet2a2 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet2a2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # -------------------------------------- Task 2B -------------------------------------- #

    n_ep = 250  # Setting number of epochs

    # --- CNN w/ spatial dropout --- #
    spat_dropout = 1  # Activating spatial dropout layer
    alexnet2b1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet2b1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- CNN wo/ spatial dropout --- #
    spat_dropout = 0  # Turning spatial dropout layer off
    alexnet2b2 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet2b2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)
