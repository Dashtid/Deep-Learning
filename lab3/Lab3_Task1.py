import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout, BatchNormalization,SpatialDropout2D,ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure

from functions.Lab3.networks import alexnet
from functions.Lab3.training_tools import train_with_adam
from functions.dataloader import get_train_test_arrays

# --- Task 1 --- #

if __name__ == "__main__":
    skin_labels_string_list = ['Mel', 'Nev']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w, skin_labels_string_list)

    # ------------------- Task 1A ------------------- #

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 8  # Number of neurons in first layer
    learning_rate = 0.0001  # Learning rate
    bs = 8  # batch size
    n_ep = 150  # Number of epochs
    dropout = 0  # 0 false, 1 true
    batch_norm = 0  # 0 false, 1 true
    spat_dropout = 0  # 0 false, 1 true

    # --- CNN wo/ dropout --- #
    alexnet1a1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1a1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- CNN w/dropout --- #
    dropout = 1  # 0 false, 1 true

    alexnet1a2 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1a2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 1B ------------------- #
    batch_norm = 1

    # # --- CNN wo/ dropout --- #
    dropout = 0  # 0 false, 1 true
    alexnet1b1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1b1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # # --- CNN w/ dropout --- #
    dropout = 1  # 0 false, 1 true
    alexnet1b2 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1b2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 1C ------------------- #
    dropout = 1  # 0 false, 1 true
    n_ep = 80  # Number of epochs
    learning_rate = 0.00001  # Learning rate

    # # --- CNN wo/ batch_norm --- #
    batch_norm = 0
    alexnet1c1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1c1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # # --- CNN w/ batch_norm --- #
    batch_norm = 1
    alexnet1c2 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1c2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 1D ------------------- #
    n_ep = 150  # Number of epochs

    # # --- CNN wo/ batch_norm --- #
    batch_norm = 0
    alexnet1d1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1d1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # # --- CNN w/ batch_norm --- #
    batch_norm = 1
    alexnet1d1 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_adam(alexnet1d1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)
