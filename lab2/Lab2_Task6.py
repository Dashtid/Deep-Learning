import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout

from functions.networks import alexnet, alexnet_with_dropout
from functions.training_tools import train_with_adam, train_with_adam_and_hinge, train_with_rmsprop, train_with_sgd
from functions.dataloader import get_train_test_arrays

# --- Task 6 --- #

if __name__ == "__main__":
    skin_labels_string_list = ['Mel', 'Nev']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/Lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w, skin_labels_string_list)

    # ------------------- Task 6A ------------------- #

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 32  # Number of neurons in first layer
    learning_rate = 0.0001  # Learning rate
    bs = 8  # batch size
    n_ep = 50  # Number of epochs

    alex6a = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6a, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 6B ------------------- #

    # ---- Base 16 ---- #
    base = 16  # Number of neurons in first layer
    alex6b1 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6b1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ---- Base 8 ---- #
    base = 8  # Number of neurons in first layer
    alex6b2 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6b2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- Adding Dropout --- #
    alex6b3 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(alex6b3, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- 150 Epochs --- #
    n_ep = 150  # Number of epochs
    alex6b4 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(alex6b4, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 6C ------------------- #

    # --- LR = 1e^-5 --- #
    learning_rate = 0.000001  # Learning rate
    alex6c1 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6c1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- 350 Epochs --- #
    n_ep = 350  # Number of epochs
    alex6c2 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6c2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 6D ------------------- #

    # --- 150 Epochs & 2 BS --- #
    n_ep = 150  # Number of epochs
    bs = 2  # batch size
    alex6d1 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6d1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- 4 BS --- #
    bs = 4  # batch size
    alex6d2 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6d2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- 8 BS --- #
    bs = 8  # batch size
    alex6d3 = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex6d3, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 6E ------------------- #
    base = 32
    learning_rate = 0.00001
    n_ep = 100

    # --- Adam --- #
    alex6e1 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(alex6e1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- SGD --- #
    alex6e2 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_sgd(alex6e2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- RMSprop --- #
    alex6e3 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_rmsprop(alex6e3, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 6F ------------------- #

    # --- BCE --- #
    alex6f1 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(alex6f1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # Changing labels
    y_test[y_test == 0] = -1
    y_train[y_train == 0] = -1

    # --- Hinge --- #
    alex6f1 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam_and_hinge(alex6f1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)
