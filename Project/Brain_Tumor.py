import os
import re
import copy
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add, AveragePooling2D, Reshape, ConvLSTM2D, Dense, Flatten, ZeroPadding2D, \
    MaxPooling2D, Conv2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM, SpatialDropout2D

from Project.Functions.Dataloader import get_file_list, load_img_array
from Project.Functions.Networks import get_unet_deep_lstm_sd
from Project.Functions.Training_tools import dice_coef, dice_loss, precision, recall, plot_history, data_generator, \
    trainer, calc_n_average_dice

# -------------------------------------------- Brain - Tumor -------------------------------------------- #

if __name__ == "__main__":
    # Setting paths to directories
    train_dir = 'Project_data/training_data_v2/brain-tumor'
    val_dir = 'Project_data/validation_data_v2/brain-tumor'
    test_dir = 'Project_data/test/brain-tumor'

    # ------------------------------------ Deep U - Net w/ Spatial SD + LSTM ------------------------------------- #

    # ----- Parameters ----- #
    base = 16  # Number of feature maps
    img_size = 256  # Size of input
    img_ch = 4  # Dimensions of input
    bs = 1  # Batch size
    lr = 0.0001  # Learning rate
    batch_norm = 1  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    n_ep = 1500  # Number of epochs
    dr_rate = 0.1  # Drop-rate
    n_masks = 3  # Number of expert segmentations
    n_test_im = 4  # Number of images to use for testing
    window_level = 0  # On/Off switch for windowing using default parameters, 0 = False, 1 = True

    # 3 networks, 4 test images
    preds = np.zeros((n_masks, n_test_im, img_size, img_size, 1))

    task = 1  # CHANGE THIS TO CHOOSE WHICH TASK YOU WANT TO RUN (task 1, task 2 or task 3)

    for i in range(n_masks):
        K.clear_session()  # Clearing weights of previous network to save memory

        # Combining paths with file names and fetching a list with images corresponding to the right expert
        train_img_path, train_msk_path = get_file_list(train_dir, 'task0' + str(task) + '_' + 'seg0' + str(i + 1))
        val_img_path, val_msk_path = get_file_list(val_dir, 'task0' + str(task) + '_' + 'seg0' + str(i + 1))
        test_img_path, test_msk_path = get_file_list(test_dir, 'task0' + str(task) + '_' + 'seg0' + str(i + 1))

        # Loading in training and validation images as an array
        x_train = load_img_array(train_img_path, img_size, img_ch, window_level)
        y_train = load_img_array(train_msk_path, img_size, img_ch, window_level)
        x_val = load_img_array(val_img_path, img_size, img_ch, window_level)
        y_val = load_img_array(val_msk_path, img_size, img_ch, window_level)
        x_test = load_img_array(test_img_path, img_size, img_ch, window_level)
        y_test = load_img_array(test_msk_path, img_size, img_ch, window_level)

        # Creating the two data-generators needed
        train_gen = data_generator(x_train, y_train, bs, 1)
        val_gen = data_generator(x_val, y_val, bs, 2)

        # Creating the network
        network = get_unet_deep_lstm_sd(base, img_size, img_ch, batch_norm, dropout, dr_rate)

        # Training the network
        hist, trained_network = trainer(network, n_ep, lr, bs, x_train, x_val, train_gen, val_gen)

        # Plotting the resulting histogram
        plot_history(hist)

        # Filling a list with our predicted mask of the test data
        preds[i] = trained_network.predict(x_test)  # One mask for each expert


    # Function that calculates the dice scores of our predicted masks and does an average on these scores
    x = calc_n_average_dice(preds, n_masks, test_dir, task, img_size)
    print(x)  # Printing the average
