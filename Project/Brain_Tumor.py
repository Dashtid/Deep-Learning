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
from Project.Functions.Networks import get_unet, get_unet_deep
from Project.Functions.Training_tools import dice_coef, dice_loss, precision, recall, plot_history, data_generator, \
    trainer, calc_n_avarage_dice

# -------------------------------------------- Brain - Growth -------------------------------------------- #

if __name__ == "__main__":
    # Setting paths to directories
    train_dir = '/content/drive/My Drive/Medical DL project/Project_data/training_data_v2/brain-tumor'
    val_dir = '/content/drive/My Drive/Medical DL project/Project_data/validation_data_v2/brain-tumor'
    test_dir = '/content/drive/My Drive/Medical DL project/Project_data/test/brain-tumor'

    expert = 1
    task = 1

    train_img_path, train_msk_path = get_file_list(train_dir, 'task0' + str(task) + '_' + 'seg0' + str(expert))
    val_img_path, val_msk_path = get_file_list(val_dir, 'task0' + str(task) + '_' + 'seg0' + str(expert))

    test_img_path, test_msk_path = get_file_list(test_dir, 'task0' + str(task) + '_' + 'seg0' + str(expert))

    x_train = load_img_array(train_img_path, 256, task)
    y_train = load_img_array(train_msk_path, 256, task)
    x_val = load_img_array(val_img_path, 256, task)
    y_val = load_img_array(val_msk_path, 256, task)
    x_test = load_img_array(test_img_path, 256, task)
    y_test = load_img_array(test_msk_path, 256, task)
