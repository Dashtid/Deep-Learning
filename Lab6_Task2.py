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

from functions.dataloader import load_streamlines
from functions.training_tools import MyBatchGenerator
from functions.networks import reg_model, plot_history

if __name__ == "__main__":
    # --------------- Task 2 ------------ #

    dataPath = '/DL_course_data/Lab5/HCP_lab/'
    subjects_list = os.listdir(dataPath)

    val_subjects_list = [0]
    train_subjects_list = subjects_list[0:3]
    val_subjects_list[0] = subjects_list[3]

    bundles_list = ['CST_left', 'CST_right']

    n_tracts_per_bundle = 20  # Number of tracts per bundle

    x_train, y_train = load_streamlines(dataPath, train_subjects_list, bundles_list, n_tracts_per_bundle)
    x_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list, n_tracts_per_bundle)

    lr = 0.001             # Learning rate
    bs = 1                 # Batch size
    n_ep = 50              # Number of epochs
    input_size = None      # Size of input (varies)
    input_dimensions = 3   # Dimensions of input
    n_units = 5            # Number of neurons
    bd = 1                 # Switch that turns on bi-directional ( 1 = ON / 0 = OFF)

    input_layer = Input(batch_shape=(bs, input_size, input_dimensions))
    model = reg_model(n_units, input_layer, bd)

    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=lr),
                  metrics=["binary_accuracy"])

    model_hist = model.fit_generator(MyBatchGenerator(x_train, y_train, batch_size=bs),
                                     epochs=n_ep, validation_data=MyBatchGenerator(x_val, y_val,
                                                                                   batch_size=bs),
                                     validation_steps=len(x_val))

    plot_history(model_hist)
