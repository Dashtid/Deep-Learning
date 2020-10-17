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

from functions.Lab6.networks import reg_model, plot_history

if __name__ == "__main__":

    # --------------- Task 1 ------------ #

    # Loading in csv files:
    dataset_train = pd.read_csv('/DL_course_data/Lab5/train_data_stock.csv')
    dataset_val = pd.read_csv('/DL_course_data/Lab5/val_data_stock.csv')

    # Reversing data so that they go from oldest to newest
    dataset_train = dataset_train.iloc[::-1]
    dataset_val = dataset_val.iloc[::-1]

    # Concatenating the training and test datasets
    dataset_total = pd.concat((dataset_train['Open'], dataset_val['Open']), axis=0)

    # Selecting the values from the “Open” column as the variables to be predicted
    training_set = dataset_train.iloc[:, 1:2].values
    val_set = dataset_val.iloc[:, 1:2].values

    # Normalizing
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    #  --- Split training data into T time steps --- #

    T = 60  # Number of time steps
    x_train = []  # List for input data
    y_train = []  # List for target data

    # Filling the lists
    for i in range(T, len(training_set)):
        x_train.append(training_set_scaled[i - T:i, 0])
        y_train.append(training_set_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Normalization of the validation set according to the normalization applied to the training set
    inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # --- Split validation data into T time steps --- #
    x_val = []

    for i in range(T, T + len(val_set)):
        x_val.append(inputs[i - T:i, 0])

    x_val = np.array(x_val)
    y_val = sc.transform(val_set)

    # Reshape to 3D array (format needed by LSTMs -> number of samples, time steps, input dimension)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    # ---------------- Setting up parameters ---------------- #

    lr = 0.001  # Learning rate
    bs = 16  # Batch size
    n_ep = 100  # Number of epochs
    input_size = 60  # Size of input
    input_dimensions = 1  # Dimensions of input
    n_units = 20  # Number of neurons
    bd = 0  # Switch that turns on bi-directional ( 1 = ON / 0 = OFF)

    input_layer = Input(batch_shape=(bs, input_size, input_dimensions))

    model = reg_model(n_units, input_layer, bd)

    model.compile(loss="mse",
                  optimizer=Adam(lr=lr),
                  metrics=["mae"])

    model_hist = model.fit(x_train, y_train,
                           batch_size=bs, epochs=n_ep,
                           validation_data=(x_val, y_val))

    plot_history(model_hist)
