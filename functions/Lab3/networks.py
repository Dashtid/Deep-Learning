import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Convolution2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure


def alexnet(img_ch, img_width, img_height, n_base, dropout, batch_norm, spat_dropout):
    model = Sequential()

    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if spat_dropout:
        model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if spat_dropout:
        model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if spat_dropout:
        model.add(SpatialDropout2D(0.1))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if spat_dropout:
        model.add(SpatialDropout2D(0.1))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if spat_dropout:
        model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def vgg16(img_ch, img_width, img_height, n_base, batch_norm):
    model = Sequential()
    # base
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*2
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*4
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Dense 64
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.summary()
    return model


def MLP(img_width, img_height, img_ch):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, input_shape=(img_width, img_height, img_ch), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #     model.summary()
    return model


def vgg16_2(img_ch, img_width, img_height, n_base):
    model = Sequential()

    # n_base
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # n_base * 2
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # n_base * 4
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # n_base * 8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # n_base * 8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='Last_ConvLayer'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense 64
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()
    return model
