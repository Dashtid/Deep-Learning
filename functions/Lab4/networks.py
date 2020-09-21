import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Convolution2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Conv2D, Conv2DTranspose, Concatenate, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure


# Unet
def conv_block(base, layer, batch_norm):
    layer_conv = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer)
    if batch_norm:
        layer_bn = BatchNormalization()(layer_conv)
        layer_act = Activation('relu')(layer_bn)
    else:
        layer_act = Activation('relu')(layer_conv)

    layer_conv2 = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_act)
    if batch_norm:
        layer_bn2 = BatchNormalization()(layer_conv2)
        layer_act2 = Activation('relu')(layer_bn2)
    else:
        layer_act2 = Activation('relu')(layer_conv2)

    return layer_act2


def deconv_block(base, conc_layer, layer, batch_norm, dropout):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer_conc = concatenate([conc_layer, layer_convT])
    if dropout:
        layer_sd = Dropout(0.2)(layer_conc)
        layer_b = conv_block(base, layer_sd, batch_norm)
    else:
        layer_b = conv_block(base, layer_conc, batch_norm)

    return layer_b


def get_unet(base, img_w, img_h, img_ch, batch_norm, dropout):

    # Defining the input layer
    layer_inp = Input(shape=(img_h, img_w, img_ch))

    # --- Contraction Phase --- #
    layer_b1 = conv_block(base, layer_inp, batch_norm)
    if dropout:
        layer_sd1 = Dropout(0.2)(layer_b1)
        layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_sd1)
    else:
        layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)
    if dropout:
        layer_sd2 = Dropout(0.2)(layer_b2)
        layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_sd2)
    else:
        layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)
    if dropout:
        layer_sd3 = Dropout(0.2)(layer_b3)
        layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_sd3)
    else:
        layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)
    if dropout:
        layer_sd4 = Dropout(0.2)(layer_b4)
        layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_sd4)
    else:
        layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)

    # --- Bottle-neck Phase --- #
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    # --- Expansion Phase --- #
    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout)
    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout)
    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout)
    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    # --- Creating the model --- #
    model = Model(inputs=layer_inp, outputs=layer_out)
    model.summary()
    return model
