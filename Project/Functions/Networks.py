import os
import re
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


def deconv_block(base, conc_layer, layer, batch_norm, dropout, dr_rate):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)
    layer_conc = concatenate([conc_layer, layer_convT])
    if dropout:
        layer_d = Dropout(dr_rate)(layer_conc)
        layer_b = conv_block(base, layer_d, batch_norm)
    else:
        layer_b = conv_block(base, layer_conc, batch_norm)

    return layer_b


def get_unet(base, img_size, img_ch, batch_norm, dropout, dr_rate):
    layer_inp = Input(shape=(img_size, img_size, img_ch))
    layer_b1 = conv_block(base, layer_inp, batch_norm)

    layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    if dropout:
        layer_d1 = Dropout(dr_rate)(layer_mp1)
        layer_b2 = conv_block(base * 2, layer_d1, batch_norm)
    else:
        layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)

    layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    if dropout:
        layer_d2 = Dropout(dr_rate)(layer_mp2)
        layer_b3 = conv_block(base * 4, layer_d2, batch_norm)
    else:
        layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)

    layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    if dropout:
        layer_d3 = Dropout(dr_rate)(layer_mp3)
        layer_b4 = conv_block(base * 8, layer_d3, batch_norm)
    else:
        layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)

    layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)
    # Bottle-neck
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout, dr_rate)

    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout, dr_rate)

    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout, dr_rate)

    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout, dr_rate)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    model = Model(inputs=layer_inp, outputs=layer_out)

    model.summary()

    return model


def deconv_block_sd(base, conc_layer, layer, batch_norm, dropout, dr_rate):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)
    layer_conc = concatenate([conc_layer, layer_convT])
    if dropout:
        layer_d = SpatialDropout2D(dr_rate)(layer_conc)
        layer_b = conv_block(base, layer_d, batch_norm)
    else:
        layer_b = conv_block(base, layer_conc, batch_norm)

    return layer_b


def get_unet_sd(base, img_size, img_ch, batch_norm, dropout, dr_rate):
    layer_inp = Input(shape=(img_size, img_size, img_ch))
    layer_b1 = conv_block(base, layer_inp, batch_norm)

    layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    if dropout:
        layer_d1 = SpatialDropout2D(dr_rate)(layer_mp1)
        layer_b2 = conv_block(base * 2, layer_d1, batch_norm)
    else:
        layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)

    layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    if dropout:
        layer_d2 = SpatialDropout2D(dr_rate)(layer_mp2)
        layer_b3 = conv_block(base * 4, layer_d2, batch_norm)
    else:
        layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)

    layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    if dropout:
        layer_d3 = SpatialDropout2D(dr_rate)(layer_mp3)
        layer_b4 = conv_block(base * 8, layer_d3, batch_norm)
    else:
        layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)

    layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)
    # Bottle-neck
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    layer_db1 = deconv_block_sd(base * 8, layer_b4, layer_b5, batch_norm, dropout, dr_rate)

    layer_db2 = deconv_block_sd(base * 4, layer_b3, layer_db1, batch_norm, dropout, dr_rate)

    layer_db3 = deconv_block_sd(base * 2, layer_b2, layer_db2, batch_norm, dropout, dr_rate)

    layer_db4 = deconv_block_sd(base, layer_b1, layer_db3, batch_norm, dropout, dr_rate)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    model = Model(inputs=layer_inp, outputs=layer_out)

    model.summary()

    return model


def deconv_block_lstm(base, conc_layer, layer, batch_norm, dropout, img_size):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)

    x1 = Reshape(target_shape=(1, np.int32(img_size), np.int32(img_size), base))(conc_layer)
    x2 = Reshape(target_shape=(1, np.int32(img_size), np.int32(img_size), base))(layer_convT)

    layer_conc = concatenate([x1, x2], axis=1)
    if dropout:
        layer_lstm = ConvLSTM2D(np.int32(base / 2), (3, 3), padding='same', return_sequences=False, go_backwards=True)(
            layer_conc)
        layer_d = Dropout(0.2)(layer_lstm)
        layer_b = conv_block(base, layer_d, batch_norm)
    else:
        layer_lstm = ConvLSTM2D(np.int32(base / 2), (3, 3), padding='same', return_sequences=False, go_backwards=True)(
            layer_conc)
        layer_b = conv_block(base, layer_lstm, batch_norm)

    return layer_b


def get_unet_lstm(base, img_size, img_ch, batch_norm, dropout):
    layer_inp = Input(shape=(img_size, img_size, img_ch))
    layer_b1 = conv_block(base, layer_inp, batch_norm)

    layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    if dropout:
        layer_d1 = Dropout(0.2)(layer_mp1)
        layer_b2 = conv_block(base * 2, layer_d1, batch_norm)
    else:
        layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)

    layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    if dropout:
        layer_d2 = Dropout(0.2)(layer_mp2)
        layer_b3 = conv_block(base * 4, layer_d2, batch_norm)
    else:
        layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)

    layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    if dropout:
        layer_d3 = Dropout(0.2)(layer_mp3)
        layer_b4 = conv_block(base * 8, layer_d3, batch_norm)
    else:
        layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)

    layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)
    # Bottle-neck
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    layer_db1 = deconv_block_lstm(base * 8, layer_b4, layer_b5, batch_norm, dropout, img_size / 8)

    layer_db2 = deconv_block_lstm(base * 4, layer_b3, layer_db1, batch_norm, dropout, img_size / 4)

    layer_db3 = deconv_block_lstm(base * 2, layer_b2, layer_db2, batch_norm, dropout, img_size / 2)

    layer_db4 = deconv_block_lstm(base, layer_b1, layer_db3, batch_norm, dropout, img_size)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    model = Model(inputs=layer_inp, outputs=layer_out)

    model.summary()

    return model


def conv_block_res(layer, base, s):
    x = Conv2D(base, (1, 1), strides=(s, s))(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(base, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(base * 4, (1, 1))(x)
    x = BatchNormalization()(x)

    # Skip connection
    x_skip = Conv2D(base * 4, (1, 1), strides=(s, s))(layer)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x


def id_block(layer, base):
    x = Conv2D(base, (1, 1))(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(base, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(base * 4, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, layer])
    x = Activation('relu')(x)

    return x


def ResNet50(base, img_size, img_ch, num_class):
    input_layer = Input((img_size, img_size, img_ch))

    # Zero-Padding
    x = ZeroPadding2D((3, 3))(input_layer)

    x = Conv2D(base, (7, 7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv and identity blocks
    x = conv_block_res(x, base, 1)
    x = id_block(x, base)
    x = id_block(x, base)

    x = conv_block_res(x, base * 2, 2)
    x = id_block(x, base * 2)
    x = id_block(x, base * 2)
    x = id_block(x, base * 2)

    x = conv_block_res(x, base * 4, 2)
    x = id_block(x, base * 4)
    x = id_block(x, base * 4)
    x = id_block(x, base * 4)
    x = id_block(x, base * 4)
    x = id_block(x, base * 4)

    x = conv_block_res(x, base * 8, 2)
    x = id_block(x, base * 8)
    x = id_block(x, base * 8)

    x = AveragePooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(num_class)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    model.summary()

    return model


def get_unet_deep(base, img_size, img_ch, batch_norm, dropout, dr_rate):
    layer_inp = Input(shape=(img_size, img_size, img_ch))
    layer_b1 = conv_block(base, layer_inp, batch_norm)

    layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    if dropout:
        layer_d1 = Dropout(dr_rate)(layer_mp1)
        layer_b2 = conv_block(base * 2, layer_d1, batch_norm)
    else:
        layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)

    layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    if dropout:
        layer_d2 = Dropout(dr_rate)(layer_mp2)
        layer_b3 = conv_block(base * 4, layer_d2, batch_norm)
    else:
        layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)

    layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    if dropout:
        layer_d3 = Dropout(dr_rate)(layer_mp3)
        layer_b4 = conv_block(base * 8, layer_d3, batch_norm)
    else:
        layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)

    layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)

    if dropout:
        layer_d4 = Dropout(dr_rate)(layer_mp4)
        layer_b5 = conv_block(base * 16, layer_d4, batch_norm)
    else:
        layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    layer_mp5 = MaxPooling2D(pool_size=(2, 2))(layer_b5)
    # Bottle-neck
    layer_b6 = conv_block(base * 32, layer_mp5, batch_norm)

    layer_db1 = deconv_block(base * 16, layer_b5, layer_b6, batch_norm, dropout, dr_rate)

    layer_db2 = deconv_block(base * 8, layer_b4, layer_db1, batch_norm, dropout, dr_rate)

    layer_db3 = deconv_block(base * 4, layer_b3, layer_db2, batch_norm, dropout, dr_rate)

    layer_db4 = deconv_block(base * 2, layer_b2, layer_db3, batch_norm, dropout, dr_rate)

    layer_db5 = deconv_block(base, layer_b1, layer_db4, batch_norm, dropout, dr_rate)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db5)
    layer_out = Activation('sigmoid')(layer_conv2)

    model = Model(inputs=layer_inp, outputs=layer_out)

    model.summary()

    return model
