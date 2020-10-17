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
    MaxPooling2D, Conv2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def load_img_array(path_list, size):
    img_data = np.zeros((len(path_list), size, size, 1), dtype='float')
    for i in range(len(path_list)):
        img_arr = sitk.ReadImage(path_list[i])
        img_arr = sitk.GetArrayFromImage(img_arr)
        img_arr = img_arr / np.max(img_arr)
        img_data[i, :, :, 0] = img_arr
    return img_data


_nsre = re.compile('([0-9]+)')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def get_file_list(data_path, msk_name):
    img_list = []
    msk_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file == 'image.nii.gz':
                mypath = os.path.join(root, file)
                img_list.append(mypath)
            if msk_name in file:
                mypath = os.path.join(root, file)
                msk_list.append(mypath)

    img_list.sort(key=natural_sort_key)
    msk_list.sort(key=natural_sort_key)
    return img_list, msk_list


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
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)
    layer_conc = concatenate([conc_layer, layer_convT])
    if dropout:
        layer_d = Dropout(0.2)(layer_conc)
        layer_b = conv_block(base, layer_d, batch_norm)
    else:
        layer_b = conv_block(base, layer_conc, batch_norm)

    return layer_b


def get_unet(base, img_size, img_ch, batch_norm, dropout):
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

    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout)

    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout)

    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout)

    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout)

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


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


Metric = [dice_coef, precision, recall]


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


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def data_generator(x_data, y_data, batch_size):
    data_gen_args = dict(rotation_range=10.,
                         # zoom_range=0.1,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)

    # image_datagen.fit(x_data)
    image_datagen.fit(x_data, augment=True, seed=1)
    label_datagen.fit(y_data, augment=True, seed=1)

    image_generator = image_datagen.flow(x_data, shuffle=False,
                                         batch_size=batch_size,
                                         seed=5)
    label_generator = label_datagen.flow(y_data, shuffle=False,
                                         batch_size=batch_size,
                                         seed=5)

    data_generator = (pair for pair in zip(image_generator, label_generator))
    return data_generator


def plot_history(net_history):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(net_history.history["loss"], label="loss")
    plt.plot(net_history.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(net_history.history["val_loss"]),
             np.min(net_history.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()


def trainer(network, n_ep, lr, bs, x_train, x_val, training_generator, validation_generator):
    network.compile(loss=dice_loss,
                    optimizer=Adam(lr=lr),
                    metrics=Metric)

    model_hist = network.fit_generator(training_generator,
                                       steps_per_epoch=len(x_train) // bs,
                                       validation_data=validation_generator,
                                       validation_steps=len(x_val) // bs,
                                       epochs=n_ep)
    return model_hist, network


def calc_n_avarage_dice(preds, n_masks, path):
    mask_list = np.zeros((n_masks, 5, 256, 256, 1))
    for i in range(n_masks):
        test_img_path, test_msk_path = get_file_list(path, 'seg0' + str(i + 1))

        y_test = load_img_array(test_msk_path, 256)
        mask_list[i] = y_test

    # Avarage
    preds = np.sum(preds, axis=0)
    preds = preds / n_masks

    mask_list = np.sum(mask_list, axis=0)
    mask_list = mask_list / n_masks

    dice_scores = np.zeros((9, 5))
    for i in range(9):
        mask_list_copy = copy.copy(mask_list)
        preds_copy = copy.copy(preds)
        thresh = (i + 1) / 10
        mask_list_copy[mask_list_copy < thresh] = 0
        mask_list_copy[mask_list_copy >= thresh] = 1

        preds_copy[preds_copy < thresh] = 0
        preds_copy[preds_copy >= thresh] = 1

        for j in range(len(preds_copy)):
            dice_scores[i][j] = dice_coef(mask_list_copy[j], preds_copy[j])

    print(dice_scores)
    dice_average = np.average(dice_scores)

    return dice_average
