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

from Project.Functions.Dataloader import get_file_list, load_img_array


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_output = true_positives / (possible_positives + K.epsilon())
    return recall_output


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


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def data_generator(x_data, y_data, batch_size, seed, flip=1, **kwargs):
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=flip)

    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(x_data, augment=True, seed=seed)
    label_datagen.fit(y_data, augment=True, seed=seed)

    image_generator = image_datagen.flow(x_data, shuffle=False,
                                         batch_size=batch_size,
                                         seed=seed)
    label_generator = label_datagen.flow(y_data, shuffle=False,
                                         batch_size=batch_size,
                                         seed=seed)

    generator = (pair for pair in zip(image_generator, label_generator))
    return generator


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
    metric = [dice_coef, precision, recall]

    network.compile(loss=dice_loss,
                    optimizer=Adam(lr=lr),
                    metrics=metric)

    model_hist = network.fit_generator(training_generator,
                                       steps_per_epoch=len(x_train) // bs,
                                       validation_data=validation_generator,
                                       validation_steps=len(x_val) // bs,
                                       epochs=n_ep)
    return model_hist, network


def calc_n_average_dice(preds, n_masks, path, task, img_size):
    n_test_im = len(preds[0])
    mask_list = np.zeros((n_masks, n_test_im, img_size, img_size, 1))

    for i in range(n_masks):
        test_img_path, test_msk_path = get_file_list(path, 'task0' + str(task) + '_' + 'seg0' + str(i + 1))
        y_test = load_img_array(test_msk_path, img_size, 1, 0)
        mask_list[i] = y_test

    # Average operation
    preds = np.sum(preds, axis=0)
    preds = preds / n_masks
    mask_list = np.sum(mask_list, axis=0)
    mask_list = mask_list / n_masks
    dice_scores = np.zeros((9, n_test_im))

    # Running the average operation for all threshold
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
