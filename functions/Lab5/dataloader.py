import os
import re
import numpy as np

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import backend as K
import copy


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def get_file_list(data_path):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            mypath = os.path.join(root, file)
            img_list.append(mypath)
    img_list.sort(key=natural_sort_key)
    return img_list


def shuffle_n_split_data(img_list, msk_list, frac):
    comb = list(zip(img_list, msk_list))
    shuffle(comb)
    img_list[:], msk_list[:] = zip(*comb)

    length_split = int(frac * len(img_list))

    train_img = img_list[:length_split]
    train_msk = msk_list[:length_split]
    val_img = img_list[length_split:]
    val_msk = msk_list[length_split:]
    return train_img, train_msk, val_img, val_msk


def get_pathlist_from_file(path):
    img_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            img_list.append(os.path.join(root, file))
    return img_list


def load_img(path_list, size, mask):
    img_data = np.zeros((len(path_list), size, size, 1), dtype='float')
    for i in range(len(path_list)):
        img = imread(path_list[i], 0)
        img = resize(img[:, :], (size, size))
        img = img.reshape(size, size) / np.max(img)
        if mask:
            img[img > 0] = 1
            img[img != 1] = 0
        img_data[i, :, :, 0] = img
    return img_data


def load_msk2(path_list, size):
    img_data = np.zeros((len(path_list), size, size, 1), dtype='float')
    for i in range(len(path_list)):
        img = imread(path_list[i], 0)
        img = resize(img[:, :], (size, size))
        img[img == 156] = 1
        img[img == 251] = 2
        img[img > 2] = 0
        img[img < 1] = 0
        img_data[i, :, :, 0] = img
    return img_data


def k_split(img_list, msk_list, folds):
    split_list = list()
    length = len(img_list)
    fold_size = int(length / folds)
    for i in range(0, folds):
        val_img = list()
        val_msk = list()
        img_copy = copy.copy(img_list)
        msk_copy = copy.copy(msk_list)
        for j in range(i * fold_size, (i + 1) * fold_size):
            if i == (folds - 1):
                val_img = img_copy[i * fold_size:]
                val_msk = msk_copy[i * fold_size:]
            else:
                val_img.append(img_copy.pop(j))
                val_msk.append(msk_copy.pop(j))

        train_img = img_copy
        train_msk = msk_copy
        split = (train_img, train_msk, val_img, val_msk)
        split_list.append(split)
    return split_list


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


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)