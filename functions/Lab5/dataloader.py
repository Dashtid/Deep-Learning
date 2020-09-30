import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import copy

import os
import random
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
import cv2


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

        val_img = img_copy[i * fold_size:(i + 1) * fold_size]
        val_msk = msk_copy[i * fold_size:(i + 1) * fold_size]
        del img_copy[i * fold_size:(i + 1) * fold_size]
        del msk_copy[i * fold_size:(i + 1) * fold_size]

        train_img = img_copy
        train_msk = msk_copy
        split = (train_img, train_msk, val_img, val_msk)
        split_list.append(split)
    return split_list


def boundry_mask(msk_list, size):
    new_msk_list = np.zeros((len(msk_list), size, size, 1), dtype='float')
    i = 0
    for msk in msk_list:
        kernel = np.ones((2, 2), np.uint8)
        input_image = msk[:, :, 0]
        erosion_image = cv2.erode(input_image, kernel, iterations=5)
        dilation_image = cv2.dilate(input_image, kernel, iterations=5)
        new = dilation_image - erosion_image
        new_msk_list[i, :, :, 0] = new
        i += 1
    return new_msk_list


def train_gen(img_train, label_train, n_train_sample, batch_size, image_size):
    """
    img_train: a list containing full directory of training images
    label_train: a list containing full directory of training masks
    n_train_sample: len(img_train)
    batch_size: integer value of batch size
    image_size: an integer value e.g, 240
    """
    data_gen_args = dict(rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         cval=0,
                         zoom_range=0.1,
                         horizontal_flip=True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)

    while True:
        samples = list(zip(img_train, label_train))
        random.shuffle(samples)
        sample_img, sample_label = zip(*samples)
        sample_img = list(sample_img)
        sample_label = list(sample_label)

        for ind in (range(0, n_train_sample, batch_size)):
            batch_img = sample_img[ind:ind + batch_size]
            batch_label = sample_label[ind:ind + batch_size]
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            length = len(batch_img)

            if length == batch_size:
                pass

            else:
                for tmp in range(batch_size - length):
                    batch_img.append(batch_img[-1])
                    batch_label.append(batch_label[-1])

            x_train = np.empty([batch_size, image_size, image_size], dtype='float32')
            y_train = np.empty([batch_size, image_size, image_size], dtype='float32')

            for ix in range(len(batch_img)):
                img_sample = batch_img[ix]
                label_sample = batch_label[ix]
                img_array = load_img_array(img_sample, image_size)
                label_array = load_img_array(label_sample, image_size)
                x_train[ix] = img_array
                y_train[ix] = label_array

            x_train = np.expand_dims(x_train, axis=3)
            y_train = np.expand_dims(y_train, axis=3)

            image_generator = image_datagen.flow(x_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)

            label_generator = label_datagen.flow(y_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            img_gen = image_generator.next()
            label_gen = label_generator.next()
            yield img_gen, label_gen


def train_gen_weight(img_train, label_train, n_train_sample, batch_size, image_size):
    """
    img_train: a list containing full directory of training images
    label_train: a list containing full directory of training masks
    n_train_sample: len(img_train)
    batch_size: integer value of batch size
    image_size: an integer value e.g, 240
    """
    data_gen_args = dict(rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         cval=0,
                         zoom_range=0.1,
                         horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)
    weight_datagen = ImageDataGenerator(**data_gen_args)
    while True:
        samples = list(zip(img_train, label_train))
        shuffle(samples)
        sample_img, sample_label = zip(*samples)
        sample_img = list(sample_img)
        sample_label = list(sample_label)
        for ind in (range(0, n_train_sample, batch_size)):
            batch_img = sample_img[ind:ind + batch_size]
            batch_label = sample_label[ind:ind + batch_size]
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            length = len(batch_img)
            if length == batch_size:
                pass
            else:
                for tmp in range(batch_size - length):
                    batch_img.append(batch_img[-1])
                    batch_label.append(batch_label[-1])
            x_train = np.empty([batch_size, image_size, image_size], dtype='float32')
            y_train = np.empty([batch_size, image_size, image_size], dtype='float32')

            for ix in range(len(batch_img)):
                img_sample = batch_img[ix]
                label_sample = batch_label[ix]
                img_array = load_img_array(img_sample, image_size)
                label_array = load_img_array(label_sample, image_size)
                x_train[ix] = img_array
                y_train[ix] = label_array

            x_train = np.expand_dims(x_train, axis=3)
            y_train = np.expand_dims(y_train, axis=3)
            weight_train = boundry_mask(y_train, image_size)

            image_generator = image_datagen.flow(x_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            label_generator = label_datagen.flow(y_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)

            weight_generator = weight_datagen.flow(weight_train, shuffle=False,
                                                   batch_size=batch_size,
                                                   seed=1)
            train_generator = combine_generator(image_generator, label_generator, weight_generator)

            return train_generator


def load_img_array(img_dir, image_size):
    img_arr = cv2.imread(img_dir, 0)
    img_arr = cv2.resize(img_arr[:, :], (image_size, image_size))
    img_arr = img_arr / 255.
    return img_arr


def combine_generator(gen1, gen2, gen3):
    while True:
        x = gen1.next()
        y = gen2.next()
        w = gen3.next()
        yield [x, w], y
