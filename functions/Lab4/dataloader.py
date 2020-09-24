import os
import re
import numpy as np

from random import shuffle
from skimage.io import imread
from skimage.transform import resize


def get_data(data_path, data_list, img_h, img_w):
    # """
    # Parameters
    # ----------
    # train_data_path : Str
    # Path to the data directory
    # train_list : List
    # A list containing the name of the images.
    # img_h : Int
    # image height to be resized to.
    # img_w : Int
    # image width to be resized to.
    # Returns
    # -------
    # img_labels : Nested List
    # A nested list containing the loaded images along with their
    # corresponding labels.
    # """
    img_labels = []
    for item in enumerate(data_list):
        img = imread(os.path.join(data_path, item[1]), as_gray=True)
        img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
        img_labels.append([np.array(img), 1])
        if item[0] % 100 == 0:
            print('Reading: {0}/{1} of train images'.format(item[0], len(data_list)))

    return img_labels


def get_data_arrays(nested_list, img_h, img_w):
    # """
    # Parameters
    # ----------
    # nested_list : nested list
    # nested list of image arrays with corresponding class labels.
    # img_h : Int
    # Image height.
    # img_w : Int
    # Image width.
    # -------
    # img_arrays : Numpy array
    # 4D Array with the size of (n_data,img_h,img_w, 1)
    # label_arrays : Numpy array
    # 1D array with the size (n_data).
    # """

    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype=np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype=np.int32)

    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind] = nested_list[ind][1]

    img_arrays = np.expand_dims(img_arrays, axis=3)
    return img_arrays, label_arrays


def get_train_test_arrays(train_data_path, test_data_path, train_list,
                          test_list, img_h, img_w):
    # """
    # Get the directory to the train and test sets, the files names and
    # the size of the image and return the image and label arrays for
    # train and test sets.
    # """
    img_data = get_data(train_data_path, train_list, img_h, img_w)
    msk_data = get_data(test_data_path, test_list, img_h, img_w)

    img, grbg = get_data_arrays(img_data, img_h, img_w)
    msk, grbg = get_data_arrays(msk_data, img_h, img_w)
    del img_data
    del msk_data
    x_data = img / np.max(img)
    y_data = msk / np.max(msk)

    return x_data, y_data


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


def load_img(path_list, size, mask):
    img_data = np.zeros((len(path_list), size, size, 1), dtype='float')
    for i in range(len(path_list)):
        img = imread(path_list[i], 0)
        img = resize(img[:, :], (size, size))
        img = img.reshape(size, size) / 255.
        if mask:
            img[img > 0] = 1
            img[img != 1] = 0
        img_data[i, :, :, 0] = img

    return img_data


def load_msk(path_list, size):
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
