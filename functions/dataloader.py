import os
from random import shuffle

import numpy as np
from skimage.io import imread
from skimage.transform import resize


# Loading data
def gen_labels(im_name, label_list):
    #   '''
    # Parameters
    # ----------
    # im_name : Str
    # The image file name.
    # Returns
    # -------
    # Label : Numpy array
    # Class label of the filename name based on its pattern.
    # '''
    i = 0
    for lbl in label_list:
        if lbl in im_name:
            label = np.array([i])
            break
        i += 1
    return label


def get_data(data_path, data_list, img_h, img_w, label_string_list):
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
        img = imread(os.path.join(data_path, item[1]), as_gray=True)  # "as_grey"
        img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
        img_labels.append([np.array(img), gen_labels(item[1], label_string_list)])
        if item[0] % 100 == 0:
            print('Reading: {0}/{1} of train images'.format(item[0], len(data_list)))
    shuffle(img_labels)
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
                          test_list, img_h, img_w, label_string_list):
    # """
    # Get the directory to the train and test sets, the files names and
    # the size of the image and return the image and label arrays for
    # train and test sets.
    # """

    train_data = get_data(train_data_path, train_list, img_h, img_w, label_string_list)
    test_data = get_data(test_data_path, test_list, img_h, img_w, label_string_list)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del train_data
    del test_data
    return train_img, test_img, train_label, test_label
