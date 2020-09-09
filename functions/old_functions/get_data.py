import os
from random import shuffle

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from functions.old_functions.gen_labels import gen_labels


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
