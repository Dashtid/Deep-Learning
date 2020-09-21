import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Convolution2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure

from functions.Lab4.networks import get_unet

# --- Task 1 --- #

if __name__ == "__main__":

    # -------------------------------------- Task 1A -------------------------------------- #

    # Setting parameters
    base = 8  # Number of feature maps in convolutional layer
    img_w = 256  # Image width
    img_h = 256  # Image height
    img_ch = 1  # Number of image channels
    bs = 8  # Batch size
    lr = 0.0001  # Learning rate
    batch_norm = 0  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    spat_dropout = 0  # On/Off switch for spatial dropout layer, 0 = False, 1 = True

    clf = get_unet(base, img_w, img_h, img_ch, batch_norm, spat_dropout)

    # -------------------------------------- Task 1B -------------------------------------- #

    # -------------------------------------- Task 1C -------------------------------------- #

    # -------------------------------------- Task 1D -------------------------------------- #

    # -------------------------------------- Task 1E -------------------------------------- #
