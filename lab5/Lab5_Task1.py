import os
import re
import copy
import warnings
import numpy as np
import tensorflow as tf

from random import shuffle
from skimage import exposure
from skimage.io import imread
from skimage.transform import resize, rescale, rotate
from matplotlib import pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Convolution2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

# --------------- Task 1 ------------ #

if __name__ = "__main__":
