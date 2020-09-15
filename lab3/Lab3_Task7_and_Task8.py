import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure

from functions.Lab3.networks import alexnet, vgg16, vgg16_2
from functions.Lab3.training_tools import train_with_generator
from functions.dataloader import datagenerator, show_paired

# --- Task 7 & 8 --- #

if __name__ == "__main__":

    # --------------------------------------Task 7 -------------------------------------- #

