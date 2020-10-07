import os
import re
import cv2
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, ConvLSTM2D, Dense, Flatten, MaxPooling2D, Conv2D, Activation, \
    Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM, Bidirectional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

if __name__ == "__main__":

    # --------------- Task 2 ------------ #

    Metric = [dice_coef, precision, recall]