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

from functions.Lab3.networks import alexnet, vgg16
from functions.Lab3.training_tools import train_with_generator
from functions.dataloader import datagenerator, show_paired

# --- Task 3 --- #

if __name__ == "__main__":

    # ------------------- Task 3A ------------------- #

    sample_dir = '/DL_course_data/lab1/X_ray/train/C4_4662.jpg'
    img = imread(sample_dir)
    row, col = img.shape

    # Scaling
    scale_factor = 0.1
    image_rescaled = rescale(img, scale_factor)
    show_paired(img, image_rescaled, "Rescaled")

    # Rotation
    angle = 90
    image_rotated = rotate(img, angle)
    show_paired(img, image_rotated, "Rotated")

    # Horizontal Flip
    horizontal_flip = img[:, ::-1]
    show_paired(img, horizontal_flip, 'Horizontal Flip')

    # Vertical Flip
    vertical_flip = img[::-1, :]
    show_paired(img, vertical_flip, 'vertical Flip')

    # Intensity rescaling
    low_bound, high_bound = 1, 50
    min_val, max_val = np.percentile(img, (low_bound, high_bound))
    better_contrast = exposure.rescale_intensity(img, in_range=(min_val, max_val))
    show_paired(img, better_contrast, 'Intensity Rescaling')

    # ------------------- Task 3B ------------------- #

    Sample = '/DL_course_data/lab1/X_ray/train/C4_4662.jpg'
    Img = imread(Sample)
    Img = np.expand_dims(Img, axis=2)
    Img = np.expand_dims(Img, axis=0)
    count = 5

    my_gen = ImageDataGenerator(rotation_range=20,
                                width_shift_range=0.2,
                                horizontal_flip=True)

    fix, ax = plt.subplots(1, count + 1, figsize=(14, 2))
    images_flow = my_gen.flow(Img, batch_size=1)

    for i, new_images in enumerate(images_flow):
        new_image = array_to_img(new_images[0], scale=True)
        ax[i].imshow(new_image, cmap="gray")

        if i >= count:
            break

    # ------------------- Task 4 ------------------- #

    train_dir = '/DL_course_data/lab2/Skin/train/'
    val_dir = '/DL_course_data/lab2/Skin/validation/'

    train_generator, val_generator = datagenerator(train_dir, val_dir)

    # --- Parameters --- #
    img_w = 256  # Witdh of input images
    img_h = 256  # Height of input images
    img_ch = 1  # Number of channels
    base = 64  # Number of neurons in first layer
    learning_rate = 0.00001  # Learning rate
    n_ep = 80  # Number of epochs
    dropout = 1  # 0 false, 1 true
    batch_norm = 1
    spat_dropout = 0  # 0 false, 1 true

    alexnet4 = alexnet(img_ch, img_w, img_h, base, dropout, batch_norm, spat_dropout)
    train_with_generator(alexnet4, learning_rate, train_generator, n_ep, val_generator)

    # ------------------- Task 5 ------------------- #

    vgg16_1 = vgg16(img_ch, img_w, img_h, base, batch_norm)
    train_with_generator(vgg16_1, learning_rate, train_generator, n_ep, val_generator)

    train_dir = '/DL_course_data/lab2/Bone/train/'
    val_dir = '/DL_course_data/lab2/Bone/validation/'

    train_generator, val_generator = datagenerator(train_dir, val_dir)

    vgg16_2 = vgg16(img_ch, img_w, img_h, base, batch_norm)
    train_with_generator(vgg16_2, learning_rate, train_generator, n_ep, val_generator)
