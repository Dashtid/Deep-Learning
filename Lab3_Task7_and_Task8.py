import os
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure

# disable_eager_execution()

from functions.networks2 import alexnet, vgg16, vgg16_2
from functions.training_tools2 import train_with_generator, plotting
from functions.dataloader import datagenerator, show_paired

# --- Task 7 & 8 --- #

if __name__ == "__main__":

    # --------------------------------------Task 7 -------------------------------------- #

    # Setting paths
    train_dir = '/DL_course_data/Lab2/Bone/train/'
    val_dir = '/DL_course_data/Lab2/Bone/validation/'

    # Creating generators
    train_generator, val_generator = datagenerator(train_dir, val_dir, target_size=(128, 128), class_mode='categorical')

    # Parameters
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 8  # Number of feature maps in first layer
    learning_rate = 0.000001  # Learning rate
    bs = 8  # Batch size
    n_ep = 80  # Number of epochs
    categorical = 1  # On/Off switch; 1 => categorical crossentropy, 0 => binary crossentropy

    vgg_task7 = vgg16_2(img_ch, img_w, img_h, base)
    model = train_with_generator(vgg_task7, learning_rate, train_generator, n_ep, val_generator, categorical)

    # --------------------------------------Task 8 -------------------------------------- #

    # Setting parameters
    img_h = 128  # Image height
    img_w = 128  # Image width
    base = 8

    # Solving tensorflow issue
    disable_eager_execution()

    sample_dir = '/DL_course_data/Lab2/Bone/train/AFF/14.jpg'

    img = imread(sample_dir)
    Img = img[:, :, 0]
    img = img / 255
    img = resize(Img, (img_h, img_w), anti_aliasing=True).astype('float32')
    print(np.shape(img))

    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    print(predictions)

    class_idx = np.argmax(predictions[0])
    print('the predicted class label is {}'.format(class_idx))
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("Last_ConvLayer")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(base * 8):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img_ = cv2.imread(sample_dir)
    img_ = cv2.resize(img_, (512, 512), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, (img_.shape[1], img_.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_, 0.6, heatmap, 0.4, 0)
    plt.figure()
    plt.imshow(img_)
    plt.figure()
    plt.imshow(superimposed_img)
