import os
from skimage.transform import resize
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from networks import lenet_with_softmax_multi, alexnet_with_dropout_and_softmax_multi
from training_tools import train_with_adam_and_accuracyplot
from functions.dataloader import get_train_test_arrays

# --- Task 10  --- #

if __name__ == "__main__":
    # Organ Images
    xray_labels_string_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/lab1/X_ray/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w, xray_labels_string_list)

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 32  # Number of neurons in first layer
    learning_rate = 0.000005  # Learning rate
    bs = 10  # batch size
    n_ep = 100  # Number of epochs

    # --- LeNet --- #
    lenet_1 = lenet_with_softmax_multi(img_ch, img_w, img_h, base)
    train_with_adam_and_accuracyplot(lenet_1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- AlexNet --- #
    alexnet_1 = alexnet_with_dropout_and_softmax_multi(img_ch, img_w, img_h, base)
    train_with_adam_and_accuracyplot(alexnet_1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)
