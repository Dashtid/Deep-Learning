import os

from functions.Lab2.networks import vgg16_with_dropout
from functions.Lab2.training_tools import train_with_adam
from functions.dataloader import get_train_test_arrays

# --- Task 8 & 9  --- #

if __name__ == "__main__":
    # Bone Images
    bone_labels_string_list = ['AFF', 'NFF']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/lab1/Bone/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w, bone_labels_string_list)

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 8  # Number of neurons in first layer
    learning_rate = 0.000004  # Learning rate
    bs = 8  # batch size
    n_ep = 100  # Number of epochs

    # Training network w/ bs = 8
    vgg16_network1_8 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_network1_8, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # Training w/ bs = 16
    base = 16
    vgg16_network1_16 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_network1_16, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # Skin Images
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    skin_labels_string_list = ['Mel', 'Nev']
    data_path = '/DL_course_data/lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path,
        train_list, test_list, img_h, img_w,skin_labels_string_list)

    # Training network w/ bs = 8
    base = 8
    vgg16_network2_8 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_network2_8, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # Training w/ bs = 16
    base = 16
    vgg16_network2_16 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_network2_16, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)
