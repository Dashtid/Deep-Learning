import os

from networks import alexnet, alexnet_with_dropout, lenet, vgg16, vgg16_with_dropout
from training_tools import train_with_adam
from functions.dataloader import get_train_test_arrays

# --- Task 7 --- #

if __name__ == "__main__":

    # ------------------- Task 7A ------------------- #
    # See the function vgg16 in functions.networks

    # ------------------- Task 7B ------------------- #
    # Loading in training data
    skin_labels_string_list = ['Mel', 'Nev']
    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path,
        train_list, test_list, img_h, img_w, skin_labels_string_list)

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 32  # Number of neurons in first layer
    learning_rate = 0.00001  # Learning rate
    bs = 8  # batch size
    n_ep = 100  # Number of epochs

    vgg16_7b = vgg16(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_7b, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 7C ------------------- #
    learning_rate = 1e-5
    n_ep = 150

    # --- VGG16 --- #
    vgg16_7c = vgg16(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_7c, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- AlexNet --- #
    alex7c = alexnet(img_ch, img_w, img_h, base)
    train_with_adam(alex7c, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 7D ------------------- #
    n_ep = 100
    base = 16

    # --- VGG16 --- #
    vgg16_7d1 = vgg16(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_7d1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- VGG16 w dropout --- #
    vgg16_7d2 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_7d2, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # ------------------- Task 7E ------------------- #

    # --- LeNet --- #
    lenet_1 = lenet(img_ch, img_w, img_h, base)
    train_with_adam(lenet_1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- AlexNet --- #
    alexnet_1 = alexnet_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(alexnet_1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)

    # --- VGG16 --- #
    vgg16_1 = vgg16_with_dropout(img_ch, img_w, img_h, base)
    train_with_adam(vgg16_1, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test)