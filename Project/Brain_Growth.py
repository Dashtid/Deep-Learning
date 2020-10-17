import os
import re
import copy
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add, AveragePooling2D, Reshape, ConvLSTM2D, Dense, Flatten, ZeroPadding2D, \
    MaxPooling2D, Conv2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM, SpatialDropout2D

from Project.Functions.Dataloader import get_file_list, load_img_array
from Project.Functions.Networks import get_unet, get_unet_deep
from Project.Functions.Training_tools import dice_coef, dice_loss, precision, recall, plot_history, data_generator, \
    trainer, calc_n_avarage_dice

# -------------------------------------------- Brain - Growth -------------------------------------------- #

if __name__ == "__main__":
    # Setting paths to directories
    train_dir = 'Project_data/training_data_v2/brain-growth'
    val_dir = 'Project_data/validation_data_v2/brain-growth'
    test_dir = 'Project_data/test/brain-growth'

    # -------------------------------------------- Regular U - Net -------------------------------------------- #

    # ----- Parameters ----- #
    base = 32  # Number of feature maps
    img_size = 256  # Size of input
    img_ch = 1  # Dimensions of input
    bs = 1  # Batch size
    lr = 0.0001  # Learning rate
    batch_norm = 1  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    n_ep = 200  # Number of epochs
    dr_rate = 0.5  # Drop-rate
    Metric = [dice_coef, precision, recall]  # Evaluation metrics used

    # 7 networks, 5 test images
    n_masks = 7
    preds = np.zeros((7, 5, img_size, img_size, img_ch))

    for i in range(n_masks):
        K.clear_session()  # Clearing weights of previous network to save memory

        # Combining paths with file names and fetching a list with images corresponding to the right expert
        train_img_path, train_msk_path = get_file_list(train_dir, 'seg0' + str(i + 1))
        val_img_path, val_msk_path = get_file_list(val_dir, 'seg0' + str(i + 1))
        test_img_path, test_msk_path = get_file_list(test_dir, 'seg0' + str(i + 1))

        # Loading in training and validation images as an array
        x_train = load_img_array(train_img_path, 256)
        y_train = load_img_array(train_msk_path, 256)
        x_val = load_img_array(val_img_path, 256)
        y_val = load_img_array(val_msk_path, 256)
        x_test = load_img_array(test_img_path, 256)

        print(test_img_path)
        print(np.shape(x_test))

        # Creating the two generators needed
        train_gen = data_generator(x_train, y_train, bs)
        val_gen = data_generator(x_val, y_val, bs)

        # Creating the first architecture
        network1 = get_unet(base, img_size, img_ch, batch_norm, dropout, dr_rate)

        # Training this network architecture
        hist1, trained_network1 = trainer(network1, n_ep, lr, bs, x_train, x_val, train_gen, val_gen)

        # Plotting the resulting histogram
        plot_history(hist1)

        # Filling a list with our predicted mask of the test data
        preds[i] = trained_network1.predict(x_test)  # One mask for each expert

    # Function that calculates the dice scores of our predicted masks and does an average on these scores
    x = calc_n_avarage_dice(preds, n_masks, test_dir)
    print(x)

    ## ----- Parameters ----- #
    # base = 16  # Number of feature maps
    # img_size = 256  # Size of input
    # img_ch = 1  # Dimensions of input
    # bs = 2  # Batch size
    # lr = 0.0001  # Learning rate
    # batch_norm = 1  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    # dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    # n_ep = 500  # Number of epochs
    # dr_rate = 0.5
    # Metric = [dice_coef, precision, recall]  # Evaluation metrics used
#
## 7 networks, 5 test images
# preds = np.zeros((7, 5, img_size, img_size, img_ch))
#
# for i in range(7):
#    K.clear_session()
#
#    # Combining paths with file names and fetching a list with those
#    train_img_path, train_msk_path = get_file_list(train_dir, 'seg0' + str(i + 1))
#    val_img_path, val_msk_path = get_file_list(val_dir, 'seg0' + str(i + 1))
#    test_img_path, test_msk_path = get_file_list(test_dir, 'seg0' + str(i + 1))
#
#    # Loading in training and validation images as an array
#    x_train = load_img_array(train_img_path, 256)
#    y_train = load_img_array(train_msk_path, 256)
#    x_val = load_img_array(val_img_path, 256)
#    y_val = load_img_array(val_msk_path, 256)
#
#    x_test = load_img_array(test_img_path, 256)
#
#    train_gen = data_generator(x_train, y_train, bs)
#    val_gen = data_generator(x_val, y_val, bs)
#
#    model = get_unet_deep(base, img_size, img_ch, batch_norm, dropout, dr_rate)
#
#    model.compile(loss=dice_loss,
#                  optimizer=Adam(lr=lr),
#                  metrics=Metric)
#
#    model_hist = model.fit(train_gen,
#                           steps_per_epoch=len(x_train) // bs,
#                           validation_data=val_gen,
#                           validation_steps=len(x_val) // bs,
#                           epochs=n_ep)
#
#    preds[i] = model.predict(x_test)
#
#    print(np.all(preds[i] == 0))
#
#    plt.figure(figsize=(4, 4))
#    plt.title("Learning curve")
#    plt.plot(model_hist.history["loss"], label="loss")
#    plt.plot(model_hist.history["val_loss"], label="val_loss")
#    plt.plot(np.argmin(model_hist.history["val_loss"]),
#             np.min(model_hist.history["val_loss"]),
#             marker="x", color="r", label="best model")
#    plt.xlabel("Epochs")
#    plt.ylabel("Loss Value")
#    plt.legend();
#
