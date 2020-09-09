# Import
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.optimizers import SGD, Adam

from functions.old_functions.MLP import MLP
from functions.old_functions.get_train_test_arrays import get_train_test_arrays
from functions.old_functions.mlp_keras import mlp_keras
from functions.old_functions.lenet import lenet

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def training(network, SSD):
    global i, errors
    for i in range(iterations):
        network.feedforward()
        network.backpropagation()
        errors = (Target - network.output) ** 2
        SSD.append(np.sum(errors))  # Objective(loss) function


def plotting(network, SSD):
    Itr = np.linspace(1, len(SSD), len(SSD))
    plt.plot(Itr, SSD)
    plt.xlabel('Iterations')
    plt.ylabel('SSD')
    print("The target values are:", Target)
    print("The predicted values are:", network.output)


if __name__ == "__main__":
    # Task 1

    # ---- Parameters ---- #

    iterations = 2000
    n_unit = 10  # Number of neurons in the Network

    # Please do notice how a third parameter is added to input, which represents the bias
    Input = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]])

    Target = np.array([[0], [0], [0], [1]])  # Target labels
    task1_MLP = MLP(Input, Target, n_unit)  # Creating the network with parameters
    SSD1 = []  # List to fill error values

    training(task1_MLP, SSD1)
    plotting(task1_MLP, SSD1)

    # Task 2

    # ---- Parameters ---- #

    iterations = 2000
    n_unit = 10  # Number of neurons in the Network

    # Different labels than task 1 since this represents the XOR-problem

    Input = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])

    Target = np.array([[0], [1], [1], [0]])  # Target labels
    task2_MLP = MLP(Input, Target, n_unit)  # Creating the network with parameters
    SSD2 = []  # List to fill error values

    # ---- Training sequence ---- #
    training(task2_MLP, SSD2)

    # ---- Plotting ---- #
    plotting(task2_MLP, SSD2)

    # Task 3

    # MPL with Tensorflow Keras

    # ---- Parameters ---- #
    Input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")  # Input data, notice how this differs from Task 1 & 2
    Target = np.array([[0], [1], [1], [0]], "float32")  # Target labels, we're still doing the XOR-problem
    n_unit = 10  # Number of Neurons

    # ---- Generating data for Task 4 & 5 ---- #

    # Loading data

    img_w, img_h = 128, 128  # Setting the width and heights of the images.
    data_path = '/DL_course_data/Lab1/Skin/'  # Path to data root with two subdirs.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path,
        train_list, test_list, img_h, img_w)

    # Constructor for a 4 layer MLP using TF Keras Model class

    # Task 4

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base_dense = 256  # Number of neurons in first layer

    # Creating a model with above parameters
    clf = mlp_keras(img_w, img_h, img_ch, base_dense)

    # Compiling the layers and setting loss & optimizer functions
    clf.compile(loss='binary_crossentropy',
                optimizer=SGD(lr=0.0001),
                metrics=['binary_accuracy'])

    # Trains the model for 150 epochs with 15 images in each batch
    clf_hist = clf.fit(x_train, y_train,
                       batch_size=16, epochs=150,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled

    # ---- Plotting ---- #
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(clf_hist.history["val_loss"]),
             np.min(clf_hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()

    # Task 5

    # ---- Parameters ---- #
    img_w = 128  # Witdh of input images
    img_h = 128  # Height of input images
    img_ch = 1  # Number of channels
    base = 32  # Number of neurons in first layer

    # Creating a CNN Network
    clf = lenet(img_ch, img_w, img_h, base)

    # Compiling the layers and setting loss & optimizer function
    clf.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=0.0001),
                metrics=['binary_accuracy'])

    # Trains the model for 200 epochs with 8 images in each batch
    clf_hist = clf.fit(x_train, y_train,
                       batch_size=8, epochs=200,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled

    # Prints a summary of the network
    clf.summary()

    # ---- Plotting ---- #
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(clf_hist.history["val_loss"]),
             np.min(clf_hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
