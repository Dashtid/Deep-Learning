import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from matplotlib import pyplot as plt


def train_with_adam(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(hist.history["val_loss"]),
             np.min(hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()


def train_with_adam_and_accuracyplot(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['sparse_categorical_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled

    # Loss Curve
    plt.figure(1, figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(hist.history["val_loss"]),
             np.min(hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()

    # Accuracy Curve
    plt.figure(2, figsize=(4, 4))
    plt.title("Accuracy curve")
    plt.plot(hist.history["sparse_categorical_accuracy"], label="Accuracy")
    plt.plot(hist.history["val_sparse_categorical_accuracy"], label="Val_accuracy")
    plt.plot(np.argmax(hist.history["val_sparse_categorical_accuracy"]),
             np.max(hist.history["val_sparse_categorical_accuracy"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Fraction")
    plt.legend()
    plt.show()


def train_with_adam_and_hinge(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='hinge',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(hist.history["val_loss"]),
             np.min(hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()


def train_with_rmsprop(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(lr=learning_rate),
                    metrics=['binary_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(hist.history["val_loss"]),
             np.min(hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()


def train_with_sgd(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='binary_crossentropy',
                    optimizer=SGD(lr=learning_rate),
                    metrics=['binary_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))  # Validation data is training data shuffled
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(hist.history["val_loss"]),
             np.min(hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
