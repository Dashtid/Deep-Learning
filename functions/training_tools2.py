import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from matplotlib import pyplot as plt


def plotting(hist):
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


def train_with_adam(network, learning_rate, x_train, y_train, bs, n_ep, x_test, y_test):

    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])
    # Trains the model
    hist = network.fit(x_train, y_train,
                       batch_size=bs, epochs=n_ep,
                       validation_data=(x_test, y_test))
    plotting(hist)


def train_with_generator(network, learning_rate, train_generator, n_ep, val_generator, categorical):

    if categorical:
        # Compiling the layers and setting loss & optimizer function
        network.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr=learning_rate),
                        metrics=['categorical_accuracy'])

        # Trains the model
        hist = network.fit_generator(train_generator,
                                     validation_data=val_generator,
                                     epochs=n_ep)
        plotting(hist)

    # Compiling the layers and setting loss & optimizer function
    network.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])

    # Trains the model for 200 epochs with 8 images in each batch
    hist = network.fit_generator(train_generator,
                                 validation_data=val_generator,
                                 epochs=n_ep)
    plotting(hist)
    return network
