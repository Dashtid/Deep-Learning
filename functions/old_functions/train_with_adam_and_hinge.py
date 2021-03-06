import numpy as np
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from matplotlib import pyplot as plt


def train_with_adam_and_hinge(network, learning_rate, x_train, y_train, bs, n_ep,x_test, y_test):
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