import numpy as np
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


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


def train_with_adam(network, learning_rate, bs, n_ep, dice, x_train, y_train, x_test, y_test):
    # Compiling the layers and setting loss & optimizer function
    if dice:
        network.compile(loss=dice_loss,
                        optimizer=Adam(lr=learning_rate),
                        metrics=[dice_coef])
        # Trains the model
        hist = network.fit(x_train, y_train,
                           batch_size=bs, epochs=n_ep,
                           validation_data=(x_test, y_test))
        plotting(hist)

    else:
        network.compile(loss='binary_crossentropy',
                        optimizer=Adam(lr=learning_rate),
                        metrics=[dice_coef])
        # Trains the model
        hist = network.fit(x_train, y_train,
                           batch_size=bs, epochs=n_ep,
                           validation_data=(x_test, y_test))
        plotting(hist)
    return network


def train_with_generator(network, learning_rate, n_ep, metrics, categorical, train_generator, val_data):
    if metrics:
        Metric = [dice_coef, precision, recall]

        if categorical:
            # Compiling the layers and setting loss & optimizer function
            network.compile(loss='categorical_crossentropy',
                            optimizer=Adam(lr=learning_rate),
                            metrics=Metric)

            # Trains the model
            hist = network.fit_generator(train_generator,
                                         validation_data=val_data,
                                         epochs=n_ep)
            plotting(hist)

        else:
            # Compiling the layers and setting loss & optimizer function
            network.compile(loss=dice_loss,
                            optimizer=Adam(lr=learning_rate),
                            metrics=Metric)

            # Trains the model for 200 epochs with 8 images in each batch
            hist = network.fit_generator(train_generator,
                                         validation_data=val_data,
                                         epochs=n_ep)
            plotting(hist)

    else:
        # Compiling the layers and setting loss as dice & optimizer as Adam
        network.compile(loss=dice_loss,
                        optimizer=Adam(lr=learning_rate),
                        metrics=[dice_coef])

        # Trains the model for 200 epochs with 8 images in each batch
        hist = network.fit_generator(train_generator,
                                     validation_data=val_data,
                                     epochs=n_ep)

        plotting(hist)
    return network
