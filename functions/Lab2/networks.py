from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout


def vgg16(img_ch, img_width, img_height, n_base):
    model = Sequential()

    # base
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*2
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*4
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense 64
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def vgg16_with_dropout(img_ch, img_width, img_height, n_base):
    model = Sequential()

    # base
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*2
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*4
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # base*8
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=n_base * 8, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense 64
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def alexnet(img_ch, img_width, img_height, n_base):
    model = Sequential()

    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def alexnet_with_dropout(img_ch, img_width, img_height, n_base):
    model = Sequential()

    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def alexnet_with_dropout_and_softmax_multi(img_ch, img_width, img_height, n_base):
    model = Sequential()

    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=n_base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(9))
    model.add(Activation('softmax'))

    model.summary()
    return model


def lenet(img_ch, img_width, img_height, base):
    # Constructor for a 4 layer CNN using TF Keras Sequential class

    # Creating the model
    model = Sequential()

    # --- Creating the first two convolutional layers --- #

    # First convolutional layer
    model.add(Conv2D(base, input_shape=(img_width, img_height, img_ch), kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(base * 2, kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Creating the last two fully connected layers
    model.add(Flatten())
    model.add(Dense(base * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def lenet_with_softmax_multi(img_ch, img_width, img_height, base):
    # Constructor for a 4 layer CNN using TF Keras Sequential class

    # Creating the model
    model = Sequential()

    # --- Creating the first two convolutional layers --- #

    # First convolutional layer
    model.add(Conv2D(base, input_shape=(img_width, img_height, img_ch), kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(base * 2, kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Creating the last two fully connected layers
    model.add(Flatten())
    model.add(Dense(base * 2, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    return model
