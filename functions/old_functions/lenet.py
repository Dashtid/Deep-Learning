from tensorflow.keras.models import Sequential, Model


def lenet(img_ch, img_width, img_height, base):
    # Constructor for a 4 layer CNN using TF Keras Sequential class

    # Creating the model
    model = Sequential()

    # --- Creating the first two convolutional layers --- #

    # First convolutional layer
    model.add(Conv2D(base, input_shape = (img_width, img_height, img_ch), kernel_size=(3, 3), activation='relu',
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



