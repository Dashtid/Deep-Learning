from lab1.Lab1 import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D


def mlp_keras(img_width, img_height, img_ch, base_dense):
    # Defining input
    input_size = (img_width, img_height, img_ch)
    inputs_layer = Input(shape=input_size, name='input_layer')

    # Creating layers and connecting them. Each layer is used as input to the next.
    flatten_layer = Flatten()(inputs_layer)
    layer1 = Dense(base_dense, activation='relu')(flatten_layer)
    layer2 = Dense(base_dense // 2, activation='relu')(layer1)
    layer3 = Dense(base_dense // 4, activation='relu')(layer2)

    # Output layer has only one node and sigmoid activation to get a binary classification
    out = Dense(1, activation='sigmoid')(layer3)

    # Creating a model with above parameters using Model in TF Keras
    clf = Model(inputs=inputs_layer, outputs=out)

    # Prints a summary of then network
    clf.summary()

    return clf
