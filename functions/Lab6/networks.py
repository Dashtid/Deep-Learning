import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, ConvLSTM2D, Dense, MaxPooling2D, Conv2D, Activation, \
    Dropout, BatchNormalization, Conv2DTranspose, concatenate, LSTM, Bidirectional


def plot_history(net_history):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(net_history.history["loss"], label="loss")
    plt.plot(net_history.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(net_history.history["val_loss"]),
             np.min(net_history.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()


def conv_block(base, layer, batch_norm):
    layer_conv = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer)
    if batch_norm:
        layer_bn = BatchNormalization()(layer_conv)
        layer_act = Activation('relu')(layer_bn)
    else:
        layer_act = Activation('relu')(layer_conv)

    layer_conv2 = Conv2D(filters=base, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_act)
    if batch_norm:
        layer_bn2 = BatchNormalization()(layer_conv2)
        layer_act2 = Activation('relu')(layer_bn2)
    else:
        layer_act2 = Activation('relu')(layer_conv2)

    return layer_act2


def deconv_block(base, conc_layer, layer, batch_norm, dropout, img_size):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer)

    x1 = Reshape(target_shape=(1, np.int32(img_size), np.int32(img_size), base))(conc_layer)
    x2 = Reshape(target_shape=(1, np.int32(img_size), np.int32(img_size), base))(layer_convT)

    layer_conc = concatenate([x1, x2], axis=1)
    if dropout:
        layer_lstm = ConvLSTM2D(np.int32(base / 2), (3, 3), padding='same', return_sequences=False, go_backwards=True)(
            layer_conc)
        layer_d = Dropout(0.2)(layer_lstm)
        layer_b = conv_block(base, layer_d, batch_norm)
    else:
        layer_lstm = ConvLSTM2D(np.int32(base / 2), (3, 3), padding='same', return_sequences=False, go_backwards=True)(
            layer_conc)
        layer_b = conv_block(base, layer_lstm, batch_norm)

    return layer_b


def get_unet(base, img_w, img_h, img_ch, batch_norm, dropout):
    # Defining the Input layer
    layer_inp = Input(shape=(img_h, img_w, img_ch))

    # --- Contraction Phase --- #
    layer_b1 = conv_block(base, layer_inp, batch_norm)
    layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)
    if dropout:
        layer_d1 = Dropout(0.2)(layer_mp1)
        layer_b2 = conv_block(base * 2, layer_d1, batch_norm)
    else:
        layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)

    layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    if dropout:
        layer_d2 = Dropout(0.2)(layer_mp2)
        layer_b3 = conv_block(base * 4, layer_d2, batch_norm)
    else:
        layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)

    layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    if dropout:
        layer_d3 = Dropout(0.2)(layer_mp3)
        layer_b4 = conv_block(base * 8, layer_d3, batch_norm)
    else:
        layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)
    layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)

    # --- Bottle-neck Phase --- #
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    # --- Expansion Phase --- #
    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout, img_h / 8)
    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout, img_h / 4)
    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout, img_h / 2)
    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout, img_h)

    # --- Output layer --- #
    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    # --- Creating the model --- #
    model = Model(inputs=layer_inp, outputs=layer_out)

    model.summary()

    return model


def reg_model(n_units, input_layer, bd):
    if bd:
        l1 = Bidirectional(LSTM(n_units,
                                return_sequences=True,
                                stateful=True),
                           merge_mode='concat')(input_layer)
    else:
        l1 = LSTM(n_units, return_sequences=True, stateful=True)(input_layer)
    l2 = Dropout(0.2)(l1)

    l3 = LSTM(n_units, return_sequences=True, stateful=True)(l2)
    l4 = Dropout(0.2)(l3)

    l5 = LSTM(n_units, return_sequences=True, stateful=True)(l4)
    l6 = Dropout(0.2)(l5)

    l7 = LSTM(n_units, stateful=True)(l6)
    l8 = Dropout(0.2)(l7)

    output_layer = Dense(1)(l8)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    return model
