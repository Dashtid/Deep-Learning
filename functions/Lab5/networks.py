from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dropout, \
    BatchNormalization, Conv2DTranspose, concatenate


# -----  U - Net ----- #

# ---- Creating the parts of the U - Net ---- #
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


def deconv_block(base, conc_layer, layer, batch_norm, dropout):
    layer_convT = Conv2DTranspose(filters=base, kernel_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer_conc = concatenate([conc_layer, layer_convT])
    if dropout:
        layer_sd = Dropout(0.2)(layer_conc)
        layer_b = conv_block(base, layer_sd, batch_norm)
    else:
        layer_b = conv_block(base, layer_conc, batch_norm)

    return layer_b


# ---- Assembling all the parts ---- #
def get_unet(base, img_w, img_h, img_ch, batch_norm, dropout, task3=1, **kwargs):
    # Defining the Input layer
    layer_inp = Input(shape=(img_h, img_w, img_ch))

    # --- Contraction Phase --- #
    layer_b1 = conv_block(base, layer_inp, batch_norm)
    if dropout:
        layer_sd1 = Dropout(0.2)(layer_b1)
        layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_sd1)
    else:
        layer_mp1 = MaxPooling2D(pool_size=(2, 2))(layer_b1)

    layer_b2 = conv_block(base * 2, layer_mp1, batch_norm)
    if dropout:
        layer_sd2 = Dropout(0.2)(layer_b2)
        layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_sd2)
    else:
        layer_mp2 = MaxPooling2D(pool_size=(2, 2))(layer_b2)

    layer_b3 = conv_block(base * 4, layer_mp2, batch_norm)
    if dropout:
        layer_sd3 = Dropout(0.2)(layer_b3)
        layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_sd3)
    else:
        layer_mp3 = MaxPooling2D(pool_size=(2, 2))(layer_b3)

    layer_b4 = conv_block(base * 8, layer_mp3, batch_norm)
    if dropout:
        layer_sd4 = Dropout(0.2)(layer_b4)
        layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_sd4)
    else:
        layer_mp4 = MaxPooling2D(pool_size=(2, 2))(layer_b4)

    # --- Bottle-neck Phase --- #
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    # --- Expansion Phase --- #
    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout)
    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout)
    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout)
    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout)

    # --- Output layer --- #
    layer_conv2 = Conv2D(filters=task3, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    # --- Creating the model --- #
    model = Model(inputs=layer_inp, outputs=layer_out)
    model.summary()
    return model


def get_unet_weight(base, img_w, img_h, img_ch, batch_norm, dropout, weight_strength, lr, metric):
    layer_inp = Input(shape=(img_h, img_w, img_ch))
    weight_maps = Input(shape=(img_w, img_h, img_ch), name='weights')
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
    # Bottle-neck
    layer_b5 = conv_block(base * 16, layer_mp4, batch_norm)

    #     if dropout:
    #         layer_b5=Dropout(0.2)(layer_b5)

    layer_db1 = deconv_block(base * 8, layer_b4, layer_b5, batch_norm, dropout)

    layer_db2 = deconv_block(base * 4, layer_b3, layer_db1, batch_norm, dropout)

    layer_db3 = deconv_block(base * 2, layer_b2, layer_db2, batch_norm, dropout)

    layer_db4 = deconv_block(base, layer_b1, layer_db3, batch_norm, dropout)

    layer_conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_db4)
    layer_out = Activation('sigmoid')(layer_conv2)

    model = Model(inputs=[layer_inp, weight_maps], outputs=layer_out)

    model.summary()
    model.compile(loss=weighted_loss(weight_maps, weight_strength),
                  optimizer=Adam(lr=lr),
                  metrics=metric)
    return model


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map)
        weight_f = weight_f * weight_strength
        weight_f = 1 / (weight_f + 1)
        weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
        return 1 - (2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    return weighted_dice_loss
