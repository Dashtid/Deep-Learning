import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


def get_autocontext_fold(y_pred, f, n_folds, img_per_fold, img_size):
    autocontext_val = y_pred[(f * img_per_fold):((f + 1) * img_per_fold)]
    autocontext_train = np.zeros(((n_folds - 1) * img_per_fold, img_size, img_size, 1))
    if f != 0:
        autocontext_train[0:(f * img_per_fold)] = y_pred[0:(f * img_per_fold)]
    if f != (n_folds - 1):
        autocontext_train[(f * img_per_fold):] = y_pred[((f + 1) * img_per_fold):]
    return autocontext_train, autocontext_val


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


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_calculated = true_positives / (possible_positives + K.epsilon())
    return recall_calculated


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_calculated = true_positives / (predicted_positives + K.epsilon())
    return precision_calculated


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
