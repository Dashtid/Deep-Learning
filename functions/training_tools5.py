import cv2
import numpy as np
from random import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_gen(img_train, label_train, n_train_sample, batch_size, image_size):
    """
    img_train: a list containing full directory of training images
    label_train: a list containing full directory of training masks
    n_train_sample: len(img_train)
    batch_size: integer value of batch size
    image_size: an integer value e.g, 240
    """
    data_gen_args = dict(rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         cval=0,
                         zoom_range=0.1,
                         horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    label_datagen = ImageDataGenerator(**data_gen_args)
    while True:
        samples = list(zip(img_train, label_train))
        shuffle(samples)
        sample_img, sample_label = zip(*samples)
        sample_img = list(sample_img)
        sample_label = list(sample_label)
        for ind in (range(0, n_train_sample, batch_size)):
            batch_img = sample_img[ind:ind + batch_size]
            batch_label = sample_label[ind:ind + batch_size]
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            length = len(batch_img)
            if length == batch_size:
                pass
            else:
                for tmp in range(batch_size - length):
                    batch_img.append(batch_img[-1])
                    batch_label.append(batch_label[-1])
            x_train = np.empty([batch_size, image_size, image_size], dtype='float32')
            y_train = np.empty([batch_size, image_size, image_size], dtype='float32')

            for ix in range(len(batch_img)):
                img_sample = batch_img[ix]
                label_sample = batch_label[ix]
                img_array = load_img_array(img_sample, image_size)
                label_array = load_img_array(label_sample, image_size)
                x_train[ix] = img_array
                y_train[ix] = label_array
            x_train = np.expand_dims(x_train, axis=3)
            y_train = np.expand_dims(y_train, axis=3)
            image_generator = image_datagen.flow(x_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            label_generator = label_datagen.flow(y_train, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            img_gen = image_generator.next()
            label_gen = label_generator.next()
            yield img_gen, label_gen


def load_img_array(img_dir, image_size):
    img_arr = cv2.imread(img_dir, 0)
    img_arr = cv2.resize(img_arr[:, :], (image_size, image_size))
    img_arr = img_arr / 255.
    return img_arr


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


class MyBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=1, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Get number of batches per epoch'
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffle indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, 1))
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb
