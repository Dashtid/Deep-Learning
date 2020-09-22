import re
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from functions.Lab4.dataloader import shuffle_n_split_data, load_img, get_file_list
from functions.Lab4.networks import get_unet
from functions.Lab4.training_tools import train_with_adam, train_with_generator, plotting

# --- Task 3 --- #

if __name__ == "__main__":
    # Setting parameters
    base = 8  # Number of feature maps in convolutional layer
    img_w = 256  # Image width
    img_h = 256  # Image height
    img_ch = 1  # Number of image channels
    bs = 8  # Batch size
    lr = 0.00001  # Learning rate
    batch_norm = 1  # On/Off switch for batch-normalization layer, 0 = False, 1 = True
    dropout = 1  # On/Off switch for dropout layer, 0 = False, 1 = True
    dice = 1  # On/Off switch for DICE-loss function, 0 = False, 1 = True
    n_ep = 150

    img_dir = '/DL_course_data/Lab3/CT/Image/'
    msk_dir = '/DL_course_data/Lab3/CT/Mask/'

    img_pathlist = get_file_list(img_dir)
    msk_pathlist = get_file_list(msk_dir)

    train_img, train_msk, val_img, val_msk = shuffle_n_split_data(img_pathlist[0:4000], msk_pathlist[0:4000], 0.8)

    # Loading in the actual images
    x_train = load_img(train_img, 256, 0)
    y_train = load_img(train_msk, 256, 1)
    x_val = load_img(val_img, 256, 0)
    y_val = load_img(val_msk, 256, 1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

    network_task3 = get_unet(base, img_w, img_h, img_ch, batch_norm, dropout)

    network_task3.train_with_generator(network_task3, lr,n_ep, metrics,categorical, traning_generator, vaL_data )


    def train_with_generator(network, learning_rate, n_ep, metrics, categorical, train_generator, val_data):