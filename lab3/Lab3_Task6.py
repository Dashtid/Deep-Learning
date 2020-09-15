import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from functions.Lab3.networks import MLP
from functions.dataloader import get_length

# --- Task 6 --- #

if __name__ == "__main__":

    # Pathways to directories
    train_data_dir = '/DL_course_data/Lab2/Bone/train/'
    validation_data_dir = '/DL_course_data/Lab2/Bone/validation/'

    # Parameters for training
    img_width = 224  # Image width
    img_height = 224  # Image height
    img_ch = 1  # Number of channels used
    epochs = 150  # Number of epochs
    batch_size = 8  # Number of images in each batch
    LR = 0.00001  # Learning rate of training

    # Number of data for each class
    Len_C1_Train = get_length(train_data_dir, 'AFF')
    Len_C2_Train = get_length(train_data_dir, 'NFF')
    Len_C1_Val = get_length(validation_data_dir, 'AFF')
    Len_C2_Val = get_length(validation_data_dir, 'NFF')

    # Loading the pre-trained model
    # Include_top=False means that the dense layers at the top of the network will not be used
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.summary()

    # Feature extraction from pretrained VGG (training data)
    datagenerator = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagenerator.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # Extracting the features from the loaded images
    features_train = model.predict_generator(
        train_generator,
        (Len_C1_Train + Len_C2_Train) // batch_size, max_queue_size=1)

    # Feature extraction from pretrained VGG (validation data)
    datagenerator = ImageDataGenerator(rescale=1. / 255)

    validation_generator = datagenerator.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # Extracting the features from the loaded images
    features_validation = model.predict_generator(
        validation_generator,
        (Len_C1_Val + Len_C2_Val) // batch_size, max_queue_size=1)

    # Training a small MLP with extracted features from the pre-trained model
    # In fact this MLP will be used instead of the dense layers of the VGG model
    # and only this MLP will be trained on the dataset.
    train_data = features_train
    train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

    validation_data = features_validation
    validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))

    mlp = MLP(1112, 7, 7)

    mlp.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=LR),
                metrics=['binary_accuracy'])

    # Trains the model for 150 epochs with 8 images in each batch
    mlp_hist = mlp.fit(train_data, train_labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(validation_data, validation_labels))

    # ---- Plotting ---- #
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(mlp_hist.history["loss"], label="loss")
    plt.plot(mlp_hist.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(mlp_hist.history["val_loss"]),
             np.min(mlp_hist.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
