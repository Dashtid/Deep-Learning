from functions.old_functions.get_data import get_data
from functions.old_functions.get_data_arrays import get_data_arrays


def get_train_test_arrays(train_data_path, test_data_path, train_list,
                          test_list, img_h, img_w, label_string_list):
    # """
    # Get the directory to the train and test sets, the files names and
    # the size of the image and return the image and label arrays for
    # train and test sets.
    # """

    train_data = get_data(train_data_path, train_list, img_h, img_w, label_string_list)
    test_data = get_data(test_data_path, test_list, img_h, img_w, label_string_list)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del train_data
    del test_data
    return train_img, test_img, train_label, test_label
