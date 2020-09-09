import numpy as np


def get_data_arrays(nested_list, img_h, img_w):
    # """
    # Parameters
    # ----------
    # nested_list : nested list
    # nested list of image arrays with corresponding class labels.
    # img_h : Int
    # Image height.
    # img_w : Int
    # Image width.
    # -------
    # img_arrays : Numpy array
    # 4D Array with the size of (n_data,img_h,img_w, 1)
    # label_arrays : Numpy array
    # 1D array with the size (n_data).
    # """

    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype=np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype=np.int32)

    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind] = nested_list[ind][1]

    img_arrays = np.expand_dims(img_arrays, axis=3)
    return img_arrays, label_arrays
