import numpy as np


# Loading data
def gen_labels(im_name, label_list):
    #   '''
    # Parameters
    # ----------
    # im_name : Str
    # The image file name.
    # Returns
    # -------
    # Label : Numpy array
    # Class label of the filename name based on its pattern.
    # '''
    i = 0
    for lbl in label_list:
        if lbl in im_name:
            label = np.array([i])
            break
        i += 1
    return label
