import os
import re
import nibabel as nib
import numpy as np
from random import shuffle


def shuffle_n_split_data(img_list, msk_list, frac):
    comb = list(zip(img_list, msk_list))
    shuffle(comb)
    img_list[:], msk_list[:] = zip(*comb)

    length_split = int(frac * len(img_list))

    train_img = img_list[:length_split]
    train_msk = msk_list[:length_split]
    val_img = img_list[length_split:]
    val_msk = msk_list[length_split:]
    return train_img, train_msk, val_img, val_msk


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def load_streamlines(datapath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = datapath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines
            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total), n_tracts_per_bundle, replace=False)

            streamlines_data = streamlines.data
            streamlines_offsets = streamlines._offsets

            for j in range(n_tracts_per_bundle):
                ix_j = ix_tracts[j]
                offset_start = streamlines_offsets[ix_j]
                if ix_j < (n_tracts_total - 1):
                    offset_end = streamlines_offsets[ix_j + 1]
                    streamline_j = streamlines_data[offset_start:offset_end]
                else:
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j))
                y.append(c)
    return X, y


def get_file_list(data_path):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            mypath = os.path.join(root, file)
            img_list.append(mypath)
    img_list.sort(key=natural_sort_key)
    return img_list
