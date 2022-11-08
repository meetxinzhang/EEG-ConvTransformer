# encoding: utf-8
"""
Forked from https://github.com/numediart/EEGLearn-Pytorch/blob/master/Utils.py
"""

import scipy.io as sio
from preprocessing.aep import azim_proj, gen_images
import numpy as np


def create_img():
    feats = sio.loadmat('Sample Data/FeatureMat_timeWin.mat')['features']
    locs = sio.loadmat('Sample Data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    images_timewin = np.array([gen_images(np.array(locs_2d),
                                          feats[:, i * 192:(i + 1) * 192], 32, normalize=True) for i in
                               range(int(feats.shape[1] / 192))
                               ])

    sio.savemat("Sample Data/images_time.mat", {"img": images_timewin})
    print("Images Created and Save in Sample Dat/images_time")
