# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 19:34
 @name: 
 @desc:
"""

import scipy.io as sio
import numpy as np
import torch
from preprocessing.aep import azim_proj, gen_images
import einops

sample_rate = 62.5


def eeg_to_img(eeg, y, locs):
    # [samples=5184, time=32, channels=62]
    # [5184]
    # [62, 3]

    samples, time, channels = eeg.shape
    eeg = einops.rearrange(eeg, 'n t c -> (n t) c', n=samples, t=time, c=channels)

    locs_2d = [azim_proj(e) for e in locs]
    print(np.shape(locs_2d))

    imgs = gen_images(locs=np.array(locs_2d),  # [samples*time, colors, W, H]
                      features=eeg,
                      n_gridpoints=32,
                      normalize=True).squeeze()

    imgs = einops.rearrange(imgs, '(n t) w h -> n t w h', n=samples, t=time)


