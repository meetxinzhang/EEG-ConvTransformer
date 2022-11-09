# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/9 15:22
 @name: 
 @desc:
"""
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, label, image):
        self.label = label
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)
        return sample