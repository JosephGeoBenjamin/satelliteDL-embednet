import glob
import sys
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import imageio as imio
import csv


class PlainTileData(Dataset):
    """ Loads all tiles in npy format in folder recursively
    """
    self.join = os.path.join

    def __init__ (self, dataPath, imgExt=".npy"):
        if not os.path.exists(dataPath): sys.exit("Enter a Valid READ_PATH")
        self.imgExt = imgExt
        self.filePath = glob.glob( READ_PATH+"/**/*"+imgExt ,recursive=True)
        self.filePath = sorted(self.filePath)

    def __getitem__(self, idx):
        img = np.load(self.filePath[idx])
        img = np.expand_dims(img, axis=0)
        img = clip_scale_bands(img)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return Variable(img), self.filePath[idx]

    def __len__(self):
        return len(self.filePath)


    def clip_scale_bands(self, bands, normalize=True, clip_min=0, clip_max=255):
        bands = np.clip(bands, clip_min, clip_max)
        if normalize: bands = bands / (clip_max - clip_min)
        return bands
