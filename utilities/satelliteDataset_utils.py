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
    scale : Default true, scales between 0-255

    __getitem__ : returns tile(image) and Path of tile file


    """

    def __init__ (self, dataPath, dataCSV=None, scale = True, imgExt=".npy"):
        if not os.path.exists(dataPath): sys.exit("Enter a Valid READ_PATH")
        self.imgExt = imgExt
        self.scale = scale
        if not dataCSV:
            self.filePathList = glob.glob( dataPath+"/**/*"+imgExt ,recursive=True)
            self.filePathList = sorted(self.filePathList)
        else :
            with open(dataCSV, "r") as cf:
                reader = csv.reader(cf)
                self.filePathList = [ os.path.join(dataPath,r) for r in reader]

    def __getitem__(self, idx):
        img = np.load(self.filePathList[idx])
        img = self.clip_scale_bands(img)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return Variable(img), self.filePathList[idx]

    def __len__(self):
        return len(self.filePathList)


    def clip_scale_bands(self, bands, normalize=True, clip_min=0, clip_max=255):
        bands = np.clip(bands, clip_min, clip_max)
        if normalize: bands = bands / (clip_max - clip_min)
        return bands


class ClubedTileData(Dataset):
    """ Loads all tiles in npy format in folder recursively
    All data are normalized 0-1 by division by 255
    """
    def __init__ (self, dataPath, dataCSV=None, img_ext=".npy"):
        for path in dataPath:
            if not os.path.exists(path): sys.exit("Enter a Valid READ_PATH")

        self.img_ext = img_ext
        self.club_sz = len(dataPath)
        self.filePathList = []
        if not dataCSV:
            for fth in range(self.club_sz):
                tempList = glob.glob( dataPath[fth]+"/**/*"+img_ext ,recursive=True)
                self.filePathList.append(sorted(tempList))
                print("Loaded files from:", dataPath[fth])
        else :
            sys.exit("Clubed data loading Not implemented for CSV")

        for path in self.filePathList:
            if not ( len(self.filePathList[0]) == len(path) ):
                sys.exit("MisMatch in the Loaded file counts of Clubbing")

    def __getitem__(self, idx):
        img = []
        for fth in range(self.club_sz):
            tempImg = np.load(self.filePathList[fth][idx])
            tempImg = self.clip_scale_bands(tempImg)
            tempImg = torch.from_numpy(tempImg).type(torch.FloatTensor)
            img.append(Variable(tempImg))
        return img

    def __len__(self):
        return len(self.filePathList[0])

    def clip_scale_bands(self, bands, normalize=True, clip_min=0, clip_max=255):
        bands = np.clip(bands, clip_min, clip_max)
        if normalize: bands = bands / (clip_max - clip_min)
        return bands


class tileNembedData(Dataset):
    """ Loads all tiles & Embeds in npy format in folder recursively
    tiles are normalized 0-1 by division by 255
    Embeds are NOT normalized
    """
    def __init__ (self, dataPath, dataCSV=None, img_ext=".npy"):
        ''' dataPath must be a list of following data
        dataPath[0] -> path to tiles
        dataPath[1] -> path to Embeds
        '''
        for path in dataPath:
            if not os.path.exists(path): sys.exit("Enter a Valid READ_PATH")

        self.img_ext = img_ext
        self.club_sz = len(dataPath)
        self.filePathList = []
        if not dataCSV:
            for fth in range(self.club_sz):
                tempList = glob.glob( dataPath[fth]+"/**/*"+img_ext ,recursive=True)
                self.filePathList.append(sorted(tempList))
                print("Loaded files from:", dataPath[fth])
        else :
            sys.exit("Clubed data loading Not implemented for CSV")

        for path in self.filePathList:
            if not ( len(self.filePathList[0]) == len(path) ):
                sys.exit("MisMatch in the Loaded file counts of Clubbing")

    def __getitem__(self, idx):
        # Tile
        tempImg = np.load(self.filePathList[0][idx])
        tempImg = self.clip_scale_bands(tempImg)
        tempImg = torch.from_numpy(tempImg).type(torch.FloatTensor)
        img = Variable(tempImg)
        # Embed - no scaling
        tempImg = np.load(self.filePathList[1][idx])
        tempImg = torch.from_numpy(tempImg).type(torch.FloatTensor)
        emb = Variable(tempImg)

        return img, emb, self.filePathList[0][idx]

    def __len__(self):
        return len(self.filePathList[0])

    def clip_scale_bands(self, bands, normalize=True, clip_min=0, clip_max=255):
        bands = np.clip(bands, clip_min, clip_max)
        if normalize: bands = bands / (clip_max - clip_min)
        return bands



class SimpleNumpyData(Dataset):
    """ Loads all tiles in npy format in folder recursively
    scale : Default true, scales between 0-255

    __getitem__ : Returns raw Numpy and Path

    """

    def __init__ (self, dataPath, dataCSV=None):
        if not os.path.exists(dataPath): sys.exit("Enter a Valid READ_PATH")

        if not dataCSV:
            self.filePathList = glob.glob( dataPath+"/**/*"+imgExt ,recursive=True)
            self.filePathList = sorted(self.filePathList)
        else :
            with open(dataCSV, "r") as cf:
                reader = csv.reader(cf)
                self.filePathList = [ os.path.join(dataPath,r) for r in reader]

    def __getitem__(self, idx):
        img = np.load(self.filePathList[idx])
        return img, self.filePathList[idx]

    def __len__(self):
        return len(self.filePathList)
