""" CDL Dataset creation (numpy format)

* Dumps tiles and Probability distribution of crops found in the tile

* Dataset: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL

"""

import numpy as np
from osgeo import gdal
import scipy.misc
import os
import sys
import csv

SAVE_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/cdl_data/"
READ_PATH = "/home/jupyter/satellite-dl/satellite-data/Geotiff/cdl-geotiff/"



#  Even          Odd
# ......       .......
# |  | |       |  |  |
#    c            c
def square_tile_gen(array, centre, tileSz=64):
    ''' Return: np.array - square numpy slice with coords as centre
    centre - (x,y) img coords
    '''
    x = centre[0]; y = centre[1]
    sz = tileSz//2
    od = tileSz%2
    res = array[:, (x-sz):(x+sz+od), (y-sz):(y+sz+od)].copy()
    return res


def compute_CDL_probs(array):
    """ Compute Probability distribution of 256 classes based of pixelwise classes
        Ratio of number pixels per class to total number of pixels
    Return: [Array] Probability distribution of 256 classes
    """
    arr = array[4,:,:].astype(int)
    prob = np.zeros((256))
    vals = np.unique(arr)
    cnts = []
    for v in vals:
        # cnts.append(np.sum(arr == v))
        prob[v] = np.sum(arr == v) / arr.size
    return prob


def cropland_dataset_gen(array, stride=1, tileSz=50,
                        savePath = "",filePrefix = "tile",
                        tifName=""):
    '''
    Dumps tiles and CDL probabilities [1D 256] of given geoTiff [RGBIR, CDL].
    Naming in matrix type
    '''
    join = os.path.join
    probSavePath = join(savePath,"tiles",filePrefix[0])
    tileSavePath = join(savePath,"cdl_prob",filePrefix[0])
    print("** SAVE PATH:",probSavePath)
    if not os.path.exists(probSavePath): os.makedirs(probSavePath)
    if not os.path.exists(tileSavePath): os.makedirs(tileSavePath)

    C,H,W = array.shape
    sz = tileSz//2
    for i in range(sz, H-sz, tileSz*stride):
        for j in range(sz, W-sz, tileSz*stride):
            tile = square_tile_gen(array, (i,j), tileSz)
            prob = compute_CDL_probs(tile)
            np.save(join(probSavePath,filePrefix[1]+"-"+str(i)+"-"+str(j)+ "-CDL_prob.npy"), prob, allow_pickle=False)
            np.save(join(tileSavePath,filePrefix[1]+"-"+str(i)+"-"+str(j)+ "-tile.npy"), tile.astype(np.uint8), allow_pickle=False)



## ===================== main routine ===========================================
for root, dirc, files in os.walk(READ_PATH):
    savePath = os.path.join(SAVE_PATH)
    if not os.path.exists(savePath):
        print("**Creating the Folder for SAVE:",savePath)
        os.makedirs(savePath)

    for ith , tif in enumerate(files):
        tifPath = os.path.join(root, tif)
        print("Generating crops on file:",tifPath)
        # GDAL array read as [C,H,W] format
        gData = gdal.Open(tifPath)
        geoArray = np.array(gData.ReadAsArray())
        # refer footnotes [1]prefix
        prefixP = root.replace(READ_PATH,"") # for folder path
        prefixN = root.replace(READ_PATH,"").replace("/","-") + "-" + str(ith) # for file name
        prefix = [prefixP,prefixN]
        cropland_dataset_gen(geoArray,
                        stride = 1,
                        savePath = savePath,
                        filePrefix = prefix, # Change PREFIX if Needed
                        tifName = tif,
                        )

'''
FOOT NOTE:
[1]prefix
prefix will add folder Names not in READ_PATH;
FilePath = /home/hogwards/griffindor/harrypotter.tif
READ_PATH = /home/hogwards/
then the prefix will be `griffindor-`
'''
