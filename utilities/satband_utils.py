'''Satellite-Bands utilities.

Repository:
[1] https://github.com/GokulNC/satellite-embeddings/

'''

import numpy as np
from imageio import imread

def clip_and_scale_bands(bands, dataset='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    
    if dataset in ['naip', 'rgb']:
        clip_min = 0; clip_max = 255
    
    return np.clip(bands, clip_min, clip_max) / (clip_max - clip_min)

def random_flip_and_rotate(img):
    """
    Does random horizontal & vertical flip and random rotation by
    (0, 90, 180, 270) degrees. Assumes img in CHW.
    """
    # Randomly horizontal flip
    if np.random.rand() < 0.5: img = np.flip(img, axis=2).copy()
    # Randomly vertical flip
    if np.random.rand() < 0.5: img = np.flip(img, axis=1).copy()
    # Randomly rotate
    rotations = np.random.choice([0, 1, 2, 3])
    if rotations > 0: img = np.rot90(img, k=rotations, axes=(1,2)).copy()
    return img
    
def load_image(path):
    """
    Reads an image, converts from HWC to CHW and returns it
    """
    img = imread(path)
    return np.moveaxis(img, -1, 0) # HWC to CHW

