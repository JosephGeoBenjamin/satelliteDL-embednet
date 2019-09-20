'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Jean, Neal, et al.
    Tile2Vec: Unsupervised representation learning for spatially distributed data.
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.

Base Code:
[1] https://github.com/ermongroup/tile2vec/blob/master/examples/Example%203%20-%20Tile2Vec%20features%20for%20CDL%20classification.ipynb


Note:
Download corresponding weights from https://github.com/ermongroup/tile2vec/ before running

'''

import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable

import sys
import glob
from networks.Tile2Vec_net import make_tilenet, ResNet18

READ_PATH = "datasets/tile2vec-data"
SAVE_PATH = "embed-results/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Setting up model
in_channels = 4
z_dim = 512
cuda = torch.cuda.is_available()
# tilenet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# Use old model for now
tilenet = ResNet18()
tilenet.to(device)


# Load parameters
model_fn = 'weights/tile2vec/naip_trained.ckpt'
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()


# Embed tiles
def tile2vec_embed_generator( readPath, savePath )

    tileList = dirs = glob.glob(readPath+"/*.npy",recursive=False)
    for tilePath in tileList:
        #Loaded numpy files are in Torch format CHW 4 channel
        tile = np.load(tilePath)
        tile = np.expand_dims(tile, axis=0)
        tile = tile / 255 # Scale to [0, 1]
        # Embed tile
        tile = torch.from_numpy(tile).float()
        tile = Variable(tile)
        tile = tile.to(device)
        em = tilenet.encode(tile)
        em = em.cpu()
        em = em.data.numpy()
        emPath = os.path.join(savePath, os.path.basename(tilePath).replace("L-", "TVE-"))
        np.save()


if __name__ == "__main__":
    if not os.path.exists(READ_PATH): sys.exit("Enter a Valid READ_PATH")
    dirs = glob.glob(READ_PATH+"/**/",recursive=True)
    for dir in dirs:
        ## Retain Folder structure
        savePath = os.path.join(SAVE_PATH, dir.replace(READ_PATH,"") )
        if not os.path.exists(savePath):
            print("Creating the Folder for SAVE:",savePath)
            os.makedirs(savePath)
        t0 = time()
        tile2vec_embed_generator(  readPath = dir,
                                    savePath = savePath
                                    )
        t1 = time()
        print('Time: {:0.3f}s'.format(t1-t0))
