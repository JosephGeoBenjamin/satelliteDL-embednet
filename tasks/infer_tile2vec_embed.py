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

READ_PATH = "/home/jupyter/satellite-dl/satellite-data/matrix-tiles/cities/Carmel-by-Sea/Carmel-by-Sea-0/"
SAVE_PATH = "/home/jupyter/satellite-dl/satellite-data/embed-results/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Setting up model

in_channels = 4
z_dim = 512
batch_size = 8
tilenet = ResNet18()
tilenet.to(device)


infer_dataset = PlainTileData(dataPath = READ_PATH )
infer_dataloader = DataLoader(infer_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=8)

# Load parameters
model_fn = 'weights/naip_trained.ckpt'
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()


if __name__ == "__main__":
    for jth, (img, path) in enumerate(infer_dataloader):
        t0 = time()
        img = img.to(device)
        with torch.no_grad():
            outEm = tilenet.encode(img)

        for i in range(len(path))
            fullPath = path[i].replace(READ_PATH, SAVE_PATH)
            emPath = fullPath.replace( os.path.basename(fullPath),
                                os.path.basename(fullPath).replace("L-", "TVE-"))
            em = outEm[i].cpu().data.numpy()
            np.save(emPath, em, allow_pickle=False)

        t1 = time()
        print('ID:{} Time:{:0.3f}s'.format(jth, t1-t0))
