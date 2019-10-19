'''
Infer autoencoder with satellite tiles as input-output
Dump Output and Embeddings
'''

import torch
from torch.utils.data import DataLoader
from utilities.satelliteDataset_utils import PlainTileData
from utilities.logger_utils import SAVE2PNG
from networks.autoencoders.SimpleVarient import AutoEncoderVx
import os
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#----------------------------------------------------

SAVE_PATH = "logs/INFERENCE/infer-16B/"
if not os.path.exists(SAVE_PATH):os.makedirs(SAVE_PATH)

#----

model = AutoEncoderVx().to(device)
model_dict = torch.load("weights/autoenc-16b-training/model_epoch-1.pth")
model.load_state_dict(model_dict)
model.eval()

batch_size = 16
DATASET_PATH='/home/jupyter/satellite-dl/satellite-data/Viz/'
infer_dataset = PlainTileData( dataPath= DATASET_PATH)
infer_dataloader = DataLoader(infer_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

#-----

if __name__ =="__main__":
    for jth, (img, readPath) in enumerate(infer_dataloader):
        t0 = time()
        img = img.to(device)
        with torch.no_grad():
            out = model(img)
            
        for i in range(len(readPath)):
            savePath = readPath[i].replace(DATASET_PATH, SAVE_PATH)
            SAVE2PNG( torch.cat([img[i],out[i]],dim=-1) , savePath)
        t1 = time()    
        print('ID:{} Time:{:0.3f}s'.format(jth, t1-t0))
