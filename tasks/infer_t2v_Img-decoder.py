'''
Infer - Reconstructed satellite tile from t2v embedding using decoder network
'''

import torch
from torch.utils.data import DataLoader
from utilities.satelliteDataset_utils import tileNembedData
from utilities.logger_utils import SAVE2PNG
from networks.autoencoders.SimpleVarient import AutoEncoderVx
import os
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#----------------------------------------------------

SAVE_PATH = "logs/INFERENCE/t2v_to_image/"
if not os.path.exists(SAVE_PATH):os.makedirs(SAVE_PATH)

#----

model = AutoEncoderVx().decoder.to(device)
model_dict = torch.load("weights/T2V-imgDec-Corrected/freq_model_epoch-4.pth")
model.load_state_dict(model_dict)
model.eval()

batch_size = 16
# [0] Path2Tiles [1] Path2Embeds
DATASET_PATH= ['/home/jupyter/satellite-dl/satellite-data/transformer-tokens/matrix-tiles/cities/T2V-Fresno/', 
               '/home/jupyter/satellite-dl/satellite-data/transformer-tokens/Tile2Vec-embed/cities/T2V-Fresno/']

infer_dataset = tileNembedData( dataPath= DATASET_PATH)
infer_dataloader = DataLoader(infer_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

#-----

if __name__ =="__main__":
    for ith, (img, emb, readPath) in enumerate(infer_dataloader):
        t0 = time()
        img = img.to(device)
        emb = emb.to(device)
        with torch.no_grad():
            out = model(emb)
        for i in range(len(readPath)):
            savePath = SAVE_PATH + os.path.basename(readPath[i])
            SAVE2PNG( torch.cat([img[i],out[i]],dim=-1) , savePath)
        t1 = time()
        print('ID:{} Time:{:0.3f}s'.format(ith, t1-t0))
        
        if ith > 30: break