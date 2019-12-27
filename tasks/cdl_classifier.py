""" Training and inference for CropDataLayer Classification task

* Groundtruth are prob distribution based on number of pixels contributing t specific class of 256 class 
  (not a typical hot encoding for single class)
"""


import torch
from torch import nn
from torch.utils.data import DataLoader
from utilities.satelliteDataset_utils import tileNembedData
from utilities.logger_utils import LOG2CSV
from networks.ResNet import ResNet_cdl
from utilities.Loss_functions import distributional_CE
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#----------------------------------------------------

TRAIN_NAME = "cdl-training-trial"
if not os.path.exists("logs/"+TRAIN_NAME): os.makedirs("logs/"+TRAIN_NAME)
if not os.path.exists("weights/"+TRAIN_NAME): os.makedirs("weights/"+TRAIN_NAME)

#----

DATASET_PATH='/home/jupyter/satellite-dl/satellite-data/transformer-tokens/matrix-tiles/cities/'


model = ResNet_cdl().to(device)

## --- Pretrained Loader
# pretrained_dict = torch.load("weights/DBCE-LinkSEResnxt101_model.pth")
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# print("Pretrained layers Loaded:", pretrained_dict.keys())
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
## ---



#-----


def training():
    #--- Configurations 
    start_epoch = 0
    num_epochs = 100
    batch_size = 128
    acc_batch = 128 / batch_size
    learning_rate = 1e-3

    criterion = distributional_CE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    
    #--- DataLoaders
    train_dataset = tileNembedData( dataPath= DATASET_PATH) # Loads Images and GroundTruth
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
    
    #--- Train
    for epoch in range(start_epoch, num_epochs):
        acc_loss = float("inf")
        running_loss = []
        for ith, (img, gt, _) in enumerate(train_dataloader):
            img = img.to(device)
            #--- forward
            output = model(img)
            loss = criterion(output, gt) / acc_batch
            acc_loss += loss

            #--- backward
            loss.backward()
            if ( (ith+1) % acc_batch == 0):
                optimizer.step()
                optimizer.zero_grad()
                print('epoch[{}/{}], Mini Batch-{} loss:{:.4f}'
                    .format(epoch, num_epochs, ith//acc_batch, acc_loss.item()))
                running_loss.append(acc_loss.item())
                acc_loss=0
                #break
        LOG2CSV(running_loss, "logs/"+TRAIN_NAME+"/trainLoss.csv")
        epoch_loss = sum(running_loss)/len(train_dataloader)
        print("*** Average Loss on all images [Epoch Loss:{}] ***".format( epoch_loss) )
        LOG2CSV([epoch, epoch_loss], "logs/"+TRAIN_NAME+"/EpochLoss.csv")
        #--- save Checkpoint
        torch.save(model.state_dict(),
                "weights/{}/model_epoch-{}.pth".format(TRAIN_NAME, epoch+1) )
    



if __name__ =="__main__":

    training()
