''' Dimentionality reduction of Embedding using PCA

* Applied to Tile2Vec embedding to obtain reduced dimenional embedding

Reference: https://stackoverflow.com/a/31432111

'''

from torch.utils.data import DataLoader
from utilities.satelliteDataset_utils import SimpleNumpyData
from utilities.logger_utils import SAVE2NUMPY
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pickle
import os

# READ_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/Tile2Vec-embed/cities/"
READ_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/Tile2Vec-embed/cities/"
FIT_OBJ = "weights/IncPCA-t2v.pickled"
SAVE_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/t2v_pca-embed/"

batch_size = 4096 * 2
np_dataset = SimpleNumpyData(dataPath = READ_PATH )



def train():
    # Dataloader
    np_dataloader = DataLoader(np_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    # incremental-PCA 
    icpa = IncrementalPCA(n_components=256, batch_size=batch_size)
    for epoch in range(2):
        tot_data_len = len(np_dataloader)
        for ith, (embs, _) in enumerate(np_dataloader):
            icpa.partial_fit(embs)
            print("Mini Batch-{}/{}".format(ith+1, tot_data_len) )
            if ((ith%100) == 0):
                with open(FIT_OBJ, 'wb') as file_pk:
                    pickle.dump(icpa, file_pk)
                print("Dumping Object")

        with open(FIT_OBJ, 'wb') as file_pk:
            pickle.dump(icpa, file_pk)
        print("END OF EPOCH")


def inference():
    # dataloader
    infer_dataloader = DataLoader(np_dataset, batch_size=1,
                         shuffle=False, num_workers=0)
    
    pickle_in = open(FIT_OBJ,"rb")
    icpa = pickle.load(pickle_in)

    tot_data_len = len(infer_dataloader)
    for ith, (emb, path) in enumerate(infer_dataloader):
        red_emb = icpa.transform(emb)
        savePath = os.path.join(SAVE_PATH, path[0].replace(READ_PATH,""))
        SAVE2NUMPY(red_emb, savePath)
        print("infered-{}/{}".format(ith+1, tot_data_len) )



if __name__ =="__main__":
    inference()
