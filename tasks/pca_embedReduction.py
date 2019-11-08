''' Dimentionality reduction of Embedding using PCA

* Applied to Tile2Vec embedding to obtain reduced dimenional embedding

Reference: https://stackoverflow.com/a/31432111

'''

from torch.utils.data import DataLoader
from utilities.satelliteDataset_utils import SimpleNumpyData
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pickle


READ_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/Tile2Vec-embed/cities/"
FIT_OBJ = "weights/IncPCA-t2v.pickled"

batch_size = 4096 * 2
np_dataset = SimpleNumpyData(dataPath = READ_PATH )
np_dataloader = DataLoader(np_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


def train():
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
    pass



if __name__ =="__main__":
    train()
