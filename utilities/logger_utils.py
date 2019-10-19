import csv
import numpy as np
import scipy.misc
import os 

def LOG2CSV(data, csv_file, flag = 'a'):
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def SAVE2NUMPY(array, savePath):
    direc = savePath.replace(os.path.basename(savePath), "") 
    if not os.path.exists(direc): os.makedirs(direc)
    out = array.cpu().data.numpy()
    np.save(savePath, out, allow_pickle=False)
    
    
def SAVE2PNG(array, savePath):
    direc = savePath.replace(os.path.basename(savePath), "") 
    if not os.path.exists(direc): os.makedirs(direc)
    out = array.cpu().data.numpy().transpose(1,2,0)
    print(out.shape)
    outPath = savePath.replace(".npy",".png")
    scipy.misc.imsave(outPath, out[:,:,0:3])