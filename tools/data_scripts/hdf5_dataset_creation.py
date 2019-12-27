'''
Convert collection numpy files into HDF5 datasets
'''
import h5py
import numpy as np
import glob
import os

READ_PATH = "/home/jupyter/satellite-dl/satellite-data/transformer-tokens/matrix-tiles/cities/"

def findGridSz(np_files): # starts from 0
    height = 0; width = 0
    for f in np_files:
        name = os.path.basename(f)
        nameSplit = name.replace(".","-").split("-")
        if height < int(nameSplit[-3]): height = int(nameSplit[-3])
        if width < int(nameSplit[-2]): width = int(nameSplit[-2])
    return height, width

def parsePosition(file): # starts from 0
    name = os.path.basename(file)
    nameSplit = name.replace(".","-").split("-")
    height = int(nameSplit[-3])
    width = int(nameSplit[-2])
    return height, width


#========================= Main Routine ===============================

hdf_file =h5py.File("matrix-tiles-cities.hdf5", "w-")

folders = glob.glob(READ_PATH+"/**/",recursive=True)
print("Folder Read Complete .....")
for folder in folders:
    np_files = glob.glob(folder+"/*.npy",recursive=False)
    if not np_files:
        print ("Skipping the folder:", folder)
        continue
    hL, wL = findGridSz(np_files)
    dsetName  = list(filter(None, folder.split("/")))[-1]
    print ("******************",dsetName, hL, wL)
    dset = hdf_file.create_dataset( dsetName,
                                    (hL+1, wL+1, 4, 50, 50),
                                    dtype=np.uint8, #### CHANGE AS NEEDED
                                    chunks=(1, 1, 4, 50, 50),
                                    )
    for f in np_files:
        np_data = np.load(f)
        hP, wP = parsePosition(f)
        print(hP, wP)
        dset[hP, wP, :,:,:] = np_data


hdf_file.close()
