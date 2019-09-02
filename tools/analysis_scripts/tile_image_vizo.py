## Python-3 Script
import os
import glob
import sys
import numpy as np
import scipy.misc
import math as m


SAVE_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/image-visuals/test/"
READ_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/image-visuals/2-vertex-viz/"


## loaded numpy matrix shape (C,H,W)
def triple_vizo(readPath, savePath, foldID="",
                tileCountH =10, IRband=True, border=2):
    TNUM =3 # Number tiles per tuple
    V1_files = glob.glob(readPath +"*-V1.npy")
    V2_files = glob.glob(readPath +"*-V2.npy")
    Cen_files = glob.glob(readPath +"*-Cen.npy")
    if not V1_files: return

    ## outimage dim Estimation
    tileCountW = int(m.ceil(len(V1_files) / tileCountH))
    npTile = np.load(V1_files[0]).transpose(1,2,0)
    outPixH = npTile.shape[0] * tileCountH + (tileCountH+1) * border
    outPixW = npTile.shape[1] * tileCountW*TNUM + (tileCountW+1) * border
    rgbMatrix = np.zeros([outPixH, outPixW, 3])
    if (IRband):
        irMatrix = np.zeros([outPixH, outPixW])

    for ith, (V1, V2, Cen) in enumerate(zip(V1_files, V2_files, Cen_files)):
        ## Slice index calculation for copying tiles
        h0 = (ith//tileCountW)*npTile.shape[0] + border*(1 + ith//tileCountW)
        h1 = h0 + npTile.shape[0]
        w0 = (ith%tileCountW)*npTile.shape[1]*TNUM + border*(1 + ith%tileCountW)
        w1 = w0 + npTile.shape[1]
        w2 = w0 + 2*npTile.shape[1]
        w3 = w0 + 3*npTile.shape[1]

        ## writing RGB images
        rgbV1 = np.load(V1)[0:3,:,:].transpose(1,2,0)
        rgbV2 = np.load(V2)[0:3,:,:].transpose(1,2,0)
        rgbCen = np.load(Cen)[0:3,:,:].transpose(1,2,0)
        rgbMatrix[h0:h1, w0:w1] = rgbV1
        rgbMatrix[h0:h1, w1:w2] = rgbCen
        rgbMatrix[h0:h1, w2:w3] = rgbV2

        ## writing IR images
        if (IRband):
            irV1 = np.load(V1)[3,:,:]
            irCen = np.load(Cen)[3,:,:]
            irV2 = np.load(V2)[3,:,:]
            irMatrix[h0:h1, w0:w1] = irV1
            irMatrix[h0:h1, w1:w2] = irCen
            irMatrix[h0:h1, w2:w3] = irV2

    ## Save Images
    scipy.misc.imsave(os.path.join(savePath, foldID+"RGB.jpg"), rgbMatrix )
    if (IRband):
        scipy.misc.imsave(os.path.join(savePath, foldID+"IR.jpg"), irMatrix )




if __name__ == "__main__":
    if not os.path.exists(READ_PATH): sys.exit("Enter a Valid READ_PATH")
    dirs = glob.glob(READ_PATH+"/**/",recursive=True)
    for dir in dirs:
        ## Retain Folder
        # savePath = os.path.join(SAVE_PATH, dir.replace(READ_PATH,"") )
        savePath = SAVE_PATH # save in singleFolder
        if not os.path.exists(savePath):
            print("**Creating the Folder for SAVE:",savePath)
            os.makedirs(savePath)

        triple_vizo(dir, savePath,
                    foldID = dir.replace(READ_PATH,"").replace("/","-"),
                    tileCountH =25,)
