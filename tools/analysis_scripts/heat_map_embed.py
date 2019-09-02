## Python-3 Script
import os
import glob
import sys
import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc

READ_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/embed-plots/inference_26_Aug/EMBEDS/"

SAVE_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/embed-plots/Result/"
# EMBED_SIZE = 256 Not Used


from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
def embed_heatmap(embedArr, embedDim, figsize = (64,64), cmap='hot' ):
    '''Return: np.array - heatmap plot as rgb-image
    embedArr : np.array - embed values
    embedDim : tuple - (H,W) values
    figSize  : tuple - (H,W) resolution
    '''

    plotArr = embedArr.reshape(embedDim)
    fig = Figure(figsize=(figsize[1],figsize[0]), dpi = 1) # figsize=WxH
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(plotArr, cmap= cmap, interpolation='nearest', aspect='auto')
    ax.set_axis_off()
    fig.add_axes(ax)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    w,h = canvas.get_width_height()

    argbArr = np.fromstring(canvas.tostring_argb(), dtype='uint8')
    argbArr = argbArr.reshape((h,w,4))
    rgbArr = argbArr[:,:,1:]
    # scipy.misc.imsave("IR.png", rgbArr)
    return rgbArr


def embed_1dnp_loader(readPath, filePattern="", log_scale=False):
    csvFile = glob.glob(readPath+"*"+filePattern+".csv")
    if not csvFile:
        print("embed_1dnp_loader skipped for:",readPath)
        return []
    print("Loading....:",readPath)
    dataframe = pd.read_csv(csvFile[0]) # takes only first file matching the pattern
    dataframe = dataframe.drop(columns="TileID")
    rowCount, colCount = dataframe.shape
    embedList = []
    for ith in range(rowCount):
        rowArr = dataframe.iloc[ith,:].to_numpy()
        if log_scale: rowArr = np.log(rowArr)
        embedList.append(rowArr.reshape(-1,1))
    return embedList.copy()

def embed_2dnp_loader(readPath, filePattern="", log_scale=False):
    csvFile = glob.glob(readPath+"*"+filePattern+".csv")
    if not csvFile:
        print("embed_1dnp_loader skipped for:",readPath)
        return []
    print("Loading....:",readPath)
    dataframe = pd.read_csv(csvFile[0]) # takes only first file matching the pattern
    dataframe = dataframe.drop(columns="TileID")
    rowCount, colCount = dataframe.shape
    matDim = int(m.ceil(m.sqrt(colCount)))
    embedList = []
    for ith in range(rowCount):
        rowArr = dataframe.iloc[ith,:].to_numpy()
        rowArr = np.resize(rowArr,(1, matDim*matDim))
        if log_scale: rowArr = np.log(rowArr)
        matArr = rowArr.reshape((matDim,matDim))
        embedList.append(matArr)
    return embedList.copy()


## Different analytics plots of Embedding saved as png
def embed_triple_plots(readPath, savePath,
                        fileID="", tileCountH =10):
    '''
    readPath: Path to files, not recursive -takes only files in this directory
    savePath: Path to save file
    '''
    TNUM =3 # Number tiles per tuple
    tupleBorder =5
    tileBorder =2

    V1_emList =  embed_2dnp_loader(readPath, "V1", log_scale=False)
    V2_emList =  embed_2dnp_loader(readPath, "V2", log_scale=False)
    Cen_emList = embed_2dnp_loader(readPath, "Cen",log_scale=False)
    if not V1_emList:
        print("embed_triple_plots skipped for:",readPath)
        return 0

    ## outimage dim Estimation
    tupleCountW = int(m.ceil(len(V1_emList) / tileCountH))
    npTile = np.zeros((100,100)) # plot figure size per embedding
    outPixH = npTile.shape[0] * tileCountH + (tileCountH+1) * tupleBorder
    outPixW =( npTile.shape[1] * tupleCountW * TNUM # tuple size
                + (tupleCountW+1) * tupleBorder        # tuple border size
                + (tupleCountW) * tileBorder * (TNUM-1)     # tile border size
                )
    hmMatrix = np.zeros([outPixH, outPixW, 3])

    for ith, (V1, V2, Cen) in enumerate(zip(V1_emList, V2_emList, Cen_emList)):
        ## Slice index calculation for copying tiles
        h0 = (ith//tupleCountW)*npTile.shape[0] + tupleBorder*(1 + ith//tupleCountW)
        h1 = h0 + npTile.shape[0]
        w0 =( (ith%tupleCountW)*npTile.shape[1]*TNUM  # inc tuple size (3tile)
                + (1 + ith%tupleCountW) * tupleBorder     # inc tuple border size
                + (ith%tupleCountW)* tileBorder * (TNUM-1)   # inc tile border size
                )
        w1 = w0 + npTile.shape[1]
        w1_ = w1+tileBorder
        w2 = w0 + 2*npTile.shape[1]+tileBorder
        w2_ = w2+tileBorder
        w3 = w0 + 3*npTile.shape[1]+ (2*tileBorder)

        ## writing RGB images
        hmV1 = embed_heatmap(V1, V1.shape, figsize = npTile.shape)
        hmV2 = embed_heatmap(V2, V2.shape, figsize = npTile.shape)
        hmCen = embed_heatmap(Cen, Cen.shape, figsize = npTile.shape)
        hmMatrix[h0:h1, w0:w1] = hmV1
        hmMatrix[h0:h1, w1_:w2] = hmCen
        hmMatrix[h0:h1, w2_:w3] = hmV2

    ## Save Images
    scipy.misc.imsave(os.path.join(savePath, fileID+"heatPlot.jpg"), hmMatrix )





if __name__ == "__main__":
    if not os.path.exists(READ_PATH): sys.exit("Enter a Valid READ_PATH")
    dirs = glob.glob(READ_PATH+"/**/",recursive=True)
    for dir in dirs:
        # savePath = os.path.join(SAVE_PATH, dir.replace(READ_PATH,"") )
        savePath = SAVE_PATH
        if not os.path.exists(savePath):
            print("**Creating the Folder for SAVE:",savePath)
            os.makedirs(savePath)

        embed_triple_plots(dir, savePath,
                        fileID = dir.replace(READ_PATH,"").replace("/","-"),
                        tileCountH =25,
                        )
