import numpy as np
from osgeo import gdal
import scipy.misc
import os
import csv

SAVE_PATH = "/home/jgeob/quater-ws/satellite-Embedding/geotiff/out/"
READ_PATH = "/home/jgeob/quater-ws/satellite-Embedding/geotiff/data/"

def pixel2coord(coordsInfo ,x, y):
    """Returns global coordinates from pixel x, y coords"""
    yoff= coordsInfo["yoff"]; a=  coordsInfo["a"]; b=  coordsInfo["b"]
    xoff=   coordsInfo["xoff"]; d=  coordsInfo["d"]; e=  coordsInfo["e"]
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def tile_logger(filePath, tileID, vertexList,
                coordsInfo=0, tifName = ""):
    ''' Creates CSV with data on each tile
    vertexList: list of pixel coord-tuple of vertices
    '''
    latLonList =[]
    if coordsInfo != 0:
        latLonList = [pixel2coord(coordsInfo,x,y) for x,y in vertexList]
    with open(filePath+"/Coordinates_log.csv", 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow( [tifName]+[tileID] + vertexList + latLonList )
    csvFile.close()


#  Even          Odd
# ......       .......
# |  | |       |  |  |
#    c            c
def square_tile_gen(array, centre, tileSz=64):
    ''' Return: np.array - square numpy slice with coords as centre
    centre - (x,y) img coords
    '''
    x = centre[0]; y = centre[1]
    sz = tileSz//2
    od = tileSz%2
    res = array[:, (x-sz):(x+sz+od), (y-sz):(y+sz+od)].copy()
    return res

def two_vertex_midpoint_tiler(array, stride=1, tileSz=64,
                     savePath = "",filePrefix = "tile",
                     coordsInfo = 0, tifName=""):
    '''
    Dumps tiles from 2 points and midpoint tile
    '''
    join = os.path.join
    C,H,W = array.shape
    sz = tileSz//2
    for i in range(sz, H-sz, tileSz*stride):
        for j in range(sz, W-sz, tileSz*stride):
            i2 = np.random.randint(sz, H-sz)
            j2 = np.random.randint(sz, W-sz)
            ic, jc =( (i2+i)//2, (j2+j)//2 )
            #Vertices
            V1 = square_tile_gen(array, (i,j), tileSz)
            V2 = square_tile_gen(array, (i2,j2), tileSz)
            # Midpoint
            Cen = square_tile_gen(array, (ic,jc), tileSz)

            tile_logger(savePath, filePrefix+"-"+str(i)+"-"+str(j),
                        [(ic,jc), (i,j), (i2,j2)],
                        coordsInfo=coordsInfo, tifName = tifName)
            ####
            #! ! ! CHANGE the `np.astype` to required type
            np.save(join(savePath,filePrefix+"-"+str(i)+"-"+str(j)+ "-V1.npy") ,V1.astype(np.uint8)    , allow_pickle=False)
            np.save(join(savePath,filePrefix+"-"+str(i)+"-"+str(j)+ "-V2.npy") ,V2.astype(np.uint8)    , allow_pickle=False)
            np.save(join(savePath,filePrefix+"-"+str(i)+"-"+str(j)+"-Cen.npy") ,Cen.astype(np.uint8), allow_pickle=False)
            ###
            ###scipy.misc.imsave(savePath+filePrefix+"-"+str(i)+"-"+str(j)+ "-V1.jpg",(V1[0:3,:,:].transpose(1,2,0)) )


def matrix_tiler(array, stride=1, tileSz=50,
                     savePath = "",filePrefix = "tile",
                     coordsInfo = 0, tifName=""):
    '''
    Dumps tiles of given geoTiff. Naming in matrix type
    '''
    join = os.path.join
    subSavePath = join(savePath, filePrefix)
    if not os.path.exists(subSavePath):
        print("** sub SAVE PATH:",subSavePath)
        os.makedirs(subSavePath)

    C,H,W = array.shape
    sz = tileSz//2
    for i in range(sz, H-sz, tileSz*stride):
        for j in range(sz, W-sz, tileSz*stride):

            L = square_tile_gen(array, (i,j), tileSz)

            tile_logger(savePath, filePrefix+"-"+str(i)+"-"+str(j),
                        [(i,j)],
                        coordsInfo=coordsInfo, tifName = tifName)
            ####
            #! ! ! CHANGE the `np.astype` to required type
            np.save(join(subSavePath,filePrefix+"L-"+str(i//tileSz)+"-"+str(j//tileSz)+ ".npy") ,L.astype(np.uint8)    , allow_pickle=False)
            ###
            # scipy.misc.imsave(join(subSavePath,filePrefix+"L-"+str(i//tileSz)+"-"+str(j//tileSz)+ ".jpg") ,(L[0:3,:,:].transpose(1,2,0)) )



for root, dirc, files in os.walk(READ_PATH):
    savePath = os.path.join(SAVE_PATH,root.replace(READ_PATH,"") )
    if not os.path.exists(savePath):
        print("**Creating the Folder for SAVE:",savePath)
        os.makedirs(savePath)

    for ith , tif in enumerate(files):
        tifPath = os.path.join(root, tif)
        print("Generating crops on file:",tifPath)
        # refer footnotes [1]prefix
        prefix = root.replace(READ_PATH,"").replace("/","-") + "-" + str(ith)

        # GDAL array read as [C,H,W] format
        gData = gdal.Open(tifPath)
        geoArray = np.array(gData.ReadAsArray())
        gInfo = gData.GetGeoTransform()
        matrix_tiler(geoArray,
                        stride = 1,
                        savePath = savePath,
                        filePrefix = prefix, # Change PREFIX if Needed
                        coordsInfo = dict(zip(['xoff', 'a', 'b', 'yoff', 'd', 'e'], gInfo)),
                        tifName = tif,
                        )

'''
FOOT NOTE:

[1]prefix
prefix will add folder Names not in READ_PATH;
FilePath = /home/hogwards/griffindor/harrypotter.tif
READ_PATH = /home/hogwards/
then the prefix will be `griffindor-`
'''
