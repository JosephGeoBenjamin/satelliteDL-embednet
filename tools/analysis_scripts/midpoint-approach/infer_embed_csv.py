## Python-3 Script
import os
import glob
import sys
import numpy as np
import scipy.misc
import math as m
import csv

SAVE_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/embed-plots/test/"
READ_PATH = "/home/jgeob/quater-ws/satellite-Embedding/Analytics/image-visuals/2-vertex-viz/"
EMBED_SIZE = 512

def MAGICAL_INFERNCE(npArray):
    return [i for i in range(512)]

def csv_add_row(csvPath, data, mode ='a'):
    '''  write given data as row in CSV file
    csvPath : Path to CSV
    data    : row data as list
    '''
    # import csv
    with open(csvPath, mode) as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(data)

## loaded numpy matrix shape (C,H,W)
def infer_triple_csv(readPath, writePath , embedSize,
                     fileID=""):
    TNUM =3 # Number tiles per tuple
    V1_files =sorted( glob.glob(readPath +"*-V1.npy"))
    V2_files =sorted( glob.glob(readPath +"*-V2.npy"))
    Cen_files =sorted( glob.glob(readPath +"*-Cen.npy"))
    if not V1_files: return
    if (len(V1_files)!=len(V2_files)) or (len(V1_files)!=len(Cen_files)):
        sys.exit("Mismatch in V1,V2,Cen files")

    titleList = ["TileID"] + [i for i in range(embedSize)]
    print(titleList )
    csv_add_row( os.path.join(writePath,fileID+"-V1.csv"), titleList, mode='w')
    csv_add_row( os.path.join(writePath,fileID+"-V2.csv"), titleList, mode='w')
    csv_add_row( os.path.join(writePath,fileID+"-Cen.csv"), titleList, mode='w')

    for ith, (V1, V2, Cen) in enumerate(zip(V1_files, V2_files, Cen_files)):
        ## writing RGB images
        rgbV1 = np.load(V1)
        rgbV2 = np.load(V2)
        rgbCen = np.load(Cen)
        v1Embed = MAGICAL_INFERNCE(rgbV1)
        v2Embed = MAGICAL_INFERNCE(rgbV2)
        cenEmbed = MAGICAL_INFERNCE(rgbCen)
        v1Row = [V1.replace(readPath,"")]   + v1Embed
        v2Row = [V2.replace(readPath,"")]   + v2Embed
        cenRow = [Cen.replace(readPath,"")] + cenEmbed
        csv_add_row( os.path.join(writePath,fileID+"-V1.csv"), v1Row)
        csv_add_row( os.path.join(writePath,fileID+"-V2.csv"), v2Row)
        csv_add_row( os.path.join(writePath,fileID+"-Cen.csv"), cenRow)

        print(fileID, ith)




if __name__ == "__main__":
    if not os.path.exists(READ_PATH): sys.exit("Enter a Valid READ_PATH")
    dirs = glob.glob(READ_PATH+"/**/",recursive=True)
    for dir in dirs:
        savePath = os.path.join(SAVE_PATH, dir.replace(READ_PATH,"") )
        if not os.path.exists(savePath):
            print("**Creating the Folder for SAVE:",savePath)
            os.makedirs(savePath)

        infer_triple_csv(dir, savePath, EMBED_SIZE,
                        fileID = dir.replace(READ_PATH,"").replace("/","-"),
                        )
