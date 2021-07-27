import torch
import numpy as np
import torch.nn as nn
import math
import os
import random
import matplotlib.pyplot as plt
import sys
import customFuncs


imgSize = 80 #40
spyralingfactor = 10
spyralthickness = 2
numspyrals = 5
toRtn = torch.full((1,1,imgSize,imgSize,imgSize), 1)
interval = 360.0/numspyrals
intervalcurrent = 0
for z in range(0, numspyrals):
    if(numspyrals != 1):
        intervalcurrent = intervalcurrent + interval
        tmpoffx = math.cos(math.radians(intervalcurrent))
        tmpoffy = math.sin(math.radians(intervalcurrent))
        offx = (imgSize/2) + (tmpoffx * imgSize/3)
        offy = (imgSize/2) + (tmpoffy * imgSize/3)
    else:
        offx = (imgSize/2)
        offy = (imgSize/2)
    for i in range(0, imgSize):
        divider = imgSize/spyralingfactor
        dividedz = i/divider
        x = round(math.cos(dividedz)*divider + offx)
        y = round(math.sin(dividedz)*divider + offy)
        if i >= 0 and i <=imgSize and x >= 0 and x <=imgSize and y >= 0 and y <=imgSize:
            for j in range(0, spyralthickness):
                for k in range(0, spyralthickness):
                    toRtn[0][0][x+j][y+k][i] = -1
                    toRtn[0][0][x-j][y-k][i] = -1
                    toRtn[0][0][x+j][y-k][i] = -1
                    toRtn[0][0][x-j][y+k][i] = -1
                    toRtn[0][0][x-j][y][i] = -1
                    toRtn[0][0][x+j][y][i] = -1
                    toRtn[0][0][x][y-j][i] = -1
                    toRtn[0][0][x][y+j][i] = -1
customFuncs.visualizeVolume(toRtn)
copyCounter = 0
prepath = "../../Input/Images3D/spyrals2"
tmppath = prepath + ".pt"
while(os.path.exists(tmppath)):
    try:
        #shutil.rmtree(dir2save)
        copyCounter += 1
        tmppath = prepath + "_" + str(copyCounter) + ".pt"
    except OSError:
        pass
try:
    customFuncs.save3DFig(toRtn, tmppath)
except OSError:
    pass
