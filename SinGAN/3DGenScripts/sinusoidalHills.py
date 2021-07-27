import torch
import numpy as np
import torch.nn as nn
import math
import os
import random
import matplotlib.pyplot as plt
import sys
import customFuncs


imgSize = 40
hillsWidth = int(round(imgSize/3))
xPeriodDivider = 3
yPeriodDivider = 3
hillPeakAmplification = 10

toRtn = torch.full((1, 1, imgSize, imgSize, imgSize), 1)
startingPoint = int(round((imgSize-hillsWidth)/2))
for i in range(0, imgSize):
    for j in range(0, imgSize):
        x = i#(i-round(size[2]/2))
        y = j#(j-round(size[3]/2))
        z = math.ceil(((math.sin(x/xPeriodDivider)-math.cos(y/yPeriodDivider))*hillPeakAmplification))
        for k in range(0, hillsWidth):
            zSpot = z+startingPoint+k
            if zSpot >= 0 and zSpot < imgSize:
                toRtn[0][0][i][j][z+startingPoint+k] = -1
customFuncs.visualizeVolume(toRtn)
copyCounter = 0
prepath = "../../Input/Images3D/sinusoidalWavest"
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
