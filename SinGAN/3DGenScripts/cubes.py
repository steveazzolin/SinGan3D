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
cubeSize = 8
numCubes = 20
toRtn = torch.full((1,1,imgSize,imgSize,imgSize), 1)
offset = round(cubeSize/2)+1
for l in range(0, numCubes):
    loc = (random.randrange(offset, imgSize-offset-1),random.randrange(offset, imgSize-offset-1),random.randrange(offset, imgSize-offset-1))
    for i in range(loc[0]-4, loc[0]+4):
        for j in range(loc[1]-4, loc[1]+4):
            for k in range(loc[2]-4, loc[2]+4):
                toRtn[0][0][i][j][k] = -1
customFuncs.visualizeVolume(toRtn)
copyCounter = 0
prepath = "../../Input/Images3D/cubes"
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
