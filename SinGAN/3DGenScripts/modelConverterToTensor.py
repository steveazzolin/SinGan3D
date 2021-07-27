import torch
import numpy as np
import torch.nn as nn
import math
import os
import random
import matplotlib.pyplot as plt
import sys
import customFuncs
import json

#imgSize = 40
imgSize = 80
jsonFile = open("../../JSONModels/crystalcluster2.json")
jsonObj = json.load(jsonFile)
offsetWidth = math.floor((imgSize - int(jsonObj["dimension"][0]["width"])) / 2.0)
offsetDepth = math.floor((imgSize - int(jsonObj["dimension"][0]["depth"])) / 2.0)
offsetHeight = math.floor((imgSize - int(jsonObj["dimension"][0]["height"])) / 2.0)
toRtn = torch.full((1,1,imgSize,imgSize,imgSize), 1)
print(offsetWidth)
print(offsetHeight)
print(offsetDepth)
for jsonVoxel in jsonObj["voxels"]:
    x = int(jsonVoxel["x"])
    y = int(jsonVoxel["y"])
    z = int(jsonVoxel["z"])
    #print(str(x) + " " + str(y) + " " + str(z))
    if "value" not in jsonVoxel or jsonVoxel["value"] < 0:
        toRtn[0][0][x + offsetWidth][z + offsetDepth][y + offsetHeight] = -1

customFuncs.visualizeVolume(toRtn)
copyCounter = 0
prepath = "../../Input/Images3D/crystalCluster2"
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