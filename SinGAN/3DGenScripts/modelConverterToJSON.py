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
from os import listdir
from os.path import isfile, join


#input = customFuncs.load3DFig("../../Dataset/Rocks/RandomlyGeneratedSamples/5.pt")
#input = customFuncs.load3DFig("../../Dataset/Cubes/cubes.pt")
inputFile = "";
name = "0"
partialInput = "../../Output/SR/crystalCluster/2.0/";
#partialInput = "../../Input/Images3D/";
inputFile = partialInput + name + ".pt";
input = customFuncs.load3DFig(inputFile);
inputSize = input.size();
xSize = inputSize[2];
zSize = inputSize[3];
ySize = inputSize[4];

outObject = {"dimension":[{"width":xSize,"height":ySize,"depth":zSize}],"voxels":[]}

for x in range(0, xSize):
    for z in range(0, zSize):
        for y in range(0, ySize):
            outObject["voxels"].append({"x":x,"y":y,"z":z,"value":input[0][0][x][z][y].item()});

copyCounter = 0
prepath = "../../Evaluation/SR/";
if not os.path.exists(prepath):
    os.makedirs(prepath);
tmppath = prepath + name + ".json"
while(os.path.exists(tmppath)):
    try:
        #shutil.rmtree(dir2save)
        copyCounter += 1
        tmppath = prepath + "_" + str(copyCounter) + ".json"
    except OSError:
        pass
try:
    with open(tmppath, 'w') as outfile:
        json.dump(outObject, outfile)
except OSError:
    pass
    
