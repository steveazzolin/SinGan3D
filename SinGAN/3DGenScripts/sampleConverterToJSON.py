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
partialInput = "../../Dataset/";
inputList = ["AbstractObject", "City", "Crystal", "CrystalCluster", "Cubes", "Rocks", "SimpleForest", "SinWaves", "Spyrals", "Trees"];
for name in inputList:
    inputFolder = partialInput + name + "/RandomlyGeneratedSamples/";
    onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
    for file in onlyfiles:
        print(file);
        inputFile = partialInput + name + "/RandomlyGeneratedSamples/";
        inputFile = inputFile + file;
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
        prepath = "../../Evaluation/" + name + "/Samples/";
        if not os.path.exists(prepath):
            os.makedirs(prepath);
        tmppath = prepath + file[:-3] + ".json"
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
    
