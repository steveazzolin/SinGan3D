import torch
import numpy as np
import torch.nn as nn
import math
import os
import random
import matplotlib.pyplot as plt
import SinGAN.customFuncs as customFuncs
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join


parser = ArgumentParser()
parser.add_argument('--input_name', help='input fig', default='')
opt = parser.parse_args()
pathName = opt.input_name
#completeName = 'Output/RandomSamples/'+pathName+'/gen_start_scale=0/'
#completeName = 'Output/SR/2.0/'
completeName = 'Input/Images3D/'
completeName = 'Input/Editing3D/'
completeName = 'Output/Editing/spyral/spyral_out/'
#completeName = 'Evaluation/JSONVoxels/'
#onlyfiles = [f for f in listdir(completeName) if isfile(join(completeName, f))]
#for file in onlyfiles:
file = pathName
tensor = torch.load(completeName+file)
print(tensor.shape)
customFuncs.visualizeVolume(tensor)