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
parser.add_argument('--input_name', help='input fig', default='start_scale=3_masked.pt')
parser.add_argument('--input_dir', help='input img dir', default='Output/Editing/trees_8/trees_8_out/')
opt = parser.parse_args()
#completeName = 'Output/RandomSamples/'+pathName+'/gen_start_scale=0/'
#completeName = 'Output/SR/2.0/'
completeName = 'Input/Images3D/'
#completeName = 'Input/Editing3D/'
completeName = 'Output/Editing/spyral/spyral_out/'
#completeName = 'Evaluation/JSONVoxels/'
#onlyfiles = [f for f in listdir(completeName) if isfile(join(completeName, f))]
#for file in onlyfiles:
file = opt.input_name
folder = opt.input_dir
tensor = torch.load(folder+file)
print(tensor.shape)
customFuncs.visualizeVolume(tensor, title=folder+file)