import torch
import numpy as np
import torch.nn as nn
import SinGAN.customFuncs as customFuncs
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images3D/')
parser.add_argument('--input_name', help='input fig', default='spyrals.pt')
parser.add_argument('--output_dir', help='output fig', default='Input/Editing3D/')
opt = parser.parse_args()
pathName = opt.input_name
fileName = pathName.split('.')[0]
inputDir = opt.input_dir
outputDir = opt.output_dir
original = torch.load(inputDir+pathName)
print('Tensor shape:', original.shape)
edited = original.clone().squeeze()

# Params
mode = 'editing'
wnd = (slice(10, 30), slice(10, 30), slice(10, 30)) # x, y, z slices
translation = (10, 10, 10) # Translation vector, used only in editing

if mode == 'editing': # Move voxels to another location
    wnd_to = tuple([slice(max(s.start + d, 0), s.stop + d) for s, d in zip(wnd, translation)]) # Compute target window by applying the translation vector to end
    patch = torch.where(edited[wnd] == -1, 0, 1) # Set to zero to highlight modified volume regions
    edited[wnd_to] = patch
elif mode == 'recovery': # Remove part of voxels from input
    wnd = (slice(0, 20), slice(0, 30), slice(0, 30))
    edited[wnd] = 1 # 1 for empty

# Plot
edited = edited.unsqueeze(0).unsqueeze(0)
customFuncs.visualizeVolume(edited, title='Edited')
edited[edited == 0] = -1

# Compute mask
mask = torch.zeros_like(edited)
mask[wnd] = 1

# Save result
torch.save(edited, outputDir+fileName+'.pt')
torch.save(mask, outputDir+fileName+'_mask.pt')
