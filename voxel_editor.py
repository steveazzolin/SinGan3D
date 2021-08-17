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
print('Tensor shape:', original.shape, " Unique values:", original.flatten().unique())
edited = original.clone().squeeze()

# Params
mode = 'editing'
wnd = (slice(17, 30), slice(28, 40), slice(15, 35)) #(slice(15, 35), slice(25, 37), slice(20, 35)) # for trees.pt
translation = (-3, -12, -15) #(5, -25, 3) # Translation vector, used only in editing

if mode == 'editing': # Move voxels to another location
    wnd_to = tuple([slice(max(s.start + d, 0), s.stop + d) for s, d in zip(wnd, translation)]) # Compute target window by applying the translation vector to end
    patch = torch.where(edited[wnd[0], wnd[1], wnd[2]] == -1, 0, 1) # Set to zero to highlight modified volume regions
    edited[wnd_to[0], wnd_to[1], wnd_to[2]] = patch
elif mode == 'recovery': # Remove part of voxels from input
    wnd = (slice(0, 20), slice(0, 30), slice(0, 30))
    edited[wnd[0], wnd[1], wnd[2]] = 0 # 1 for empty

# Plot
edited = edited.unsqueeze(0).unsqueeze(0)
customFuncs.visualizeVolume(edited, title='Edited')
edited[edited == 0] = -1

# Compute mask
mask = torch.zeros_like(edited)
mask[0, 0, wnd_to[0], wnd_to[1], wnd_to[2]] = 1
customFuncs.visualizeMask(mask, title="Mask")

# Save result
torch.save(edited, outputDir+fileName+'.pt')
torch.save(mask, outputDir+fileName+'_mask.pt')
