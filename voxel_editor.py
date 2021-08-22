import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import SinGAN.customFuncs as customFuncs
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images3D/')
parser.add_argument('--input_name', help='input fig', default='simpleforest_80.pt')
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
'''
# for trees_80.pt
wnd = (slice(30, 70), slice(50, 74), slice(40, 70)) 
translation = (10, -50, 6) # Translation vector, used only in editing 
'''
# for simpleforest_80.pt
wnd = (slice(60, 80), slice(34, 60), slice(0, 40)) # x, y, z
translation = (-60, -34, 0) # Translation vector, used only in editing. Set to (0,0,0) to print the wnd selection

if mode == 'editing': # Move voxels to another location
    wnd_to = tuple([slice(max(s.start + d, 0), s.stop + d) for s, d in zip(wnd, translation)]) # Compute target window by applying the translation vector to end
    patch = torch.where(edited[wnd[0], wnd[1], wnd[2]] == -1, 0, 1) # Set to zero to highlight modified volume regions
    edited[wnd_to[0], wnd_to[1], wnd_to[2]] = patch
elif mode == 'recovery': # Remove part of voxels from input
    edited[wnd[0], wnd[1], wnd[2]] = 0 # 1 for empty

# Plot
edited = edited.unsqueeze(0).unsqueeze(0)
customFuncs.visualizeVolume(edited, title='Edited', show=False)
edited[edited == 0] = -1

# Compute mask
mask = torch.zeros_like(edited)
mask[0, 0, wnd_to[0], wnd_to[1], wnd_to[2]] = 1
customFuncs.visualizeMask(mask, title="Mask", show=False)
plt.show()

# Save result
torch.save(edited, outputDir+fileName+'.pt')
torch.save(mask, outputDir+fileName+'_mask.pt')
