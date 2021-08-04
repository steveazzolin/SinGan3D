import torch
import numpy as np
import torch.nn as nn
import math
import os
import random
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize3D

def get3D(toConvert):
    #print(toConvert[0][0][38][32]) # 0,0 upper left corner x goes down y goes right
    expandedTensor = toConvert.unsqueeze(-1).expand(toConvert.shape[0], toConvert.shape[1], toConvert.shape[2], toConvert.shape[3], toConvert.shape[2])
    tmp = toConvert[0][:][:][:]
    #plt.imshow(tmp.permute((1, 2, 0)).cpu().numpy())
    #plt.show()
    expandedTensor = torch.full((toConvert.shape[0], 1, toConvert.shape[2], toConvert.shape[3], toConvert.shape[3]), 1)
    #print(expandedTensor.shape)
    for i in range(0, toConvert.shape[2]):
        for j in range(0, toConvert.shape[3]):
            for k in range(0, expandedTensor.shape[4]):
            #Zcoord = (math.floor(toConvert.shape[3]/2)-1)
                expandedTensor[0][0][i][j][k] = toConvert[0][0][i][j]#(1 - norm01(toConvert[0][0][i][j])) > 0.5
            #print(toConvert[0][0][i][j])
            #expandedTensor[0][0][i][j][Zcoord+1] = expandedTensor[0][0][i][j][Zcoord]
            #expandedTensor[0][0][i][j][Zcoord+2] = expandedTensor[0][0][i][j][Zcoord]
            #expandedTensor[0][0][i][j][Zcoord-1] = expandedTensor[0][0][i][j][Zcoord]
            #expandedTensor[0][0][i][j][Zcoord-2] = expandedTensor[0][0][i][j][Zcoord]
    #print(expandedTensor[0][0][37][32][31])
    #print(expandedTensor[0][0][38][32][30])
    #print(expandedTensor[0][0][39][32][31])
    #plt.imshow(tmp.permute((1, 2, 0)).cpu().numpy())
    #plt.show()
    #tmp = expandedTensor[0][0][:][:][:].clone()
    #for i in range(0, expandedTensor.shape[2]):
        #for j in range(0, expandedTensor.shape[3]):
            #for k in range(0, expandedTensor.shape[4]):
                #tmp[i][j][k] = -300#expandedTensor[0][0][i][j][k] < 0
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.voxels(tmp, edgecolor='k')

    #plt.show()
    visualizeVolume(expandedTensor)
    return expandedTensor
    
def genImageFunc(size):
    toRtn = torch.full((size[0],size[1],size[2],size[3],size[4]), 1)
    width = int(round(size[4]/3))
    startingPoint = int(round((size[4]-width)/2))
    for i in range(0, size[2]):
        for j in range(0, size[3]):
            x = i#(i-round(size[2]/2))
            y = j#(j-round(size[3]/2))
            z = math.ceil(((math.sin(x/3)-math.cos(y/3))*4))
            for k in range(0, width):
                toRtn[0][0][i][j][z+startingPoint+k] = -1
                #toRtn[0][0][i][j][z+startingPoint+15+k] = -1
    visualizeVolume(toRtn)
    return toRtn
    
def genImage3Dv2(size):
    toRtn = torch.full((size[0],size[1],size[2],size[3],size[4]), 1)
    for i in range(0, size[2]):
        for j in range(0, size[3]):
            for k in range(0, round(size[4]/5)):
                toRtn[0][0][i][j][k] = -1
    for l in range(0, 20):
        loc = (random.randrange(8,35),random.randrange(8,35),random.randrange(round(size[4]/5),35))
        for i in range(loc[0]-4, loc[0]+4):
            for j in range(loc[1]-4, loc[1]+4):
                for k in range(loc[2]-4, loc[2]+4):
                    toRtn[0][0][i][j][k] = -1
    visualizeVolume(toRtn)
    return toRtn
    
def genImageSpyral(size):
    toRtn = torch.full((size[0],size[1],size[2],size[3],size[4]), 1)
    for i in range(0, size[4]):
        divider = size[4]/10
        dividedz = i/divider
        x = round(math.cos(dividedz)*divider + size[2]/2)
        y = round(math.sin(dividedz)*divider + size[3]/2)
        if i >= 0 and i <=size[4] and x >= 0 and x <=size[2] and y >= 0 and y <=size[3]:
            toRtn[0][0][x][y][i] = -1
            toRtn[0][0][x+1][y+1][i] = -1
            toRtn[0][0][x-1][y-1][i] = -1
            toRtn[0][0][x+1][y-1][i] = -1
            toRtn[0][0][x-1][y+1][i] = -1
            toRtn[0][0][x-1][y][i] = -1
            toRtn[0][0][x+1][y][i] = -1
            toRtn[0][0][x][y-1][i] = -1
            toRtn[0][0][x][y+1][i] = -1
    visualizeVolume(toRtn)
    return toRtn
        
        
def get3DPyramid(real,reals,opt):
    real = real[:,0:1,:,:]
    print(opt.stop_scale)
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        print(opt.stop_scale-i)
        print(scale)
        print(real.shape)
        curr_real = imresize3D(real,scale,opt)
        print(curr_real.shape)
        #print(curr_real)
        #visualizeVolume(curr_real)
        reals.append(curr_real)
    return reals
    
def norm01(num):
    return (num+1)/(2)
    
def visualizeVolume(tensor, title=None):
    voxels = tensor[0][0][:][:][:] < 0
    edited = tensor[0][0][:][:][:] == 0 # Highlight zero values
    fig = plt.figure()
    if title is not None: 
        fig.canvas.set_window_title(title)
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, edgecolor='k')
    ax.voxels(edited, facecolors='r', edgecolor='k')
    plt.show()
    
def save3DFig(tensor, file):
    torch.save(tensor, file)
    
def load3DFig(file):
    tensor = torch.load(file)
    return tensor