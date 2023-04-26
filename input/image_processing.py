# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:46:50 2021

@author: Yudu Li
"""

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Numpy FFT/IFFT
# =============================================================================

## ------- size_dims_simp ------- ##
def size_dims_simp(data,dims):
    datasize = []
    for n in range(len(dims)):
        datasize.append(data.shape[dims[n]])
    
    return datasize

## --------- Fn_x2k ---------- ##
def Fn_x2k_numpy(image, dims, dont_shift=False):
    
    scale_fctr    = 1/np.sqrt(np.prod(size_dims_simp(image, dims)))
    
    if dont_shift:
        for ind_dim in range(len(dims)):
            image = np.fft.fft(image,axis=dims[ind_dim])
    else:
        for ind_dim in range(len(dims)):
            image = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image,axes=dims[ind_dim]),axis=dims[ind_dim]),axes=dims[ind_dim])
    
    image = image*scale_fctr
    
    return image

## --------- Fn_k2x ---------- ##
def Fn_k2x_numpy(data, dims, dont_shift=False):
    
    scale_fctr    = np.sqrt(np.prod(size_dims_simp(data, dims)))
    
    if dont_shift:
        for ind_dim in range(len(dims)):
            data = np.fft.ifft(data,axis=dims[ind_dim])
    else:
        for ind_dim in range(len(dims)):
            data = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(data,axes=dims[ind_dim]),axis=dims[ind_dim]),axes=dims[ind_dim])
    
    data = data*scale_fctr
    
    return data

# =============================================================================
# Torch FFT/IFFT
# =============================================================================

## --------- Fn_x2k ---------- ##
def Fn_x2k_torch(image, dims, dont_shift=False):
    
    scale_fctr    = 1/np.sqrt(np.prod(size_dims_simp(image, dims)))
    
    if dont_shift:
        for ind_dim in range(len(dims)):
            image = torch.fft.fft(image,dim=dims[ind_dim])
    else:
        for ind_dim in range(len(dims)):
            image = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(image,dim=dims[ind_dim]),dim=dims[ind_dim]),dim=dims[ind_dim])
    
    image = image*scale_fctr
    
    return image

## --------- Fn_k2x ---------- ##
def Fn_k2x_torch(data, dims, dont_shift=False):
    
    scale_fctr    = np.sqrt(np.prod(size_dims_simp(data, dims)))
    
    if dont_shift:
        for ind_dim in range(len(dims)):
            data = torch.fft.ifft(data,dim=dims[ind_dim])
    else:
        for ind_dim in range(len(dims)):
            data = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(data,dim=dims[ind_dim]),dim=dims[ind_dim]),dim=dims[ind_dim])
    
    data = data*scale_fctr
    
    return data



# =============================================================================
# cenInd
# =============================================================================
def cenInd(Nd,Cd):
    ind = np.int_(np.arange(np.floor(Nd/2)+1+np.ceil(-Cd/2),np.floor(Nd/2)+np.ceil(Cd/2)+1,1))
    ind = ind - 1
    
    return ind

'''        
fig = plt.figure
plt.imshow(abs(image_tmp_k.numpy()),cmap='gray',vmin=0,vmax=1e-1)
plt.colorbar()
plt.show
'''
