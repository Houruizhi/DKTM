import math
import numpy as np

import torch
import torch.nn.functional as F

def image2patch(image, patch_size):
    n,c,h,w = image.shape
    offset = patch_size - 1
    image_unfold = F.unfold(image, kernel_size=patch_size).reshape(n, c, patch_size**2, h-offset, w-offset)
    return image_unfold.reshape(n*c*patch_size**2,-1)

def patch2image(image_unfold, W_unfold, patch_size, image_shape):
    n,c,h,w = image_shape
    offset = patch_size - 1
    image_unfold = image_unfold.reshape(n,c,patch_size,patch_size,h-offset,w-offset)
    W_unfold = W_unfold.reshape(n,c,patch_size,patch_size,h-offset,w-offset)
    image_recon = torch.zeros(*image_shape).to(image_unfold.device).type(image_unfold.dtype)
    image_w = torch.zeros(*image_shape).to(image_unfold.device).type(image_unfold.dtype)
    k=0
    for i in range(patch_size):
        for j in range(patch_size):
            image_recon[:,:,i:h+i-offset,j:w+j-offset] += image_unfold[:,:,i,j,:,:]
            image_w[:,:,i:h+i-offset,j:w+j-offset] += W_unfold[:,:,i,j,:,:]
            k += 1
    return image_recon, image_w

def crop_patch(image, location, patch_size=3):
    x, y = location
    return image[...,x:x+patch_size,y:y+patch_size]

def find_sim_patches(similarity, number=8):
    L = similarity.shape[-1]
    assert number <= L
    values, indices = similarity.sort()
    return indices[:number]

def set_index(image_shape, search_win, patch_size):
    n,c,h,w = image_shape
    loc2index = np.arange(0,(h-patch_size+1)*(w-patch_size+1)).reshape(h-patch_size+1,w-patch_size+1)
    index2loc = {}
    search_indexes = {}
    k = 0
    for i in range(h-patch_size+1):
        for j in range(w-patch_size+1):
            top = max(0, i-search_win)
            bottom = min(h, i+search_win+1)
            left = max(0, j-search_win)
            right = min(w, j+search_win+1)
            index2loc[loc2index[i,j]] = (i, j)
            search_indexes[loc2index[i,j]] = loc2index[...,top:bottom,left:right].reshape(-1)
            k += 1
    return search_indexes, index2loc, loc2index

def patch_similarity(image_patch, image_unfold, normalize=False):
    '''
    image_patch: c
    image_unfold: N, c
    '''
    if normalize:
        image_patch = F.normalize(image_patch, dim=-1)
        image_unfold = F.normalize(image_unfold, dim=-1)
    return (image_patch.abs() - image_unfold.abs()).abs().sum(0)

def block_matching(image_unfold, search_indexes, similar_number=16):
    c, L = image_unfold.shape
    similar_indexes = {}
    for i in range(0, L):
        patch = image_unfold[...,i].unsqueeze(-1)
        search_index = search_indexes[i]
        similarity = patch_similarity(patch, image_unfold[...,search_index])
        sim_index = find_sim_patches(similarity, similar_number).cpu()
        similar_indexes[i] = search_index[sim_index]
    return similar_indexes