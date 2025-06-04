import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import math
import glob
from tqdm import tqdm

from src.fastmri.math import complex_abs
from src.fastmri.transforms import to_tensor
from src.fastmri.coil_combine import rss_complex, rss
from src.fastmri.subsample import RandomMaskFunc
from src.fastmri.fft import fft2c, ifft2c, fftshift

from src.utils import *
from src.utils_selfsupervise import uniform_selection_
from src.metrics import psnr, ssim
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from src.fastmri.subsample import RandomMaskFunc
from src.tvcs.tv_op import nabla_x, nabla_y

from src.models.SSUnrolling import SSUnrolling
# from src.models.AUnrolling import SSUnrolling
# from src.models.MambaUnrolling import SSUnrolling

import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

COIL_DIM = 0

def get_kernel(h, w, s=3):
    X = torch.arange(0, h) - h // 2
    Y = torch.arange(0, w) - w // 2
    x, y = torch.meshgrid(X, Y)
    gauss_1 = torch.exp(- (x  ** 2 + y ** 2) / (2*s**2))
    gauss_1 = gauss_1.clamp(0.01)
    return gauss_1.view(1,1,h,w)

def get_map(h, w, c1=0.5, c2=0.5, s=3):
    X = torch.arange(0, h) - int(h * c1)
    Y = torch.arange(0, w) - int(w * c2)
    x, y = torch.meshgrid(X, Y)
    gauss_1 = (x ** 2 + y ** 2).clamp(0.01)**s
    return gauss_1.view(1,1,h,w)

def compute_loss(model, M, data_kspace, mask, weight_map, tv_weight=0):
    x = model(M * data_kspace, M)
    residual = mask*(fft2c(x) - data_kspace)
    loss = (weight_map*residual).pow(2).sum(-1).mean()
    x_spatial = x.squeeze(1).permute(0,3,1,2)
    loss_tv = (nabla_x(x_spatial).abs().clamp(0.01, 0.5) + nabla_y(x_spatial).abs().clamp(0.01, 0.5)).mean()
    return loss + tv_weight*1e2*loss_tv

def self_supervised_training(model, data_kspace, mask, lr=1e-3, target_tensor=None, weight_map=None, tv_weight=0):
    
    MILESTONE = 200
    NUM_EPOCHS = 1000
    LEARNING_RATE_DECAY = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_record = []
    model_EMA = copy.deepcopy(model)
    for i in range(NUM_EPOCHS):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        loss_all = 0
        
        M = uniform_selection_(mask, 0.8, (8,8))
        
        loss = compute_loss(model, M, data_kspace, mask, weight_map, tv_weight)

        loss_all = loss_all + loss
        
        loss_all.backward()
        optimizer.step()
        
        alpha = min(1 - 1 / (i+1), 0.6)
        for ema_param, param in zip(model_EMA.parameters(), model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1-alpha)*param.data
        
        loss_record.append(loss_all.item())
            
        if i % 200 == 199:
            print(i, f'Loss: {loss_all.item()}')
            if (target_tensor is not None):
                model.eval()
                with torch.no_grad():
                    rec_im = model(data_kspace, mask)
                
                PSNR, SSIM, _ = compare_psnr_ssim_tensor(rec_im, target_tensor)

                print(i, f'PSNR : {PSNR}')

    return model_EMA, loss_record

def train_multimodels(data_kspace, mask, target_tensor=None):
    n,c,h,w,_ = data_kspace.shape
    # settings = [
    #     (0.0, 0),
    #     (0.1, 0),
    #     (0.2, 0),
    #     (0.3, 0),
    #     (0.4, 0)
    # ]
    settings = [
        (0.0, 0),
        (0.1, 0),
        (0.2, 0),
        (0.4, 0),
        (0.8, 0)
    ]
    
    models_all = nn.ModuleList()
    for idx, (weight_sigma, tv_weight) in enumerate(settings):
        weight_map = get_map(h, w, 0.5, 0.5, weight_sigma).unsqueeze(-1).cuda()
        model = SSUnrolling(
            num_cascades=5,
            num_of_layers=12,
            chans=64,
            random_aug=False
        )
        model = model.cuda()
        
        time0 = time.time()
        model, loss_record = self_supervised_training(model, data_kspace, mask, target_tensor=target_tensor, weight_map=weight_map, tv_weight=tv_weight)
        print('computing time: ', time.time() - time0)
        models_all.append(model)
        # np.save(f'./results/loss_record_{idx}.npy', loss_record)
        
    if target_tensor is not None:
        rec_im_all = []
        for k in range(len(models_all)):
            rec_im = models_all[k](data_kspace, mask)
            PSNR, _, _ = compare_psnr_ssim_tensor(rec_im, target_tensor)
            print(f'single model {k} PSNR: {PSNR}')
            rec_im_all.append(rec_im)
    
        rec_im_all = torch.cat(rec_im_all, dim=0).mean(0, keepdim=True)
        PSNR, _, rec_im = compare_psnr_ssim_tensor(rec_im_all, target_tensor)
        print(f'ensemble models PSNR: {PSNR}')
    return models_all

def main():
    mask_types = [
        # 'radial_010',
        # 'radial_020',
        # 'radial_030',
        'spiral_010',
        # 'spiral_020',
        # 'spiral_030',
        # 'cart_010',
        # 'cart_020',
        # 'cart_030',
        # 'random_010',
        # 'random_020',
        # 'random_030',
    ]
    data_root = f'/home/rzhou/Datasets/MRIDatasets/SIAT/test/'
    save_model_root = f'checkpoints/SSEnsemble'
    
    files = os.listdir(data_root)
    files.sort()
    for mask_type in mask_types:
        save_model_path = os.path.join(save_model_root, mask_type)
        os.makedirs(save_model_path, exist_ok=True)

        mask = loadmat(f'/home/rzhou/Datasets/mri_masks/{mask_type}.mat')['mask']
        mask = to_tensor(mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
        for file_ in files:
            target_np = loadmat(os.path.join(data_root, file_))['Img']
            target_np = target_np / np.abs(target_np).max()
            target_tensor, target_rss = process_MRI_data(target_np, None)

            data_kspace = fft2c(target_tensor).cuda()
            data_kspace = data_kspace * mask
        
            PSNR, _, rec_im = compare_psnr_ssim_tensor(ifft2c(data_kspace), target_tensor)
            
            print(mask_type, file_)

            model = train_multimodels(data_kspace, mask, target_tensor)
            return
            # torch.save(model.state_dict(), os.path.join(save_model_path, file_.split('.')[0]+'.pth'))   

if __name__ == '__main__':
    main()

'''
0 PSNR single: 23.004910893985517, ensemble: 25.108569463253435, Loss: 147.2166748046875
200 PSNR single: 28.701677539194925, ensemble: 29.62772356196954, Loss: 13.627339363098145
400 PSNR single: 29.598626119477146, ensemble: 30.447896534077312, Loss: 9.721273422241211
600 PSNR single: 29.99219315972928, ensemble: 30.856478097297906, Loss: 7.814594268798828
800 PSNR single: 30.19025745363293, ensemble: 31.044120827871282, Loss: 7.381653308868408
1000 PSNR single: 30.36421559985866, ensemble: 31.209699238035135, Loss: 6.166981220245361
1200 PSNR single: 30.330251464070006, ensemble: 31.35823803065054, Loss: 5.5465922355651855
1400 PSNR single: 30.44518341580275, ensemble: 31.456525554030627, Loss: 5.3465657234191895
1600 PSNR single: 30.42643275394785, ensemble: 31.53434470275095, Loss: 5.258810520172119
1800 PSNR single: 30.55403108065766, ensemble: 31.59176488891077, Loss: 4.529401779174805
2000 PSNR single: 30.647828658032843, ensemble: 31.647603006935302, Loss: 4.467841148376465
2200 PSNR single: 30.54419696644933, ensemble: 31.654333595516174, Loss: 4.070754051208496
2400 PSNR single: 30.542158204081122, ensemble: 31.66884142298343, Loss: 3.7558176517486572
2600 PSNR single: 30.594015495381925, ensemble: 31.715923747662416, Loss: 3.760395050048828
2800 PSNR single: 30.541238601411784, ensemble: 31.706129205332644, Loss: 3.440110683441162
3000 PSNR single: 30.58970863140342, ensemble: 31.732481212505487, Loss: 3.401249647140503
3200 PSNR single: 30.462735084347543, ensemble: 31.6873895761074, Loss: 3.2901992797851562
3400 PSNR single: 30.492209193190504, ensemble: 31.70792610716387, Loss: 3.0951011180877686
3600 PSNR single: 30.571793600036273, ensemble: 31.730893281733398, Loss: 3.1677494049072266
3800 PSNR single: 30.57874950629218, ensemble: 31.737822856845558, Loss: 2.9744131565093994
'''
