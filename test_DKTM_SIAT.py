import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import os
import cv2
import numpy as np
import copy
from scipy.io import loadmat, savemat

from src.fastmri.transforms import to_tensor
from src.fastmri.fft import fft2c, ifft2c, fftshift, fft2
from src.utils import *
from src.utils_TRPA import recon_TRPA
from src.utils_selfsupervise import *

from src.models.SSUnrolling import SSUnrolling
from test_SIAT_ensembleSSDU import train_multimodels
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    mask_types = [
        'spiral_010',
        # 'spiral_020',
        # 'spiral_030',
        # 'radial_010',
        # 'radial_020',
        # 'radial_030',
        # 'cart_010',
        # 'cart_020',
        # 'cart_030',
        # 'random_010',
        # 'random_020',
        # 'random_030',
    ]
    data_root = f'/home/rzhou/Datasets/MRIDatasets/SIAT/test/'
    save_image_root = f'/home/rzhou/ALLMRICodes/results_MRI/results_SIAT/noise_free/DKTM'
    
    models_all = nn.ModuleList()
    for _ in range(5):
        models_all.append(
            SSUnrolling(
                num_cascades=5,
                num_of_layers=12,
                chans=64,
                random_aug=False
            )
            )
    models_all = models_all.cuda()

    files = os.listdir(data_root)
    files.sort()
    for mask_type in mask_types:
        save_image_path = os.path.join(save_image_root, mask_type)
        os.makedirs(save_image_path, exist_ok=True)

        mask = loadmat(f'/home/rzhou/Datasets/mri_masks/{mask_type}.mat')['mask']
        mask = to_tensor(mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
        for file_ in files:
            target_np = loadmat(os.path.join(data_root, file_))['Img']
            target_tensor, target_rss = process_MRI_data(target_np, None)

            data_kspace = fft2c(target_tensor).cuda()
            data_kspace = data_kspace * mask
            chps = torch.load(os.path.join('checkpoints/SIAT/SSEnsemble', mask_type, file_.split('.')[0]+'.pth'))
            models_all.load_state_dict(chps)
            # models_all = train_multimodels(data_kspace, mask, target_tensor)
            time0 = time.time()
            rec_im, loss_record, time1, scorenet = score_distilation(models_all, data_kspace, mask, target_tensor=target_tensor, transfer_learning=True, NUM_EPOCHS=1000)
            print('recon+training: ', time.time() - time0)
            PSNR, _, _ = compare_psnr_ssim_tensor(rec_im, target_tensor)
            return
            
            print(mask_type, file_, PSNR, time1)
            rec_im = torch.view_as_complex(rec_im).squeeze().detach().cpu().numpy()
            savemat(os.path.join(save_image_path, file_.split('.')[0]+'.mat'), {'rec_im': rec_im})
if __name__ == '__main__':
    main()