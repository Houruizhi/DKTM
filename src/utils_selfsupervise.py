import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from .fastmri.coil_combine import rss

from .models.SSUnrolling import SSUnrolling
from .models.condrefinenet import CondRefineNetDilated

def get_random_patch(tensor, patch_size = 96):
    n,c,h,w = tensor.shape
    h_i = torch.randint(h-patch_size, [1]).item()
    w_i = torch.randint(w-patch_size, [1]).item()
    return tensor[...,h_i:h_i+patch_size,w_i:w_i+patch_size]

def get_kernel(h, w, s=3):
    X = torch.arange(0, h) - h // 2
    Y = torch.arange(0, w) - w // 2
    x, y = torch.meshgrid(X, Y)
    gauss_1 = torch.exp(- (x  ** 2 + y ** 2) / (2*s**2))
    gauss_1 = gauss_1 / gauss_1.sum()
    return gauss_1.view(1,1,h,w)

def get_kernel_1d(d, s=3):
    x = torch.arange(0, d) - d // 2
    gauss_1 = torch.exp(- (x  ** 2) / (2*s**2))
    gauss_1 = gauss_1 / gauss_1.sum()
    return gauss_1

def aug_contrast(patch):
    patch_abs = rss(patch, 1).unsqueeze(1)
    patch_angle = patch / (1e-6 + patch_abs)
    patch_abs_enhance = patch_abs.pow(0.5 + np.random.rand(1).item())
    return patch_angle * patch_abs_enhance
    
def conv_image(patch, kszie_h, kszie_w, sigma_h, sigma_w):
    c = patch.shape[1]
    kernel_h = get_kernel_1d(kszie_h, sigma_h).view(1,1,kszie_h,1).repeat(c,1,1,1).cuda().float()
    kernel_w = get_kernel_1d(kszie_w, sigma_w).view(1,1,1,kszie_w).repeat(c,1,1,1).cuda().float()

    patch_conv = F.pad(patch, pad=(kszie_w//2,kszie_w//2,kszie_h//2,kszie_h//2))
    patch_conv = F.conv2d(patch_conv, kernel_w, groups=c)
    patch_conv = F.conv2d(patch_conv, kernel_h, groups=c)
    return patch_conv

def aug_enhance(patch):
    kernel_size = [3,5,7]
    kszie_h = np.random.choice(kernel_size)
    kszie_w = np.random.choice(kernel_size)
    
    sigma_h = 0.5 + 10*np.random.rand(1)
    sigma_w = 0.5 + 10*np.random.rand(1)
    
    patch_conv = conv_image(patch, kszie_h, kszie_w, sigma_h, sigma_w)
    
    m_value = patch.abs().max()
    
    patch_detail = patch - patch_conv
    patch_detail = (1 + 3*np.random.rand(1).item()) * patch_detail
    patch = patch_conv + patch_detail
    
    return patch.clamp(-m_value, m_value)
    
def augment_data(patch):        
    if torch.rand(1) > 0.5:
        patch = patch.permute(0,1,3,2)
    if torch.rand(1) > 0.5:
        patch = patch.flip([3])
    if torch.rand(1) > 0.5:
        patch = patch.flip([2])
    if torch.rand(1) > 0.8:
        patch = aug_contrast(patch)
    if torch.rand(1) > 0.8:
        patch = aug_enhance(patch)
    return patch

def dsm_loss(scorenet, image_target, max_sigma=1/3.):
    samples = image_target.cuda()
    # sigma_min default: 0.001
    used_sigmas = torch.exp(torch.rand(samples.shape[0],1,1,1)*(math.log(max_sigma) - math.log(0.01)) + math.log(0.01))
    used_sigmas = used_sigmas.to(samples.device)
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    inputs = perturbed_samples
    scores = scorenet(inputs, used_sigmas)

    target = - noise / (used_sigmas ** 2)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(-3,-2,-1)) * used_sigmas.squeeze() ** 2
    # with torch.no_grad():
    #     denoised = perturbed_samples + used_sigmas**2*scores
    return loss.sum(dim=0)/samples.shape[0]

def mse_loss(scorenet, image_target, max_sigma=1/3.):
    samples = image_target.cuda()
    # sigma_min default: 0.001
    used_sigmas = torch.exp(torch.rand(samples.shape[0],1,1,1)*(math.log(max_sigma) - math.log(0.01)) + math.log(0.01))
    used_sigmas = used_sigmas.to(samples.device)
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    scores = scorenet(perturbed_samples, used_sigmas)
    return ((scores - noise) ** 2).sum()/samples.shape[0]

def mae_loss(scorenet, image_target, max_sigma=1/3.):
    samples = image_target.cuda()
    used_sigmas = 0.01*torch.ones(samples.shape[0],1,1,1).to(samples.device)
    
    mask = (torch.rand_like(image_target) < 0.8).byte()
    predict = scorenet(image_target*mask, used_sigmas)

    return ((1-mask)*(predict-image_target)).pow(2).sum()/samples.shape[0]

def uniform_selection_(input_mask, rho=0.5, small_acs_block=(4, 4)):
    center_kx = input_mask.shape[-3]//2
    center_ky = input_mask.shape[-2]//2

    loss_mask = torch.rand(*input_mask.shape).to(input_mask.device) < rho
    loss_mask[...,center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2, :] = 1
    return input_mask * loss_mask

def recon_image(scorenet, data_kspace, mask, N=1):
    scorenet.eval()
    rec_im = torch.zeros_like(data_kspace).squeeze(1)
    from .utils_TRPA import recon_TRPA, denoiser_dsm
    for _ in range(N):
        rec_im_, psnrs = recon_TRPA(
            lambda x,y,z:denoiser_dsm(scorenet,x,y,z), 
            data_kspace.squeeze(1), 
            mask.squeeze(1),
            sense_map=None,
            target_tensor=None,
            gamma=1.07,
            lam=3e-4,
            rho=0.0033,
            max_iter=115,
            eps=7e-9,
            step_lr=1,
            repeat=1,
            normlize_input=False,
            verbose=False,
            crop_win=256)
        rec_im += rec_im_
    return rec_im.unsqueeze(1) / N

import time
import copy
from .utils import process_weights, compare_psnr_ssim_tensor
def score_distilation(models_all, data_kspace, mask, target_tensor=None, transfer_learning=True, NUM_EPOCHS=600, loss_type=0):
    time0 = time.time()
    score_net, loss_record = score_distilation_training(models_all, data_kspace, mask, target_tensor=target_tensor, transfer_learning=transfer_learning, NUM_EPOCHS=NUM_EPOCHS, loss_type=loss_type)
    time1 = time.time() - time0
    print('score net training time:', time.time() - time0)
    
    rec_im = recon_image(score_net, data_kspace, mask, 5)
    return rec_im, loss_record, time1, score_net

def score_distilation_training(models_all, data_kspace, mask, target_tensor=None, F=128, NUM_EPOCHS=1000, transfer_learning=True, loss_type=0):
    
    models_all.eval()
    
    if transfer_learning:
        scorenet = CondRefineNetDilated(3,3,F).cuda()
        scorenet.load_state_dict(process_weights(torch.load(f'/home/rzhou/Checkpoints/scorenet_div2k_F{F}.pth')))
        scorenet.train()
        scorenet.begin_conv = nn.Conv2d(2, F, 3, stride=1, padding=1).cuda()
        
        scorenet.end_conv = nn.Conv2d(F, 2, 3, stride=1, padding=1).cuda()
    else:
        scorenet = CondRefineNetDilated(2,2,F).cuda()
        
    optimizer = torch.optim.Adam(
        scorenet.parameters(), 
        lr=5e-3, 
        weight_decay=0.000, 
        betas=(0.9, 0.999), 
        amsgrad=False
    )

    loss_record = []
    model_EMA = copy.deepcopy(scorenet)
    batch_size = 4
    loss_sum = 0
    for i in range(NUM_EPOCHS):
        patches = []
        for _ in range(batch_size):
            with torch.no_grad():
                M = uniform_selection_(mask, 0.8, (8,8))
                if len(models_all) > 1:
                    recon_tensor = models_all[np.random.choice(range(len(models_all)))](M*data_kspace, M)
                else:
                    recon_tensor = models_all[0](M*data_kspace, M)
            recon_tensor = recon_tensor.squeeze(1).permute(0,3,1,2)
            patch = get_random_patch(recon_tensor, 128)
            # patch = recon_tensor
            patch = augment_data(patch)
            patches.append(patch)
            
        batch_data = torch.cat(patches, dim=0)
        
        scorenet.train()
        scorenet.zero_grad()
        optimizer.zero_grad()

        if loss_type == 0:
            loss = dsm_loss(scorenet, batch_data) 
        elif loss_type == 1:
            loss = mse_loss(scorenet, batch_data) 
        elif loss_type == 2:
            loss = mae_loss(scorenet, batch_data) 
        loss.backward()
        loss_record.append(loss.item())

        loss_sum += loss.item()
        optimizer.step()

        if transfer_learning:
            # best: 0.995
            alpha = min(1 - 1 / (i+1), 0.995)
            for ema_param, param in zip(model_EMA.parameters(), scorenet.parameters()):
                ema_param.data = (alpha * ema_param.data) + (1-alpha)*param.data
        else:
            model_EMA = scorenet

        if (target_tensor is not None) and (i % 100 == 99):
            rec_im = recon_image(scorenet, data_kspace, mask, 1)
            
            PSNR, _, _ = compare_psnr_ssim_tensor(rec_im, target_tensor)
            print(f'Iter: {i}, PSNR:{PSNR}, Loss: {loss_sum/(i+1)}')
            
    return model_EMA, loss_record