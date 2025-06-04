"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from os import umask
from secrets import choice
from typing import List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fastmri.sensitivity import SensitivityModel
from ..fastmri.normModule import NormDnCNN, NormUnet

from ..fastmri.fft import fft2c, ifft2c
from ..fastmri.math import complex_abs
from ..fastmri.coil_combine import rss, rss_complex, sense_expand, sense_reduce
from ..fastmri.math import complex_mul, complex_conj
from ..fastmri.transforms import mask_center

from .condDncnn import DnCNN as CDncnn

def augment_tensor(img, mode, reverse=False):
    '''
    img: (n,c,h,w)
    '''
    if not reverse:
        if mode == 0:
            return img
        elif mode == 1:
            return img.permute(0,1,3,2)
        elif mode == 2:
            return img.flip(dims=[-1])
        elif mode == 3:
            return img.flip(dims=[-2])
        elif mode == 4:
            return img.permute(0,1,3,2).flip(dims=[-1])
        elif mode == 5:
            return img.permute(0,1,3,2).flip(dims=[-2])
        elif mode == 6:
            return img.flip(dims=[-1]).flip(dims=[-2])
        elif mode == 7:
            return img.permute(0,1,3,2).flip(dims=[-1]).flip(dims=[-2])
    else:
        if mode == 0:
            return img
        elif mode == 1:
            return img.permute(0,1,3,2)
        elif mode == 2:
            return img.flip(dims=[-1])
        elif mode == 3:
            return img.flip(dims=[-2])
        elif mode == 4:
            return img.flip(dims=[-1]).permute(0,1,3,2)
        elif mode == 5:
            return img.flip(dims=[-2]).permute(0,1,3,2)
        elif mode == 6:
            return img.flip(dims=[-2]).flip(dims=[-1])
        elif mode == 7:
            return img.flip(dims=[-2]).flip(dims=[-1]).permute(0,1,3,2)
        
def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return complex_mul(x, sens_maps)

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul(x, complex_conj(sens_maps)).sum(
        dim=1, keepdim=True
    )
    
class NormModule(nn.Module):
    def __init__(self, num_of_layers, chans, num_cascades, input_channel=1):
        super().__init__()
        self.model = CDncnn(2*input_channel,2*input_channel,num_of_layers=num_of_layers,num_features=chans,num_classes=num_cascades,residual=False)
        # self.model = CondRefineNetDilated(num_cascades, in_channels=2, ngf=chans//4)
        # self.model = CUnet(2,2,chans=chans//2,num_classes=num_cascades)
        
    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        # print(x.shape)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, t, aug=False) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        if aug:
            res = 0
            for _ in range(3):
                aug_id = np.random.randint(0,8)
                inputs = augment_tensor(x, aug_id)
                output = self.model(inputs, torch.tensor(t).int().cuda())
                res = res + augment_tensor(output, aug_id, reverse=True)
            x = res / 3
        else:
            x = self.model(x, torch.tensor(t).int().cuda())
            
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x

class SSUnrolling(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_of_layers=12,
        num_cascades=5,
        chans=32,
        input_channel=1,
        multi_coil=False,
        random_aug=False
    ):
        super().__init__()

        self.model = NormModule(num_of_layers, chans, num_cascades)

        self.num_cascades = num_cascades
        
        self.random_aug = random_aug
        self.multi_coil = multi_coil
        if self.multi_coil:
            sense_model = NormDnCNN(16, num_of_layers=5, norm_layer='bn', residual=False)
            self.sens_net = SensitivityModel(sense_model)
            
    def forward(self, masked_kspace, mask, see_features=False, random_aug=False) -> torch.Tensor:
        if self.multi_coil:
            sens_maps = self.sens_net(masked_kspace, mask)
        
        x0 = ifft2c(masked_kspace).clone()

        if see_features:
            middle_res = []

        x_t = x0
        diffusion_times = list(range(0, self.num_cascades))[::-1]
        for t in diffusion_times:

            if self.multi_coil:
                x_t = sens_reduce(x_t, sens_maps)
            
            x_t = self.model(x_t, t, self.random_aug)

            if self.multi_coil:
                x_t = sens_expand(x_t, sens_maps)
                
            x_t = ifft2c((1-mask)*fft2c(x_t) + mask*masked_kspace)
            
            if see_features:
                middle_res.append(x_t)
                
        if self.multi_coil:
            if see_features:
                return x_t, sens_maps, middle_res
            else:
                return x_t, sens_maps
        else:
            if see_features:
                return x_t, middle_res
            else:
                return x_t