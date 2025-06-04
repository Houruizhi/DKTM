"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from os import umask
from secrets import choice
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fastmri.normModule import NormDnCNN, NormUnet
from ..fastmri.sensitivity import SensitivityModel

from ..fastmri.fft import fft2c, ifft2c
from ..fastmri.math import complex_abs
from ..fastmri.coil_combine import rss, rss_complex
from ..fastmri.math import complex_mul, complex_conj
from ..fastmri.transforms import mask_center


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return complex_mul(x, sens_maps)

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul(x, complex_conj(sens_maps)).sum(
        dim=1, keepdim=True
    )

class Modl(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades=10,
        chans=32,
        num_of_layers=17,
        mode='multi_coil',
        module='dncnn',
        norm_layer='bn',
        residual=True,
        sense_chans=64,
        if_norm=True
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.mode = mode
        if self.mode == 'multi_coil':
            if module == 'dncnn':
                sense_model = NormDnCNN(sense_chans, num_of_layers=5, norm_layer=norm_layer, residual=residual, if_norm=if_norm)
            elif module == 'unet':
                sense_model = NormUnet(sense_chans, residual=residual)
            else:
                raise NotImplementedError

            self.sens_net = SensitivityModel(sense_model)

        if module == 'dncnn':
            self.models = nn.ModuleList(
                [NormDnCNN(chans, num_of_layers=num_of_layers, norm_layer=norm_layer, residual=residual) for _ in range(num_cascades)]
            )
        elif module == 'unet':
            self.models = nn.ModuleList(
                [NormUnet(chans, residual=residual) for _ in range(num_cascades)]
            )
        else:
            raise NotImplementedError

        self.dc_weights = nn.Parameter(1e-3*torch.ones(num_cascades))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, see_features=False, x0=None) -> torch.Tensor:
        if self.mode == 'multi_coil':
            sens_maps = self.sens_net(masked_kspace, mask)

        if x0 is None:
            x = ifft2c(masked_kspace)
        else:
            x = x0

        if see_features:
            middle_res = []
            
        for step, model in enumerate(self.models):
            if self.mode == 'multi_coil':
                x = sens_reduce(x, sens_maps)
            
            u = model(x)
            if self.mode == 'multi_coil':
                u = sens_expand(u, sens_maps)
            x = self.dc_weights[step]*fft2c(u) + masked_kspace
            x = ifft2c(x / (self.dc_weights[step] + mask))

            if see_features:
                middle_res.append(x)
                
        if see_features:
            if self.mode == 'multi_coil':
                return x, middle_res, sens_maps
            else:
                return x, middle_res
        else:
            return x
