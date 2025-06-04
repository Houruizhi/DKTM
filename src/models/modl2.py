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

from ..fastmri.normModule import NormUnet
from ..fastmri.normDnCNN import NormDnCNN
from ..fastmri.sensitivity import SensitivityModel

from ..fastmri.fft import fft2c, ifft2c
from ..fastmri.coil_combine import sense_expand, sense_reduce

class Modl(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 10,
        sens_chans: int = 64,
        sens_pools: int = 4,
        chans: int = 32,
        norm: bool = True,
        norm_layer: str = 'bn',
        mode: str = 'single_coil',
        model_name='dncnn'
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
            if model_name == 'dncnn':
                self.sens_net = SensitivityCNNModel(chans)
            else:
                self.sens_net = SensitivityModel(sens_chans, sens_pools)

        if model_name == 'dncnn':
            self.models = nn.ModuleList(
                [NormDnCNN(chans, 2, 2, if_norm=norm, norm_layer=norm_layer, if_residual=True) for _ in range(num_cascades)]
            )
        elif model_name == 'unet':
            self.models = nn.ModuleList(
                [NormUnet(chans, residual=True) for _ in range(num_cascades)]
            )
            
        self.dc_weights = nn.Parameter(torch.ones(num_cascades))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == 'multi_coil':
            sens_maps = self.sens_net(masked_kspace, mask)

        kspace_pred = masked_kspace.clone()
        x = ifft2c(kspace_pred)

        for step, model in enumerate(self.models):
            if self.mode == 'multi_coil':
                x = sense_reduce(x, sens_maps)
                
            u = model(x)
            
            if self.mode == 'multi_coil':
                u = sense_expand(u, sens_maps)
                
            x = self.dc_weights[step]*fft2c(u) + masked_kspace
            x = ifft2c(x / (self.dc_weights[step] + mask))
        
        if self.mode == 'multi_coil':
            return x, sens_maps
        else:
            return x
