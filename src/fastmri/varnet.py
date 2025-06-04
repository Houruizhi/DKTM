"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft import fft2c, ifft2c
from .coil_combine import sense_expand, sense_reduce, rss_complex
from .normUnet import NormUnet

def mask_center(x: torch.Tensor, pad) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[...,pad[0]:pad[1],:] = x[...,pad[0]:pad[1],:]

    return mask

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        model
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.norm_unet = model

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1e-6 + rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        num_low_freqs = 10
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        masked_kspace = mask_center(masked_kspace, [pad, pad+num_low_freqs])

        # convert to image space
        x = ifft2c(masked_kspace)

        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x

class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mode='multi_coil'
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
        if mode == 'multi_coil':
            self.sens_net = SensitivityModel(NormUnet(sens_chans, sens_pools, residual=False))

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, residual=False), mode=mode) for _ in range(num_cascades)]
        )
        self.mode = mode
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, see_features=False) -> torch.Tensor:
        if self.mode == 'multi_coil':
            sens_maps = self.sens_net(masked_kspace, mask)
        else:
            sens_maps = None

        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        if see_features:
            return ifft2c(kspace_pred), kspace_pred, sens_maps
        else:
            return kspace_pred

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, mode):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.mode = mode

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = ifft2c(current_kspace)
        if self.mode == 'multi_coil':
            model_term = sense_reduce(model_term, sens_maps, COIL_DIM=1)

        model_term = self.model(model_term)

        if self.mode == 'multi_coil':
            model_term = sense_expand(model_term, sens_maps)

        model_term = fft2c(model_term)
        return current_kspace - soft_dc - model_term
