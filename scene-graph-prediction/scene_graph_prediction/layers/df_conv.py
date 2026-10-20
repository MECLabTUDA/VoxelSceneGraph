# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""
from typing import Sequence

import torch
from torch import nn

from .dcn import ModulatedDeformConv, DeformConv

_SIZE_T = int | Sequence[int]


class DFConv(nn.Module):
    """Deformable convolutional layer for 2D and 3D."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_dim: int,
            with_modulated_dcn: bool = True,
            kernel_size: _SIZE_T = 3,
            stride: int = 1,
            groups: int = 1,
            dilation: _SIZE_T = 1,
            deformable_groups: int = 1,
            bias: bool = False):
        super().__init__()

        assert n_dim in [2, 3], "Only 2D and 3D are supported"
        self.n_dim = n_dim

        if isinstance(kernel_size, Sequence):
            assert len(kernel_size) == 2
            assert isinstance(stride, Sequence)
            assert len(stride) == 2
            assert isinstance(dilation, Sequence)
            assert len(dilation) == 2

            padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size

        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv

        if n_dim == 2:
            self.offset = torch.nn.Conv2d(
                in_channels,
                deformable_groups * offset_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=1,
                dilation=dilation
            )
        else:
            self.offset = torch.nn.Conv3d(
                in_channels,
                deformable_groups * offset_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=1,
                dilation=dilation
            )

        nn.init.kaiming_uniform_(self.offset.weight, a=1)
        torch.nn.init.constant_(self.offset.bias, 0.)

        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )

        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.with_modulated_dcn:
            offset = self.offset(x)
            return self.conv(x, offset)

        offset_mask = self.offset(x)
        offset = offset_mask[:, :18, ...]
        mask = offset_mask[:, -9:, ...].sigmoid()
        return self.conv(x, offset, mask)
