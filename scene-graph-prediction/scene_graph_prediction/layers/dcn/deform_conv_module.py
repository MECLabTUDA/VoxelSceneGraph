import math

import torch
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair

from .deform_conv_func import deform_conv, modulated_deform_conv


class DeformConv(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            deformable_groups: int = 1,
            bias: bool = False
    ):
        if bias:
            raise NotImplementedError

        assert in_channels % groups == 0, f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, f'out_channels {out_channels} cannot be divisible by groups {groups}'

        super().__init__()

        self.with_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        # Init weights
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std_v = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std_v, std_v)

    def forward(self, input_tensor: torch.Tensor, offset: torch.Tensor):
        return deform_conv(
            input_tensor, offset, self.weight, self.stride,
            self.padding, self.dilation, self.groups,
            self.deformable_groups
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}, " \
               f"padding={self.padding}, groups={self.groups}, deformable_groups={self.deformable_groups}, " \
               f"bias={self.with_bias})"


class ModulatedDeformConv(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            deformable_groups: int = 1,
            bias: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Init weights
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std_v = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input_tensor: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv(
            input_tensor, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}, " \
               f"padding={self.padding}, groups={self.groups}, deformable_groups={self.deformable_groups}, " \
               f"bias={self.with_bias})"


class ModulatedDeformConvPack(ModulatedDeformConv):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, deformable_groups, bias
        )

        self.conv_offset_mask = torch.nn.Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True
        )

        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_tensor: torch.Tensor, *_) -> torch.Tensor:
        out = self.conv_offset_mask(input_tensor)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            input_tensor, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups
        )
