# Ultralytics YOLO, AGPL-3.0 license
"""Convolution modules."""

import math

import torch
import torch.nn as nn


def autopad(
        k: int | tuple[int, ...],
        p: int | tuple[int, ...] | None = None,
        dilation: int = 1
):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if dilation > 1:
        k = dilation * (k - 1) + 1 if isinstance(k, int) else [dilation * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding: int | tuple[int, ...] | None = None,
            groups: int = 1,
            dilation: int = 1,
            act: bool | nn.Module = True
    ):
        """Initialize Conv layer with given arguments including activation."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim
        if n_dim == 2:
            self.conv = nn.Conv2d(
                in_channels, out_channels, k, stride, autopad(k, padding, dilation),
                groups=groups, dilation=dilation, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, k, stride, autopad(k, padding, dilation),
                groups=groups, dilation=dilation, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding: int | tuple[int, ...] | None = None,
            groups: int = 1,
            dilation: int = 1,
            act: bool | nn.Module = True
    ):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(in_channels, out_channels, k, stride, padding, groups=groups, dilation=dilation, act=act)
        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.cv2 = conv_cls(
            in_channels, out_channels, 1, stride, autopad(1, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 1,
            act: bool | nn.Module = nn.ReLU()
    ):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.n_dim = n_dim
        self.conv1 = Conv(n_dim, in_channels, out_channels, 1, act=False)
        self.conv2 = DWConv(n_dim, in_channels, out_channels, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            dilation: int = 1,
            act: bool | nn.Module = True
    ):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(
            n_dim, in_channels, out_channels, k,
            stride=stride, groups=math.gcd(in_channels, out_channels), dilation=dilation, act=act
        )


class DWConvTranspose(nn.Module):
    """Depth-wise transpose convolution."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding1: int = 0,
            padding2: int = 0
    ):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim
        if n_dim == 2:
            self.transpose = nn.ConvTranspose2d(
                in_channels, out_channels, k, stride, padding1, padding2, groups=math.gcd(in_channels, out_channels)
            )
        else:
            self.transpose = nn.ConvTranspose3d(
                in_channels, out_channels, k, stride, padding1, padding2, groups=math.gcd(in_channels, out_channels)
            )

    def forward(self, x):
        return self.transpose(x)


class ConvTranspose(nn.Module):
    """Convolution transpose layer."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding: int = 0,
            bn: bool = True,
            act: bool | nn.Module = nn.ReLU()
    ):
        """Initialize ConvTranspose layer with batch normalization and activation function."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim
        if n_dim == 2:
            self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, k, stride, padding, bias=not bn)
            self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        else:
            self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, k, stride, padding, bias=not bn)
            self.bn = nn.BatchNorm3d(out_channels) if bn else nn.Identity()
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding: int | tuple[int, ...] | None = None,
            groups: int = 1,
            act: bool | nn.Module = True
    ):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        assert n_dim in [2, 3]
        self.n_dim = n_dim
        super().__init__()
        self.conv = Conv(
            n_dim, in_channels * 2 ** n_dim, out_channels, k,
            stride=stride, padding=padding, groups=groups, act=act
        )

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        if self.n_dim == 2:
            return self.conv(
                torch.cat((
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ), 1)
            )
        return self.conv(
            torch.cat((
                x[..., ::2, ::2, ::2],
                x[..., 1::2, ::2, ::2],
                x[..., ::2, 1::2, ::2],
                x[..., 1::2, 1::2, ::2],
                x[..., ::2, ::2, 1::2],
                x[..., 1::2, ::2, 1::2],
                x[..., ::2, 1::2, 1::2],
                x[..., 1::2, 1::2, 1::2],
            ), 1)
        )


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            groups: int = 1,
            act: bool | nn.Module = True
    ):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        self.n_dim = n_dim
        c_ = out_channels // 2  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, c_, k, stride=stride, groups=groups, act=act)
        self.cv2 = Conv(n_dim, c_, c_, 5, stride=1, groups=c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int | tuple[int, ...] = 3,
            stride=1,
            padding: int | tuple[int, ...] | None = None,
            groups: int = 1,
            act: bool | nn.Module = True,
            bn: bool = False
    ):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and padding == 1
        self.n_dim = n_dim
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if n_dim == 2:
            self.bn = nn.BatchNorm2d(num_features=in_channels) \
                if bn and out_channels == in_channels and stride == 1 \
                else None
        else:
            self.bn = nn.BatchNorm3d(num_features=in_channels) \
                if bn and out_channels == in_channels and stride == 1 \
                else None

        self.conv1 = Conv(
            n_dim, in_channels, out_channels, k,
            stride=stride, padding=padding, groups=groups, act=False
        )
        self.conv2 = Conv(
            n_dim, in_channels, out_channels, 1,
            stride=stride, padding=(padding - k // 2), groups=groups, act=False
        )

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, n_dim: int, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        assert n_dim in [2, 3]
        self.n_dim = n_dim

        if n_dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        else:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Conv3d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, n_dim: int, kernel_size: int = 7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert n_dim in [2, 3]
        self.n_dim = n_dim

        assert kernel_size in [1, 3, 7], "kernel size must be 1, 3, or 7"
        padding = kernel_size // 2
        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.cv1 = conv_cls(2, 1, kernel_size, padding=padding, bias=False)

        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, n_dim: int, in_channels: int, kernel_size: int = 7):
        """Initialize CBAM with given input channel (in_channels) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(n_dim, in_channels)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension: int = 1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.dim)
