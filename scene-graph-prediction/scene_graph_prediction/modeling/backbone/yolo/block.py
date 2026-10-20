# Ultralytics YOLO, AGPL-3.0 license
"""Block modules."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, n_dim: int, in_channels: int = 16):
        """Initialize a convolutional layer with a given number of input channels."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim

        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.conv = conv_cls(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(
            x.view(b, 2 * self.n_dim, self.in_channels, a)
            .transpose(2, 1)
            .softmax(1)
        ).view(b, 2 * self.n_dim, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, n_dim: int, in_channels: int, hidden_channels: int = 256, out_channels: int = 32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        # FIXME different k for 3D?
        self.n_dim = n_dim
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, k=3)
        # nn.Upsample(scale_factor=2, mode='nearest')
        conv_trans_cls = nn.ConvTranspose2d if n_dim == 2 else nn.ConvTranspose3d
        self.upsample = conv_trans_cls(hidden_channels, hidden_channels, 2, 2, 0, bias=True)
        self.cv2 = Conv(n_dim, hidden_channels, hidden_channels, k=3)
        self.cv3 = Conv(n_dim, hidden_channels, out_channels)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, n_dim: int, in_channels: int, hidden_channels: int, out_channels: int):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(n_dim, in_channels, hidden_channels, 3, stride=2, act=nn.ReLU())
        self.stem2a = Conv(n_dim, hidden_channels, hidden_channels // 2, 2, stride=1, padding=0, act=nn.ReLU())
        self.stem2b = Conv(n_dim, hidden_channels // 2, hidden_channels, 2, stride=1, padding=0, act=nn.ReLU())
        self.stem3 = Conv(n_dim, hidden_channels * 2, hidden_channels, 3, stride=2, act=nn.ReLU())
        self.stem4 = Conv(n_dim, hidden_channels, out_channels, 1, 1, act=nn.ReLU())

        max_pool_cls = nn.MaxPool2d if n_dim == 2 else nn.MaxPool3d
        self.pool = max_pool_cls(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            k: int = 3,
            n: int = 6,
            lightconv: bool = False,
            shortcut: bool = False,
            act: nn.Module = nn.ReLU()
    ):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(
            block(n_dim, in_channels if i == 0 else hidden_channels, hidden_channels, k=k, act=act)
            for i in range(n)
        )
        self.sc = Conv(n_dim, in_channels + n * hidden_channels, out_channels // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(n_dim, out_channels // 2, out_channels, 1, stride=1, act=act)  # excitation conv
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: tuple[int, ...] = (5, 9, 13)
    ):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        self.cv2 = Conv(n_dim, hidden_channels * (len(k) + 1), out_channels, 1, stride=1)
        max_pool_cls = nn.MaxPool2d if n_dim == 2 else nn.MaxPool3d
        self.m = nn.ModuleList([max_pool_cls(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int = 5
    ):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        self.cv2 = Conv(n_dim, hidden_channels * 4, out_channels, 1, 1)
        max_pool_cls = nn.MaxPool2d if n_dim == 2 else nn.MaxPool3d
        self.m = max_pool_cls(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n=1
    ):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(n_dim, in_channels, out_channels, 1, stride=1)
        self.m = nn.Sequential(*(Conv(n_dim, out_channels, out_channels, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.n_dim = n_dim
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.cv2 = Conv(2 * hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        # self.attention = ChannelAttention(2 * self.hidden_channels)  # or SpatialAttention()
        self.m = nn.Sequential(
            *(Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, k=((3, 3), (3, 3)), e=1.0)
              for _ in range(n))
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.n_dim = n_dim
        self.hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, (2 + n) * self.hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.ModuleList(
            Bottleneck(
                n_dim, self.hidden_channels, self.hidden_channels, shortcut, groups=groups, k=((3, 3), (3, 3)), e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.hidden_channels, self.hidden_channels), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        self.cv2 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        self.cv3 = Conv(n_dim, 2 * hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.Sequential(*[
            Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, k=((1, 1), (3, 3)), e=1.0)
            for _ in range(n)
        ])

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups, e)
        hidden_channels = int(out_channels * e)
        self.m = nn.Sequential(*[
            Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, k=((1, 3), (3, 1)), e=1)
            for _ in range(n)
        ])


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            e=0.5
    ):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        self.cv2 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        self.m = nn.Sequential(*[RepConv(n_dim, hidden_channels, hidden_channels) for _ in range(n)])
        self.cv3 = Conv(n_dim, hidden_channels, out_channels, 1, 1) \
            if hidden_channels != out_channels \
            else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups, e)
        hidden_channels = int(out_channels * e)
        self.m = TransformerBlock(n_dim, hidden_channels, hidden_channels, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e=0.5
    ):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups, e)
        hidden_channels = int(out_channels * e)
        self.m = nn.Sequential(*(GhostBottleneck(n_dim, hidden_channels, hidden_channels) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            k: int = 3,
            stride: int = 1
    ):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv = nn.Sequential(
            GhostConv(n_dim, in_channels, hidden_channels, 1, stride=1),  # pw
            DWConv(
                n_dim, hidden_channels, hidden_channels, k, stride=stride, act=False
            ) if stride == 2 else nn.Identity(),  # dw
            GhostConv(n_dim, hidden_channels, out_channels, 1, stride=1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(n_dim, in_channels, in_channels, k, stride=stride, act=False),
                Conv(n_dim, in_channels, out_channels, 1, stride=1, act=False)
            ) if stride == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            shortcut: bool = True,
            groups: int = 1,
            k: tuple[int, ...] | tuple[tuple[int, ...], ...] = (3, 3),
            e: float = 0.5
    ):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, k[0], stride=1)
        self.cv2 = Conv(n_dim, hidden_channels, out_channels, k[1], stride=1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            e: float = 0.5
    ):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, stride=1)
        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.cv2 = conv_cls(in_channels, hidden_channels, 1, stride=1, bias=False)
        self.cv3 = conv_cls(hidden_channels, hidden_channels, 1, stride=1, bias=False)
        self.cv4 = Conv(n_dim, 2 * hidden_channels, out_channels, 1, stride=1)
        bn_cls = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        self.bn = bn_cls(2 * hidden_channels)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            hidden_channels: int,
            stride: int = 1,
            e: int = 4  # out_channels = e * hidden_channels
    ):
        """Initialize convolution with given parameters."""
        super().__init__()
        out_channels = e * hidden_channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, k=1, stride=1, act=True)
        self.cv2 = Conv(n_dim, hidden_channels, hidden_channels, k=3, stride=stride, padding=1, act=True)
        self.cv3 = Conv(n_dim, hidden_channels, out_channels, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(n_dim, in_channels, out_channels, k=1, stride=stride, act=False)) \
            if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            hidden_channels: int,
            stride: int = 1,
            is_first: bool = False,
            n: int = 1,
            e: int = 4  # out_channels = e * hidden_channels
    ):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            max_pool_cls = nn.MaxPool2d if n_dim == 2 else nn.MaxPool3d
            self.layer = nn.Sequential(
                Conv(n_dim, in_channels, hidden_channels, k=7, stride=2, padding=3, act=True),
                max_pool_cls(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(n_dim, in_channels, hidden_channels, stride=stride, e=e)]
            blocks.extend([
                ResNetBlock(n_dim, e * hidden_channels, hidden_channels, stride=1, e=e)
                for _ in range(n - 1)
            ])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.n_dim = n_dim
        self.nh = nh
        self.hc = out_channels // nh
        self.ec = Conv(n_dim, in_channels, ec, k=1, act=False) if in_channels != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(n_dim, in_channels, out_channels, k=3, stride=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, *dhw = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, *dhw)

        # TODO test 3D
        if self.n_dim == 2:
            aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        else:
            aw = torch.einsum("bmcdhw,bnmc->bmdhwn", embed, guide)

        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)

        # TODO test 3D
        if self.n_dim == 2:
            aw += self.bias[None, :, None, None]
        else:
            aw += self.bias[None, :, None, None, None]

        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, *dhw)
        x *= aw.unsqueeze(2)
        return x.view(bs, -1, *dhw)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            n=1, ec=128, nh=1, gc=512, shortcut=False, groups=1, e=0.5
    ):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, (3 + n) * self.hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.ModuleList(
            Bottleneck(
                n_dim, self.hidden_channels, self.hidden_channels, shortcut, groups=groups, k=((3, 3), (3, 3)), e=1.0
            )
            for _ in range(n)
        )
        self.attn = MaxSigmoidAttnBlock(n_dim, self.hidden_channels, self.hidden_channels, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, n_dim: int, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.projections = nn.ModuleList([conv_cls(in_channels, ec, kernel_size=1) for in_channels in ch])
        adap_max_pool_cls = nn.AdaptiveMaxPool2d if n_dim == 2 else nn.AdaptiveMaxPool3d
        self.im_pools = nn.ModuleList([adap_max_pool_cls((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, n_dim: int, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim
        self.norm = nn.BatchNorm2d(embed_dims) if n_dim == 2 else nn.BatchNorm3d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        if self.n_dim == 2:
            x = torch.einsum(("bchw,bkc->bkhw", x, w))
        else:
            x = torch.einsum(("bcdhw,bkc->bkdhw", x, w))
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int, shortcut=True, groups=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(in_channels, out_channels, shortcut, groups=groups, k=k, e=e)
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = RepConv(n_dim, in_channels, hidden_channels, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int, n=1, shortcut=True, groups=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups=groups, e=e)
        hidden_channels = int(out_channels * e)  # hidden channels
        self.m = nn.Sequential(*(
            RepBottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, e=1.0)
            for _ in range(n)
        ))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            hidden_channels2: int,
            n=1
    ):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.hidden_channels = hidden_channels // 2
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        self.cv2 = nn.Sequential(
            RepCSP(hidden_channels // 2, hidden_channels2, n),
            Conv(n_dim, hidden_channels2, hidden_channels2, 3, 1)
        )
        self.cv3 = nn.Sequential(
            RepCSP(hidden_channels2, hidden_channels2, n),
            Conv(n_dim, hidden_channels2, hidden_channels2, 3, 1)
        )
        self.cv4 = Conv(n_dim, hidden_channels + (2 * hidden_channels2), out_channels, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            hidden_channels2: int,
    ):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(n_dim, in_channels, out_channels, hidden_channels, hidden_channels2)
        self.hidden_channels = hidden_channels // 2
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, hidden_channels // 2, hidden_channels2, 3, 1)
        self.cv3 = Conv(n_dim, hidden_channels2, hidden_channels2, 3, 1)
        self.cv4 = Conv(n_dim, hidden_channels + (2 * hidden_channels2), out_channels, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.n_dim = n_dim
        self.cv1 = Conv(n_dim, in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        avg_pool_func = nn.functional.avg_pool2d if self.n_dim == 2 else nn.functional.avg_pool3d
        x = avg_pool_func(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int):
        """
        Initializes ADown module with convolution layers to downsample input from channels in_channels to out_channels.
        """
        super().__init__()
        self.hidden_channels = out_channels // 2
        self.cv1 = Conv(n_dim, in_channels // 2, self.hidden_channels, 3, 2, 1)
        self.cv2 = Conv(n_dim, in_channels // 2, self.hidden_channels, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        avg_pool_func = nn.functional.avg_pool2d if self.n_dim == 2 else nn.functional.avg_pool3d
        x = avg_pool_func(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        max_pool_func = nn.functional.max_pool2d if self.n_dim == 2 else nn.functional.max_pool3d
        x2 = max_pool_func(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: int, hidden_channels: int, k: int = 5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        max_pool_cls = nn.MaxPool2d if self.n_dim == 2 else nn.MaxPool3d
        self.cv2 = max_pool_cls(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = max_pool_cls(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = max_pool_cls(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(n_dim, 4 * hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, n_dim: int, in_channels: int, out_channels: list[int], k=1, stride=1, padding=None, groups=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        assert n_dim in [2, 3]
        super().__init__()
        self.out_channels = out_channels
        conv_cls = nn.Conv2d if n_dim == 2 else nn.Conv3d
        self.conv = conv_cls(
            in_channels, sum(out_channels), k, stride=stride, padding=autopad(k, padding), groups=groups, bias=True
        )

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.out_channels, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, shortcut=False, groups=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, in_channels, hidden_channels, 1, 1)
        self.cv3 = Conv(n_dim, (2 + n) * hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.ModuleList(
            Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, c3k=False, e=0.5, groups=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups=groups, e=e)
        self.m = nn.ModuleList(
            C3k(n_dim, self.hidden_channels, self.hidden_channels, 2, shortcut, groups=groups)
            if c3k else
            Bottleneck(n_dim, self.hidden_channels, self.hidden_channels, shortcut, groups=groups)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, shortcut=True, groups=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups, e)
        hidden_channels = int(out_channels * e)  # hidden channels
        self.m = nn.Sequential(*(
            Bottleneck(n_dim, hidden_channels, hidden_channels, shortcut, groups=groups, k=(k, k), e=1.0)
            for _ in range(n)
        ))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, n_dim: int, ed: int) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(n_dim, ed, ed, 7, 1, 3, groups=ed, act=False)
        self.conv1 = Conv(n_dim, ed, ed, 3, 1, 1, groups=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, n_dim: int, in_channels, out_channels, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        self.hidden_channels = hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(n_dim, in_channels, in_channels, 3, groups=in_channels),
            Conv(n_dim, in_channels, 2 * hidden_channels, 1),
            RepVGGDW(n_dim, 2 * hidden_channels)
            if lk else
            Conv(n_dim, 2 * hidden_channels, 2 * hidden_channels, 3, groups=2 * hidden_channels),
            Conv(n_dim, 2 * hidden_channels, out_channels, 1),
            Conv(n_dim, out_channels, out_channels, 3, groups=out_channels)
        )

        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, shortcut=False, lk=False, groups=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(n_dim, in_channels, out_channels, n, shortcut, groups, e)
        self.m = nn.ModuleList(
            CIB(n_dim, self.hidden_channels, self.hidden_channels, shortcut, e=1.0, lk=lk)
            for _ in range(n)
        )


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, n_dim: int, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(n_dim, dim, h, 1, act=False)
        self.proj = Conv(n_dim, dim, dim, 1, act=False)
        self.pe = Conv(n_dim, dim, dim, 3, stride=1, groups=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, *DHW = x.shape
        N = np.prod(DHW)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, *DHW) + self.pe(v.reshape(B, C, *DHW))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, n_dim: int, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(n_dim, c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(n_dim, c, c * 2, 1), Conv(n_dim, c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        hidden_channels (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(in_channels=128, out_channels=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, n_dim: int, in_channels, out_channels, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert in_channels == out_channels
        self.hidden_channels = int(in_channels * e)
        self.cv1 = Conv(n_dim, in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, 2 * self.hidden_channels, in_channels, 1)

        self.attn = Attention(n_dim, self.hidden_channels, attn_ratio=0.5, num_heads=self.hidden_channels // 64)
        self.ffn = nn.Sequential(
            Conv(n_dim, self.hidden_channels, self.hidden_channels * 2, 1),
            Conv(n_dim, self.hidden_channels * 2, self.hidden_channels, 1, act=False)
        )

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.hidden_channels, self.hidden_channels), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        hidden_channels (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> out_channelspsa = C2PSA(in_channels=256, out_channels=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = out_channelspsa(input_tensor)
    """

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert in_channels == out_channels
        self.hidden_channels = int(in_channels * e)
        self.cv1 = Conv(n_dim, in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv(n_dim, 2 * self.hidden_channels, in_channels, 1)

        self.m = nn.Sequential(*(
            PSABlock(n_dim, self.hidden_channels, attn_ratio=0.5, num_heads=self.hidden_channels // 64)
            for _ in range(n)
        ))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.hidden_channels, self.hidden_channels), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms
    and feature extraction.

    Attributes:
        hidden_channels (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> model = C2fPSA(in_channels=64, out_channels=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, n_dim: int, in_channels, out_channels, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert in_channels == out_channels
        super().__init__(n_dim, in_channels, out_channels, n=n, e=e)
        self.m = nn.ModuleList(
            PSABlock(n_dim, self.hidden_channels, attn_ratio=0.5, num_heads=self.hidden_channels // 64)
            for _ in range(n)
        )


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> model = SCDown(in_channels=64, out_channels=128, k=3, stride=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, n_dim: int, in_channels, out_channels, k, stride):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(n_dim, in_channels, out_channels, 1, stride=1)
        self.cv2 = Conv(n_dim, out_channels, out_channels, k=k, stride=stride, groups=out_channels, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
