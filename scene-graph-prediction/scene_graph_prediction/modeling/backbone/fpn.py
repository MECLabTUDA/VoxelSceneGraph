"""
FPN used in Retina-U-Net as in: https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/models/backbone.py#L22.
"""

import torch

from ..abstractions.backbone import FeatureMaps, Backbone, Images, AnchorStrides
from ..utils.build_layers import build_conv, NormType


class _StackedBlock(torch.nn.Sequential):
    """
    Conv block used in RetinaNet (1 block of 2 convs).
    Note: supports 2D and 3D.
    """
    MAX_OUT = 320

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple[int, ...],
            stride: tuple[int, ...] | int = 1,
            norm: NormType | None = None
    ):
        padding = tuple([ks // 2 for ks in kernel_size])

        if out_channels > self.MAX_OUT:
            out_channels = self.MAX_OUT

        # Block 0
        c0 = build_conv(n_dim, in_channels, out_channels, kernel_size, stride, padding, norm, activation=True)
        c1 = build_conv(n_dim, out_channels, out_channels, kernel_size, 1, padding, norm, activation=True)
        super().__init__(c0, c1)
        self.n_dim = n_dim
        self.out_channels = c1.out_channels


class _Interpolate(torch.nn.Module):
    def __init__(self, n_dim: int):
        super().__init__()
        self.n_dim = n_dim
        self.scale_factor = 2 if n_dim == 2 else (1, 2, 2)
        self.mode = "bilinear" if n_dim == 2 else "trilinear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class FPN(Backbone):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf.
    Returns levels P0 to P6 included.
    Note: Supports 2D and 3D.
    """

    def __init__(self, n_dim: int, in_channels: int):
        super().__init__()
        assert n_dim in [2, 3]
        self.n_dim = n_dim
        self.in_channels = in_channels
        start_channels = 48 if self.n_dim == 2 else 32
        expansion_factor = 4 if self.n_dim == 2 else 2
        self.out_channels = start_channels * expansion_factor

        # Conv down encoder
        special_ks = (1, 3, 3) if n_dim == 3 else (3, 3)
        normal_ks = (3,) * self.n_dim
        special_padding = (0, 1, 1) if n_dim == 3 else (1, 1)
        normal_padding = (1,) * self.n_dim
        special_stride = (1, 2, 2) if n_dim == 3 else 2
        normal_stride = 2

        self.conv0 = _StackedBlock(n_dim, in_channels, start_channels, special_ks, stride=1, norm=NormType.Instance)

        self.conv1 = _StackedBlock(n_dim, self.conv0.out_channels, self.conv0.out_channels * expansion_factor,
                                   special_ks, stride=special_stride, norm=NormType.Instance)

        self.conv2 = _StackedBlock(n_dim, self.conv1.out_channels, self.conv1.out_channels * expansion_factor,
                                   special_ks, stride=special_stride, norm=NormType.Instance)

        self.conv3 = _StackedBlock(n_dim, self.conv2.out_channels, self.conv2.out_channels * expansion_factor,
                                   normal_ks, stride=special_stride, norm=NormType.Instance)

        self.conv4 = _StackedBlock(n_dim, self.conv3.out_channels, self.conv3.out_channels * expansion_factor,
                                   normal_ks, stride=normal_stride, norm=NormType.Instance)

        self.conv5 = _StackedBlock(n_dim, self.conv4.out_channels, self.conv4.out_channels * expansion_factor,
                                   normal_ks, stride=normal_stride, norm=NormType.Instance)

        self.conv6 = _StackedBlock(n_dim, self.conv5.out_channels, self.conv5.out_channels * expansion_factor,
                                   normal_ks, stride=special_stride, norm=NormType.Instance)

        # Conv lateral decoder
        self.p6_conv1 = build_conv(n_dim, self.conv6.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p5_conv1 = build_conv(n_dim, self.conv5.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p4_conv1 = build_conv(n_dim, self.conv4.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p3_conv1 = build_conv(n_dim, self.conv3.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p2_conv1 = build_conv(n_dim, self.conv2.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p1_conv1 = build_conv(n_dim, self.conv1.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.p0_conv1 = build_conv(n_dim, self.conv0.out_channels, self.out_channels,
                                   kernel_size=1, stride=1, padding=0, norm=None, activation=False)

        # Up-sampling using a conv
        # Note: nnDet uses nearest-neighbor interpolation rather than linear
        self.p6_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=special_stride, stride=special_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p5_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=normal_stride, stride=normal_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p4_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=normal_stride, stride=normal_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p3_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=special_stride, stride=special_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p2_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=special_stride, stride=special_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p1_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=special_stride, stride=special_stride, padding=0,
                                      norm=None, activation=False, transposed=True)

        # Feature refinement at each individual scale
        self.p6_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=normal_ks, stride=1,
                                   padding=normal_padding, norm=None, activation=False)
        self.p5_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=normal_ks, stride=1,
                                   padding=normal_padding, norm=None, activation=False)
        self.p4_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=normal_ks, stride=1,
                                   padding=normal_padding, norm=None, activation=False)
        self.p3_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=normal_ks, stride=1,
                                   padding=normal_padding, norm=None, activation=False)
        self.p2_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=special_ks, stride=1,
                                   padding=special_padding, norm=None, activation=False)
        self.p1_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=special_ks, stride=1,
                                   padding=special_padding, norm=None, activation=False)
        self.p0_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=special_ks, stride=1,
                                   padding=special_padding, norm=None, activation=False)

    def forward(self, x: Images) -> FeatureMaps:
        # Conv down
        c0_out = self.conv0(x)
        c1_out = self.conv1(c0_out)
        c2_out = self.conv2(c1_out)
        c3_out = self.conv3(c2_out)
        c4_out = self.conv4(c3_out)
        c5_out = self.conv5(c4_out)
        c6_out = self.conv6(c5_out)

        # Conv up (pre-out)
        p6_pre_out = self.p6_conv1(c6_out)
        p5_pre_out = self.p5_conv1(c5_out) + self.p6_upsample(p6_pre_out)
        p4_pre_out = self.p4_conv1(c4_out) + self.p5_upsample(p5_pre_out)
        p3_pre_out = self.p3_conv1(c3_out) + self.p4_upsample(p4_pre_out)
        p2_pre_out = self.p2_conv1(c2_out) + self.p3_upsample(p3_pre_out)
        p1_pre_out = self.p1_conv1(c1_out) + self.p2_upsample(p2_pre_out)
        p0_pre_out = self.p0_conv1(c0_out) + self.p1_upsample(p1_pre_out)

        # Out
        p0_out = self.p0_conv2(p0_pre_out)
        p1_out = self.p1_conv2(p1_pre_out)
        p2_out = self.p2_conv2(p2_pre_out)
        p3_out = self.p3_conv2(p3_pre_out)
        p4_out = self.p4_conv2(p4_pre_out)
        p5_out = self.p5_conv2(p5_pre_out)
        p6_out = self.p6_conv2(p6_pre_out)

        # Note: p1_out is usually not needed...
        return [p0_out, p1_out, p2_out, p3_out, p4_out, p5_out, p6_out]

    @property
    def feature_strides(self) -> AnchorStrides:
        if self.n_dim == 2:
            return (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)
        return (1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8), (2, 16, 16), (4, 32, 32), (4, 64, 64)


class FPN2to6(FPN):
    """FPN but returning only level 2 to 6 baked into it since it's the most common setup anyway."""
    def forward(self, x: Images) -> FeatureMaps:
        return super().forward(x)[2:]

    @property
    def feature_strides(self) -> AnchorStrides:
        return super().feature_strides[2:]
