# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import build_pooler, ROIHeadName
from scene_graph_prediction.modeling.utils.build_layers import build_conv3x3, NormType, build_conv
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractorBase, \
    FPNXconv1fcFeatureExtractorBase, RetinaNetClsTowerFeatureExtractorBase
from ...abstractions.box_head import ROIBoxFeatureExtractor
from ...abstractions.mask_head import MaskHeadFeatures


@ROI_MASK_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(ResNet50Conv5ROIFeatureExtractorBase):
    """Note: supports 2D and 3D."""

    def __init__(self, cfg: CfgNode, _: int, anchor_strides: AnchorStrides, __: bool = False, ___: bool = False):
        super().__init__(cfg, _, anchor_strides, ROIHeadName.Mask, __, ___, level_mapper_name="AlwaysZero")


@ROI_MASK_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(FPNXconv1fcFeatureExtractorBase):
    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            _: bool = False,
            __: bool = False
    ):
        super().__init__(cfg, in_channels, anchor_strides, ROIHeadName.Mask, _, __, level_mapper_name="AlwaysZero")


@ROI_MASK_FEATURE_EXTRACTORS.register("RetinaNetClsTowerFeatureExtractor")
class RetinaNetClsTowerFeatureExtractor(RetinaNetClsTowerFeatureExtractorBase):
    """
    Classifier head from RetinaNet, but without the final the prediction convolution.
    Note: supports 2D and 3D.
    """
    is_mask_head_compatible = True

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            _: bool = False,
            cat_all_levels: bool = False
    ):
        super().__init__(
            cfg,
            in_channels,
            anchor_strides,
            ROIHeadName.Mask,
            _,
            cat_all_levels,
            level_mapper_name="AlwaysZero"
        )


@ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(ROIBoxFeatureExtractor):
    """Note: Support 2D and 3D."""
    is_mask_head_compatible = True

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels, anchor_strides)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        self.pooler = build_pooler(cfg, ROIHeadName.Mask, anchor_strides, level_mapper_name="AlwaysZero")

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        assert layers
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        prev_features = in_channels
        blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            module = build_conv3x3(
                self.n_dim,
                prev_features,
                layer_features,
                dilation=dilation,
                stride=1,
                use_gn=use_gn,
            )
            prev_features = layer_features
            blocks.append(module)
            blocks.append(torch.nn.ReLU())
        self.blocks = torch.nn.Sequential(*blocks)
        # noinspection PyUnboundLocalVariable
        self.out_channels = layer_features

    def forward_without_pool(self, x: torch.Tensor) -> MaskHeadFeatures:
        x = self.blocks(x)
        return x

    def output_size(self) -> tuple[int, ...]:
        return (self.out_channels,) + self.pooler.output_size


@ROI_MASK_FEATURE_EXTRACTORS.register("SmallUNetFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(ROIBoxFeatureExtractor):
    """
    UNet with depth 3 inspired by the Retina UNet FPN with a max spatial downscaling of 4x8x8.
    Note: Support 2D and 3D.
    """
    is_mask_head_compatible = True

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels, anchor_strides)
        n_dim = self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        self.pooler = build_pooler(cfg, ROIHeadName.Mask, anchor_strides, level_mapper_name="AlwaysZero")

        start_channels = in_channels
        expansion_factor = 4 if n_dim == 2 else 2
        self.out_channels = start_channels * expansion_factor
        from scene_graph_prediction.modeling.backbone.fpn import _StackedBlock

        # Conv down encoder
        special_ks = (1, 3, 3) if n_dim == 3 else (3, 3)
        normal_ks = (3,) * n_dim
        special_padding = (0, 1, 1) if n_dim == 3 else (1, 1)
        special_stride = (1, 2, 2) if n_dim == 3 else 2
        normal_stride = 2

        self.conv0 = _StackedBlock(n_dim, in_channels, start_channels, special_ks, stride=1, norm=NormType.Instance)

        self.conv1 = _StackedBlock(n_dim, self.conv0.out_channels, self.conv0.out_channels * expansion_factor,
                                   special_ks, stride=special_stride, norm=NormType.Instance)

        self.conv2 = _StackedBlock(n_dim, self.conv1.out_channels, self.conv1.out_channels * expansion_factor,
                                   normal_ks, stride=normal_stride, norm=NormType.Instance)

        self.conv3 = _StackedBlock(n_dim, self.conv2.out_channels, self.conv2.out_channels,
                                   normal_ks, stride=normal_stride, norm=NormType.Instance)

        # Conv lateral decoder
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
        self.p3_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=normal_stride, stride=normal_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p2_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=normal_stride, stride=normal_stride, padding=0,
                                      norm=None, activation=False, transposed=True)
        self.p1_upsample = build_conv(n_dim, self.out_channels, self.out_channels,
                                      kernel_size=special_stride, stride=special_stride, padding=0,
                                      norm=None, activation=False, transposed=True)

        # Feature refinement at each individual scale
        self.p0_conv2 = build_conv(n_dim, self.out_channels, self.out_channels,
                                   kernel_size=special_ks, stride=1,
                                   padding=special_padding, norm=None, activation=False)

    def forward_without_pool(self, x: torch.Tensor) -> MaskHeadFeatures:
        # Conv down
        c0_out = self.conv0(x)
        c1_out = self.conv1(c0_out)
        c2_out = self.conv2(c1_out)
        c3_out = self.conv3(c2_out)

        # Conv up (pre-out)
        p3_pre_out = self.p3_conv1(c3_out)
        p2_pre_out = self.p2_conv1(c2_out) + self.p3_upsample(p3_pre_out)
        p1_pre_out = self.p1_conv1(c1_out) + self.p2_upsample(p2_pre_out)
        p0_pre_out = self.p0_conv1(c0_out) + self.p1_upsample(p1_pre_out)

        # Out
        p0_out = self.p0_conv2(p0_pre_out)

        return p0_out

    def output_size(self) -> tuple[int, ...]:
        return (self.out_channels,) + self.pooler.output_size


def build_roi_mask_feature_extractor(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides
) -> ROIBoxFeatureExtractor:
    extractor = ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return extractor(cfg, in_channels, anchor_strides)
