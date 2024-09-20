# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn.functional import relu

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadFeatures
from scene_graph_prediction.modeling.backbone import resnet
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import build_pooler, build_pooler_extra_args, ROIHeadName
from scene_graph_prediction.modeling.utils.build_layers import build_group_norm, build_fc, build_conv, NormType


@ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(ROIBoxFeatureExtractor):
    """
    We consider the output of the pooler as a 2D/3D volume and use convolutions to extract information.
    Then we use adaptive pooling to convert the features to a 1D feature vector.
    # Note: not tested...
    Note: supports 2D and 3D.
    """

    def __init__(self, cfg: CfgNode, _: int, anchor_strides: AnchorStrides, __: bool = False, ___: bool = False):
        super().__init__(cfg, _, __, ___)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            n_dim=self.n_dim,
            block_module_name=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = build_pooler(cfg, ROIHeadName.BoundingBox, anchor_strides)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) if self.n_dim == 2 else torch.nn.AdaptiveAvgPool3d(1)
        # Only made possible by the average pooling
        self.representation_size = self.head.out_channels

    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        x = self.head(x)
        # Pool from NxCxDxHxW to NxCx1x1x1
        x = self.avg_pool(x)

        # Flatten the features
        return x.squeeze()


@ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(ROIBoxFeatureExtractor):
    """
    We consider the output of the pooler as a flattened vector and feed it to an MLP.
    Note: supports 2D and 3D.
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            half_out: bool = False,
            cat_all_levels: bool = False
    ):
        super().__init__(cfg, in_channels, half_out, cat_all_levels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        input_size = in_channels * cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2
        if self.n_dim == 3:
            input_size *= cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH

        representation_size = cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = build_pooler_extra_args(cfg, "ROI_BOX_HEAD", anchor_strides, in_channels, cat_all_levels)

        self.fc6 = build_fc(input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = build_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.representation_size = out_dim

    def forward_without_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = relu(self.fc6(x))
        x = relu(self.fc7(x))
        return x


@ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(ROIBoxFeatureExtractor):
    """
    Heads for FPN for classification.
    # Note: not tested...
    Note: supports 2D and 3D.
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            _: bool = False,
            __: bool = False
    ):
        super().__init__(cfg, in_channels, _, __)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        self.pooler = build_pooler(cfg, ROIHeadName.BoundingBox, anchor_strides)

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        x_convs = []
        conv_module = torch.nn.Conv2d if self.n_dim == 2 else torch.nn.Conv3d
        for ix in range(num_stacked_convs):
            x_convs.append(
                conv_module(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                x_convs.append(build_group_norm(in_channels))
            x_convs.append(torch.nn.ReLU(inplace=True))

        self.x_convs = torch.nn.Sequential(*x_convs)
        for layer in self.x_convs.modules():
            if isinstance(layer, conv_module):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if not use_gn:
                    torch.nn.init.constant_(layer.bias, 0)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) if self.n_dim == 2 else torch.nn.AdaptiveAvgPool3d(1)
        self.representation_size = conv_head_dim

    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        x = self.x_convs(x)
        # Pool from NxCxDxHxW to NxCx1x1x1
        x = self.avg_pool(x)
        return x.squeeze()


@ROI_BOX_FEATURE_EXTRACTORS.register("RetinaNetClsTowerFeatureExtractor")
class RetinaNetClsTowerFeatureExtractor(ROIBoxFeatureExtractor):
    """
    Classifier head from RetinaNet, but without the final the prediction convolution.
    Note: supports 2D and 3D.
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            _: bool = False,
            cat_all_levels: bool = False
    ):
        super().__init__(cfg, in_channels, _, cat_all_levels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        self.pooler = build_pooler_extra_args(cfg, ROIHeadName.BoundingBox, anchor_strides, in_channels, cat_all_levels)

        n_features = 256 if self.n_dim == 2 else 128
        # Classification
        representation_size = cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE
        self.cls_tower = torch.nn.Sequential(
            build_conv(self.n_dim, in_channels, n_features,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True),
            build_conv(self.n_dim, n_features, representation_size,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True)
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) if self.n_dim == 2 else torch.nn.AdaptiveAvgPool3d(1)
        self.representation_size = representation_size

    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        x = self.cls_tower(x)
        # Pool from NxCxDxHxW to NxCx1x1x1
        x = self.avg_pool(x)
        return x.squeeze()


@ROI_BOX_FEATURE_EXTRACTORS.register("DownScaleConvFeatureExtractor")
class DownScaleConvFeatureExtractor(ROIBoxFeatureExtractor):
    """
    A set of 2 convolutional layer with max pooling and batch normalization,
    that reduce the resolution by a factor 4 along each axis.
    Compared to other extractors, there is no average pooling and instead the number of final channels is adjusted
    to fit the representation size (given the final the spatial shape post-convolution).
    Note: supports 2D and 3D.
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            _: bool = False,
            cat_all_levels: bool = False
    ):
        super().__init__(cfg, in_channels, _, cat_all_levels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        self.pooler = build_pooler_extra_args(cfg, ROIHeadName.BoundingBox, anchor_strides, in_channels, cat_all_levels)

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        resolution_depth = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH
        representation_size = cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE

        if self.n_dim == 2:
            # 2x2=4 downscale factor, per level and 2 levels = 16
            post_conv_size = resolution * resolution // 16
        else:
            # 2x2x2=8 downscale factor, per level and 2 levels = 64
            post_conv_size = resolution_depth * resolution * resolution // 64
        assert representation_size % post_conv_size == 0, \
            f"Representation size {representation_size} is not divisiable by {post_conv_size}."
        final_features = representation_size // post_conv_size
        n_features = 256 if self.n_dim == 2 else 128

        if self.n_dim == 2:
            self.x_convs = nn.Sequential(
                nn.Conv2d(in_channels, n_features, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(n_features, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(n_features, final_features, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(final_features, momentum=0.01),
            )
        else:
            self.x_convs = nn.Sequential(
                nn.Conv3d(in_channels, n_features, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(n_features, momentum=0.01),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                nn.Conv3d(n_features, final_features, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(final_features, momentum=0.01),
            )
        self.representation_size = representation_size

    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        x = self.x_convs(x)
        # Then just flatten
        return x.view(x.shape[0], -1)


def build_feature_extractor(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides,
        # Only produce half the output features as they are split between detection and attribute prediction
        half_out: bool = False,
        cat_all_levels: bool = False,
        roi_head: ROIHeadName = ROIHeadName.BoundingBox
) -> ROIBoxFeatureExtractor:
    extractor = ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL[roi_head.value].FEATURE_EXTRACTOR]
    return extractor(cfg, in_channels, anchor_strides, half_out, cat_all_levels)
