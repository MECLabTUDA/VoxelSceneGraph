# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from scene_graph_prediction.modeling.abstractions.relation_head import ROIRelationMaskFeatureExtractor
from scene_graph_prediction.modeling.registries import *


@ROI_RELATION_MASK_FEATURE_EXTRACTORS.register("Factor444RelationMaskFeatureExtractor")
class Factor444RelationMaskFeatureExtractor(ROIRelationMaskFeatureExtractor):
    """Convolutional feature extractor which uses masks 4x larger than Pooler outputs."""

    def __init__(self, cfg: CfgNode, out_channels: int):
        super().__init__(cfg, out_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        resolution_depth = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH
        if self.n_dim == 2:
            self.rect_size = resolution * 4, resolution * 4
        else:
            self.rect_size = resolution_depth * 4, resolution * 4, resolution * 4

        if self.n_dim == 2:
            self.rect_conv = nn.Sequential(
                nn.Conv2d(2, out_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // 2, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels, momentum=0.01),
            )
        else:
            self.rect_conv = nn.Sequential(
                nn.Conv3d(2, out_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels // 2, momentum=0.01),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels, momentum=0.01),
            )

    def get_orig_rect_size(self) -> tuple[int, ...]:
        return self.rect_size

    def forward(self, masks: torch.FloatTensor) -> torch.FloatTensor:
        return self.rect_conv(masks)


@ROI_RELATION_MASK_FEATURE_EXTRACTORS.register("Factor488RelationMaskFeatureExtractor")
class Factor488RelationMaskFeatureExtractor(ROIRelationMaskFeatureExtractor):
    """Convolutional feature extractor which uses masks (4,8,8) times larger than Pooler outputs."""

    def __init__(self, cfg: CfgNode, out_channels: int):
        super().__init__(cfg, out_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        resolution_depth = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH
        if self.n_dim == 2:
            self.rect_size = resolution * 8, resolution * 8
        else:
            self.rect_size = resolution_depth * 4, resolution * 8, resolution * 8

        if self.n_dim == 2:
            self.rect_conv = nn.Sequential(
                nn.Conv2d(2, out_channels // 4, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // 4, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(out_channels // 4, out_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // 2, momentum=0.01),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels, momentum=0.01),
            )
        else:
            self.rect_conv = nn.Sequential(
                nn.Conv3d(2, out_channels // 4, kernel_size=7, stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels // 4, momentum=0.01),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                nn.Conv3d(out_channels // 4, out_channels // 2, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels // 2, momentum=0.01),
                nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels, momentum=0.01),
            )

    def get_orig_rect_size(self) -> tuple[int, ...]:
        return self.rect_size

    def forward(self, masks: torch.FloatTensor) -> torch.FloatTensor:
        return self.rect_conv(masks)


def build_relation_mask_feature_extractor(cfg: CfgNode, out_channels: int) -> ROIRelationMaskFeatureExtractor:
    extractor = ROI_RELATION_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_RELATION_HEAD.MASK_FEATURE_EXTRACTOR]
    return extractor(cfg, out_channels)
