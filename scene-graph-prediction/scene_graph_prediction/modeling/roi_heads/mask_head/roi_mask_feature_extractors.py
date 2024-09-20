# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.backbone import resnet
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import build_pooler, ROIHeadName
from scene_graph_prediction.modeling.utils.build_layers import build_conv3x3
from scene_graph_prediction.structures import BoxList
from ...abstractions.mask_head import ROIMaskFeatureExtractor, MaskHeadFeatures


@ROI_MASK_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(ROIMaskFeatureExtractor):
    """Note: supports 2D and 3D."""

    def __init__(self, cfg: CfgNode, _: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, _, anchor_strides)
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

        self.pooler = build_pooler(cfg, ROIHeadName.Mask, anchor_strides)
        self.out_channels = self.head.out_channels

    def forward(self, x: list[torch.Tensor], proposals: list[BoxList]) -> MaskHeadFeatures:
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(ROIMaskFeatureExtractor):
    """Note: Support 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels, anchor_strides)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        self.pooler = build_pooler(cfg, ROIHeadName.Mask, anchor_strides)

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

    def forward(self, x: list[torch.Tensor], proposals: list[BoxList]) -> MaskHeadFeatures:
        x = self.pooler(x, proposals)
        x = self.blocks(x)
        return x


def build_roi_mask_feature_extractor(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides
) -> ROIMaskFeatureExtractor:
    extractor = ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return extractor(cfg, in_channels, anchor_strides)
