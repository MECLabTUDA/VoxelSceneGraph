from typing import Sequence

import torch

from scene_graph_prediction.modeling.abstractions.keypoint_head import KeypointHeadTargets, KeypointHeadFeatures
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import build_pooler, ROIHeadName


@ROI_KEYPOINT_FEATURE_EXTRACTORS.register("KeypointRCNNFeatureExtractor")
class KeypointRCNNFeatureExtractor(ROIKeypointFeatureExtractor):
    """Note: supports 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides, ):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        self.pooler = build_pooler(cfg, ROIHeadName.Keypoint, anchor_strides)

        input_features = in_channels
        layers: Sequence[int] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        assert layers
        next_feature = input_features
        conv_module = torch.nn.Conv2d if self.n_dim == 2 else torch.nn.Conv3d
        blocks = []
        for layer_features in layers:
            module = conv_module(next_feature, layer_features, 3, stride=1, padding=1)
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            torch.nn.init.constant_(module.bias, 0)
            next_feature = layer_features
            blocks.append(module)
            blocks.append(torch.nn.ReLU())
        self.blocks = torch.nn.Sequential(*blocks)

        # noinspection PyUnboundLocalVariable
        self.representation_size = layer_features

    def forward(self, x: KeypointHeadFeatures, proposals: KeypointHeadTargets) -> KeypointHeadFeatures:
        x = self.pooler(x, proposals)
        x = self.blocks(x)
        return x


def build_roi_keypoint_feature_extractor(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides
) -> ROIKeypointFeatureExtractor:
    extractor = ROI_KEYPOINT_FEATURE_EXTRACTORS[cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR]
    return extractor(cfg, in_channels, anchor_strides)
