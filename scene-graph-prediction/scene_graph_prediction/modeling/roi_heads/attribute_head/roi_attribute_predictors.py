# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.abstractions.attribute_head import AttributeHeadFeatures, AttributeLogits
from scene_graph_prediction.modeling.registries import *


@ROI_ATTRIBUTE_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(ROIAttributePredictor):
    """Note: supports 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        num_attributes = cfg.INPUT.N_ATT_CLASSES

        self.att_score = torch.nn.Linear(in_channels, num_attributes)

        torch.nn.init.normal_(self.att_score.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.att_score.bias, 0)

    def forward(self, x: AttributeHeadFeatures) -> AttributeLogits:
        att_logit = self.att_score(x)
        return att_logit


def build_roi_attribute_predictor(cfg: CfgNode, in_channels: int) -> ROIAttributePredictor:
    predictor = ROI_ATTRIBUTE_PREDICTOR[cfg.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
