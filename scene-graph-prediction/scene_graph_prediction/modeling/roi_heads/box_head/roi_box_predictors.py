# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.abstractions.box_head import ClassLogits, BboxRegression
from scene_graph_prediction.modeling.registries import *


@ROI_BOX_PREDICTOR.register("BoxPredictor")
class BoxPredictor(ROIBoxPredictor):
    """Note: supports ND."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        representation_size = in_channels
        num_classes = cfg.INPUT.N_OBJ_CLASSES

        self.cls_score = torch.nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = torch.nn.Linear(representation_size, num_bbox_reg_classes * 2 * self.n_dim)

        torch.nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.cls_score.bias, 0)

        torch.nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[ClassLogits, BboxRegression]:
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred


def build_roi_box_predictor(cfg: CfgNode, in_channels: int) -> ROIBoxPredictor:
    predictor = ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
