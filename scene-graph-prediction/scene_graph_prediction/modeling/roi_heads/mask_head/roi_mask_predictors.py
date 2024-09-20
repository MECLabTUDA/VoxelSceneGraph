# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.abstractions.mask_head import MaskHeadFeatures, MaskLogits
from scene_graph_prediction.modeling.registries import *


@ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(ROIMaskPredictor):
    """Note: Support 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        num_classes = cfg.INPUT.N_OBJ_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if self.n_dim == 2:
            self.conv5_mask = torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        else:
            self.conv5_mask = torch.nn.ConvTranspose3d(in_channels, dim_reduced, 2, 2, 0)
            self.mask_fcn_logits = torch.nn.Conv3d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which corresponds to kaiming_normal_ in PyTorch
                torch.nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x: MaskHeadFeatures) -> MaskLogits:
        x = torch.nn.functional.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


@ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(ROIMaskPredictor):
    """Note: Support 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        num_classes = cfg.INPUT.N_OBJ_CLASSES
        num_inputs = in_channels

        if self.n_dim == 2:
            self.mask_fcn_logits = torch.nn.Conv2d(num_inputs, num_classes, 1, 1, 0)
        else:
            self.mask_fcn_logits = torch.nn.Conv3d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which corresponds to kaiming_normal_ in PyTorch
                torch.nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x: MaskHeadFeatures) -> MaskLogits:
        return self.mask_fcn_logits(x)


def build_roi_mask_predictor(cfg: CfgNode, in_channels: int) -> ROIMaskPredictor:
    predictor = ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
