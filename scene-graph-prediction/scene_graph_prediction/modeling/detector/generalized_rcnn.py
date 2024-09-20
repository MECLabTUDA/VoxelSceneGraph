# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Implements the Generalized R-CNN framework."""

from yacs.config import CfgNode

from .base_detector import BaseDetector
from ..backbone import build_backbone
from ..region_proposal import build_rpn
from ..roi_heads import build_roi_heads


class GeneralizedRCNN(BaseDetector):
    """
    Main class for Generalized R-CNN. Currently, supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    Note: Only supports two stage methods.
    """

    def __init__(self, cfg: CfgNode):
        backbone = build_backbone(cfg)
        super().__init__(
            cfg,
            backbone,
            build_rpn(cfg, backbone.out_channels, backbone.feature_strides),
            build_roi_heads(
                cfg,
                in_channels=backbone.out_channels,
                anchor_strides=backbone.feature_strides,
                is_rpn_only=cfg.MODEL.RPN_ONLY,
                has_boxes=True,
                has_masks=cfg.MODEL.MASK_ON,
                has_keypoints=cfg.MODEL.KEYPOINT_ON,
                has_attributes=cfg.MODEL.ATTRIBUTE_ON,
                has_relations=cfg.MODEL.RELATION_ON,
            )
        )
