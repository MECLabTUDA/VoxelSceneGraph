# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Any, Callable

from yacs.config import CfgNode

from scene_graph_prediction.utils.registry import Registry
from .abstractions.attribute_head import ROIAttributePredictor
from .abstractions.backbone import Backbone, AnchorStrides
from .abstractions.box_head import ROIBoxFeatureExtractor, ROIBoxPredictor, ROIBoxHead, ROIBoxMaskFeatureExtractor
from .abstractions.keypoint_head import ROIKeypointFeatureExtractor, ROIKeypointPredictor
from .abstractions.mask_head import ROIMaskPredictor, ROIMaskHead
from .abstractions.region_proposal import RPNHead, RPN
from .abstractions.relation_head import ROIRelationFeatureExtractor

# Backbones
BACKBONES: Registry[str, Callable[[CfgNode], Backbone]] = Registry()

# RPN
RPN_HEADS: Registry[str, Callable[[CfgNode, int, int, int], RPNHead]] = Registry()
RPNS: Registry[str, Callable[[CfgNode, int, AnchorStrides], RPN]] = Registry()

# Box Prediction
ROI_BOX_FEATURE_EXTRACTORS: Registry[str, Callable[[CfgNode, int, AnchorStrides, bool, bool], ROIBoxFeatureExtractor]] \
    = Registry()
ROI_BOX_MASK_FEATURE_EXTRACTORS: Registry[str, Callable[[CfgNode, int, int], ROIBoxMaskFeatureExtractor]] = Registry()
ROI_BOX_PREDICTOR: Registry[str, Callable[[CfgNode, int], ROIBoxPredictor]] = Registry()

ROI_BOX_HEADS: Registry[str, Callable[[CfgNode, int, AnchorStrides], ROIBoxHead]] = Registry()

# Attribute Prediction
# Feature extractors for attributes were (before the refactoring) just a copy of the ones for the box head
# noinspection DuplicatedCode
ROI_ATTRIBUTE_PREDICTOR: Registry[str, Callable[[CfgNode, int], ROIAttributePredictor]] = Registry()

# Keypoint Prediction
ROI_KEYPOINT_FEATURE_EXTRACTORS: Registry[str, Callable[[CfgNode, int, AnchorStrides], ROIKeypointFeatureExtractor]] = \
    Registry()
# noinspection DuplicatedCode
ROI_KEYPOINT_PREDICTOR: Registry[str, Callable[[CfgNode, int], ROIKeypointPredictor]] = Registry()

# Mask Prediction
# noinspection DuplicatedCode
ROI_MASK_FEATURE_EXTRACTORS: Registry[str, Callable[[CfgNode, int, AnchorStrides], ROIBoxMaskFeatureExtractor]] = \
    Registry()
ROI_MASK_PREDICTOR: Registry[str, Callable[[CfgNode, tuple[int, ...]], ROIMaskPredictor]] = Registry()

ROI_MASK_HEADS: Registry[str, Callable[[CfgNode, int, AnchorStrides], ROIMaskHead]] = Registry()

# Relation Prediction
ROI_RELATION_FEATURE_EXTRACTORS: Registry[str, Callable[[CfgNode, int], ROIRelationFeatureExtractor]] = Registry()
ROI_RELATION_PREDICTOR: Registry[str, Any] = Registry()
