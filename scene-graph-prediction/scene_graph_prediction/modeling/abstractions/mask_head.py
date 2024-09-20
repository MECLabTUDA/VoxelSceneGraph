# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .backbone import AnchorStrides
from .box_head import BoxHeadTestProposals, BoxHeadFeatures
from .loss import MaskHeadLossDict
from .region_proposal import FeatureMaps

# noinspection DuplicatedCode
MaskHeadTargets = list[BoxList]  # fields: labels,mask -> torch.Tensor,AbstractMaskList

# ROIMask Feature Extraction
# We use the same extractors as the ROIBoxHeads,
# but we use a different alias to signify that we don't reuse the features produced by the ROIBoxHead
MaskHeadFeatures = torch.Tensor


# ROIMask Feature Extraction
class ROIMaskFeatureExtractor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__()

    @abstractmethod
    def forward(self, x: list[torch.Tensor], proposals: list[BoxList]) -> MaskHeadFeatures:
        raise NotImplementedError


# ROIMask Predictor
MaskLogits = torch.LongTensor


class ROIMaskPredictor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: MaskHeadFeatures) -> MaskLogits:
        raise NotImplementedError


# ROIKeypointHead
# fields: labels,keypoints (from KeypointHeadTargets) + logits  -> torch.Tensor
MaskHeadProposals = list[BoxList]


class ROIMaskHead(torch.nn.Module, ABC):
    """Generic Box Head class."""
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def subsample(
            self,
            proposals: BoxHeadTestProposals
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals."""
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            features: FeatureMaps | BoxHeadFeatures,
            proposals: BoxHeadTestProposals
    ) -> MaskLogits:
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            mask_logits: MaskLogits,
            proposals: BoxHeadTestProposals
    ) -> MaskHeadProposals:
        """Convert attribute logits to actual predictions and store them in the proposals."""
        raise NotImplementedError

    @abstractmethod
    def loss(
            self, mask_logits: MaskLogits, proposals: BoxHeadTestProposals, targets: MaskHeadTargets
    ) -> MaskHeadLossDict:
        """Compute a loss given attribute logits and proposals with relevant ground truth fields."""
        raise NotImplementedError
