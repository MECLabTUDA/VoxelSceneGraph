# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .box_head import BoxHeadTestProposals, BoxHeadFeatures
from .loss import KeypointHeadLossDict
from .region_proposal import FeatureMaps

# noinspection DuplicatedCode
KeypointHeadTargets = list[BoxList]  # fields: labels,keypoints -> torch.Tensor

# ROIKeypoint Feature Extraction
# We use the same extractors as the ROIBoxHeads,
# but we use a different alias to signify that we don't reuse the features produced by the ROIBoxHead
KeypointHeadFeatures = torch.Tensor


# ROIKeypoint Feature Extraction
class ROIKeypointFeatureExtractor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: list[torch.Tensor], proposals: list[BoxList]) -> KeypointHeadFeatures:
        raise NotImplementedError


# ROIKeypoint Predictor
# noinspection DuplicatedCode
KeypointLogits = torch.Tensor


class ROIKeypointPredictor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: KeypointHeadFeatures) -> KeypointLogits:
        raise NotImplementedError


# ROIKeypointHead
# fields: labels,keypoints (from KeypointHeadTargets) + keypoint_logits,keypoints  -> torch.Tensor
KeypointHeadProposals = list[BoxList]


class ROIKeypointHead(torch.nn.Module, ABC):
    """Generic Box Head class."""
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def subsample(
            self,
            proposals: BoxHeadTestProposals,
            targets: KeypointHeadTargets
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: FeatureMaps | BoxHeadFeatures, proposals: BoxHeadTestProposals) -> KeypointLogits:
        """
        :param features: extracted from box_head
        :param proposals: extracted from box_head
        """
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            kp_logits: KeypointLogits,
            proposals: BoxHeadTestProposals
    ) -> KeypointHeadProposals:
        """Convert keypoint logits to actual predictions and store them in the proposals."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, kp_logits: KeypointLogits, proposals: KeypointHeadProposals) -> KeypointHeadLossDict:
        """Compute a loss given attribute logits and proposals with relevant ground truth fields."""
        raise NotImplementedError
