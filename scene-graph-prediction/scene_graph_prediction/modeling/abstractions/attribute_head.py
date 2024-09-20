# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .box_head import BoxHeadTestProposals
from .loss import AttributeHeadLossDict
from .region_proposal import FeatureMaps

BoxHeadTargets = list[BoxList]  # fields: labels,attributes -> torch.Tensor

# ROIAttribute Feature Extraction
# We use the same extractors as the ROIBoxHeads,
# but we use a different alias to signify that we don't reuse the features produced by the ROIBoxHead
AttributeHeadFeatures = torch.Tensor

# ROIAttribute Predictor
# noinspection DuplicatedCode
AttributeLogits = torch.Tensor


class ROIAttributePredictor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: AttributeHeadFeatures) -> AttributeLogits:
        raise NotImplementedError


# ROIAttributeHead
# fields: labels,attributes (from BoxHeadFilteredProposals) + attribute_logits -> torch.Tensor
AttributeHeadProposals = list[BoxList]


class ROIAttributeHead(torch.nn.Module, ABC):
    """Generic Box Head class."""
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, features: FeatureMaps, proposals: BoxHeadTestProposals) -> AttributeLogits:
        """
        :param features: extracted from box_head
        :param proposals: extracted from box_head
        """
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            attribute_logits: AttributeLogits,
            proposals: BoxHeadTestProposals
    ) -> AttributeHeadProposals:
        """Convert attribute logits to actual predictions and store them in the proposals."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, attribute_logits: AttributeLogits, proposals: BoxHeadTestProposals) -> AttributeHeadLossDict:
        """Compute a loss given attribute logits and proposals with relevant ground truth fields."""
        raise NotImplementedError
