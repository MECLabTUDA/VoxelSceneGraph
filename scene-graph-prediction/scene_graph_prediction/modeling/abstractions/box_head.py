# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .region_proposal import FeatureMaps, RPNProposals
from ..utils.pooler import Pooler

BoxHeadTargets = list[BoxList]  # fields: labels,attributes -> torch.LongTensor

# ROIBox Feature Extraction
BoxHeadFeatures = torch.Tensor


class ROIBoxFeatureExtractor(torch.nn.Module, ABC):
    n_dim: int
    representation_size: int
    pooler: Pooler

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int, half_out: bool = False, cat_all_levels: bool = False):
        """
        :param half_out: tells the feature extractor to only produce half of the features that are configured.
                         (because features from attributes will fill the other half)
        :param cat_all_levels: option for when this extractor is used for relation prediction.
        """
        super().__init__()

    def forward(self, x: FeatureMaps, proposals: list[BoxList]) -> BoxHeadFeatures:
        """
        Use a pooler to convert the list of features from the RPN to a Tensor then uses self.forward_without_pool.
        I.e. takes care of ROIAlign-ing object features.
        """
        x = self.pooler(x, proposals)
        return self.forward_without_pool(x)

    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        """
        This method needs to exist for the relation head,
        as we want to compute features for union boxes without pooling.
        """
        raise RuntimeError


# ROIBox Predictor
ClassLogits = torch.Tensor
BboxRegression = torch.Tensor


class ROIBoxPredictor(torch.nn.Module, ABC):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[ClassLogits, BboxRegression]:
        raise NotImplementedError


# ROIBox PostProcessing
BoxHeadTrainProposal = BoxHeadTargets  # extra field: matched_idxs, pred_logits (internal use)
# fields: matched_idxs,pred_scores,pred_label,boxes_per_cls (,pred_logits: internal use) -> torch.Tensor
BoxHeadTestProposal = BoxList
BoxHeadTestProposals = list[BoxHeadTestProposal]


# ROIBoxHead
class ROIBoxHead(torch.nn.Module, ABC):
    """Generic Box Head class."""
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def subsample(
            self,
            proposals: RPNProposals,
            targets: BoxHeadTargets
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals: add groundtruth fields to proposals and return sampling mask."""
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            features: FeatureMaps,
            proposals: RPNProposals
    ) -> tuple[BoxHeadFeatures, ClassLogits, BboxRegression]:
        """
        Notes on fields added to the proposals:
        - proposals need the field "objectness"
        - targets need the fields "labels", and "attributes"
        - during training: "labels", "attributes", "matched_idxs" (internal use); "pred_logits"
        - during testing: "matched_idxs" (internal use); "pred_logits",
                          "pred_scores", "pred_labels", "boxes_per_cls" (#nms, #cls, 4/8 2D/3D)
        Note: adding the field "matched_idxs" to the proposals to avoid
              having to match proposals with targets in other ROI heads.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            x: BoxHeadFeatures,
            class_logits: ClassLogits,
            box_regression: BboxRegression,
            proposals: BoxHeadTestProposals
    ) -> tuple[BboxRegression, BoxHeadTestProposals]:
        """
        Convert class logits and regressions to actual predictions and store them in the proposals.
        Since NMS is applied, we also need to sample the BoxHeadFeatures.
        """
        raise NotImplementedError
