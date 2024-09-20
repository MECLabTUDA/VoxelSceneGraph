# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .attribute_head import AttributeLogits
from .box_head import BoxHeadTestProposals, BoxHeadFeatures, ClassLogits
from .loss import RelationHeadLossDict, LossDict
from .region_proposal import FeatureMaps

# Fields: labels,predict_logits,boxes_per_cls -> torch.Tensor
# Fields if attributes on: attributes,attribute_logits (no gt use)
RelationHeadProposals = BoxHeadTestProposals
RelationHeadTargets = list[BoxList]  # fields: labels,relation -> torch.Tensor

# ROIRelation Feature Extraction
# We use the same extractors as the ROIBoxHeads,
# but we use a different alias to signify that we don't reuse the features produced by the ROIBoxHead
RelationHeadFeatures = torch.Tensor


# Relation Context Extraction
class RelationLSTMContext(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
            self,
            x: BoxHeadFeatures,
            proposals: list[BoxList],
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, AttributeLogits | None]:
        """
        :returns: obj_logits, obj_predictions, edge_ctx, att_logits
        """
        raise NotImplementedError


class RelationContext(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
            self,
            x: BoxHeadFeatures,
            proposals: RelationHeadProposals,
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        :returns: obj_logits, obj_predictions, edge_ctx, bi_predictions
        """
        raise NotImplementedError


# ROIRelation Feature Extraction

class ROIRelationMaskFeatureExtractor(torch.nn.Module, ABC):
    """
    Convert subject+object masks to features that can be added to the ines produced by the ROI box feature extractor.
    I.e. the produced features need to have the same size as the Pooler output.
    The original size of the masks is left open, but must be supplied by the get_orig_rect_size method.
    I.e. we could use masks that are 2, 4, 8... times bigger than the Pooler output
    """
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, out_channels: int):
        # in_channels will always be 2 (one mask for the subject and one for the object)
        super().__init__()

    @abstractmethod
    def get_orig_rect_size(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, masks: torch.FloatTensor) -> torch.FloatTensor:
        """Features of shape N x POOLER_RESOLUTION_DEPTH x POOLER_RESOLUTION x POOLER_RESOLUTION"""
        raise NotImplementedError


class ROIRelationFeatureExtractor(torch.nn.Module, ABC):
    n_dim: int
    representation_size: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(
            self,
            x: FeatureMaps,
            proposals: list[BoxList],
            rel_pair_idxs: list[torch.LongTensor]
    ) -> RelationHeadFeatures:
        raise NotImplementedError


# ROIRelation Predictor
RelationLogits = torch.LongTensor

T = TypeVar("T")


class ROIRelationPredictor(torch.nn.Module, ABC, Generic[T]):
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            roi_features: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list[T]]:
        """
        :returns:
            obj_dists: logits of object label distribution
            rel_dists: logits of relation label distribution
            att_dists: optionally, logits of attributes label distribution (if attribute_on)
            add_losses_required: optionally, any prediction that is required to compute additional losses for the method
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_attribute_refinement() -> bool:
        """
        Whether this predictor can do attribute refinement.
        This allows us to know whether attribute features are only used for prediction or also for refinement.
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def extra_losses(
            self, add_losses_required: list[T], rel_binaries: list[torch.LongTensor], rel_labels: list[torch.LongTensor]
    ) -> LossDict:
        """Compute extra loss terms for this predictor."""
        return {}


# ROIRelationHead
RelationHeadTrainProposal = BoxHeadTestProposals  # No extra fields
# "pred_labels" and "pred_scores" (and "pred_attributes") are updated with enhanced predictions,
# "rel_pair_idxs", "pred_rel_scores", and "pred_rel_labels" are added
RelationHeadTestProposals = list[BoxList]


class ROIRelationHead(torch.nn.Module, ABC):
    """Generic Relation Head class."""
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__()

    @abstractmethod
    def subsample_relation_pairs(
            self,
            proposals: RelationHeadProposals,
            targets: RelationHeadTargets
    ) -> tuple[list[torch.LongTensor], list[torch.LongTensor], list[torch.LongTensor]]:
        """
        Sample relation pairs for training and compute binary relatedness matrices.
        :returns:
                  rel_pair_idxs: list of relation pairs (one tensor per image)
                  rel_labels: list of relation labels (one tensor per image)
                  rel_binaries: list of symmetric binary matrices (one tensor per image), i.e. are two objects related
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_relation_pairs(self, proposals: RelationHeadProposals) -> list[torch.LongTensor]:
        """
        Prepare relation pairs for prediction (testing).
        :returns: list of relation pair indexes for each proposal.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            features: FeatureMaps,
            rel_pair_idxs: list[torch.LongTensor],
            proposals: RelationHeadProposals
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            rel_pair_idxs: list[torch.LongTensor],
            refined_obj_logits: list[ClassLogits],
            relation_logits: list[RelationLogits],
            refined_att_logits: list[AttributeLogits] | None,
            proposals: RelationHeadProposals
    ) -> RelationHeadTestProposals:
        """Convert logits to actual predictions and store them in the proposals."""
        raise NotImplementedError

    @abstractmethod
    def loss(
            self,
            refined_obj_logits: list[ClassLogits],
            relation_logits: list[RelationLogits],
            refined_att_logits: list[AttributeLogits] | None,
            add_losses_required: list,
            proposals: RelationHeadProposals,
            rel_binaries: list[torch.LongTensor],
            rel_labels: list[torch.LongTensor]
    ) -> RelationHeadLossDict:
        """
        Compute a loss given attribute logits and proposals with relevant ground truth fields.
        Also allow for predictor-specific loss terms using values stored in add_losses_required.
        """
        raise NotImplementedError
