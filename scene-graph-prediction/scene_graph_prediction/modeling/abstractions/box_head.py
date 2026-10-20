# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .backbone import AnchorStrides
from .loss import BoxHeadLossDict
from .region_proposal import FeatureMaps, RPNProposals
from ..utils.pooler import Pooler

BoxHeadTargets = list[BoxList]  # fields: labels,attributes -> torch.LongTensor

# ROIBox Feature Extraction
BoxHeadFeatures = torch.Tensor


class ROIBoxFeatureExtractor(torch.nn.Module, ABC):
    """
    Feature extractors for the box head.
    Sometimes, we want to do the pooling separately, so we have the forward_without_pool to do feature refinement
    on features of the right size.
    Additionally, we have two kinds of feature extractors:
    - those who use convolutions and preserve the shape of the features.
    - those which flatten features
    Only the first ones are compatible with the mask head.
    However, they require some average pooling to convert the output with representation_size channels to a final flat
    representation of representation_size variables. This is not suitable for the mask head.
    So this has to be done by the mask head automatically if needed.
    Note: the ability to share the feature extractor between the box head and other heads causes a lot of constraints...
    """

    n_dim: int
    representation_size: int
    pooler: Pooler
    # Whether we need to do some average pooling on the output
    is_mask_head_compatible: bool = False

    # noinspection PyUnusedLocal
    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            half_out: bool = False,
            cat_all_levels: bool = False
    ):
        """
        :param half_out: tells the feature extractor to only produce half of the features that are configured.
                         (because features from attributes will fill the other half)
        :param cat_all_levels: option for when this extractor is used for relation prediction.
        """
        super().__init__()
        self.cfg = cfg.clone()

    def forward(self, x: FeatureMaps, proposals: list[BoxList]) -> BoxHeadFeatures:
        """
        Use a pooler to convert the list of features from the RPN to a Tensor then uses self.forward_without_pool.
        I.e. takes care of ROIAlign-ing object features.
        """
        x = self.pooler(x, proposals)
        return self.forward_without_pool(x)

    @abstractmethod
    def forward_without_pool(self, x: torch.Tensor) -> BoxHeadFeatures:
        """
        This method needs to exist for the relation head,
        as we want to compute features for union boxes without pooling.
        """
        raise NotImplementedError

    @abstractmethod
    def output_size(self) -> tuple[int, ...]:
        """
        Size of the output of the feature extractor. Can be different from the representation size.
        For extractors that are not mask head compatible, the output should be (self.representation_size,).
        Otherwise, it will be of the form Cx(Dx)HxW.
        """
        raise NotImplementedError


# ROIRelation Feature Extraction
class ROIBoxMaskFeatureExtractor(torch.nn.Module, ABC):
    """
    Extractor for mask-based features.
    While the input can be of any size, the resulting output should be the same size as the Pooler's output.
    That way, we can merge this output, and the one from the Pooler easily, before continuing feature extraction.
    in_channels is one for the box head (one mask per object) and 2 for the relation head (subject and object masks).
    """
    n_dim: int

    # noinspection PyUnusedLocal
    def __init__(self, cfg: CfgNode, in_channels: int, out_channels: int):
        super().__init__()

    @abstractmethod
    def input_size(self) -> tuple[int, ...]:
        """Size of the input."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, masks: torch.FloatTensor) -> torch.FloatTensor:
        """Features of shape N x POOLER_RESOLUTION_DEPTH x POOLER_RESOLUTION x POOLER_RESOLUTION"""
        raise NotImplementedError


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
    def assign_label_to_proposals(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
        """
        Perform box-wise sampling.
        Assign ground truth LABELS and MATCHED_IDXS to proposals for any roi head that does not do its own sampling,
        e.g. mask, attribute heads. Each head can then index relevant fields using MATCHED_IDXS and the targets.
        Relation heads also rely on this sampling, but this gets computed AFTER the final box prediction.
        Note: the resulting mask may get ignored, as only the GT labels matching are needed.
        Note: for relations, we have to assign labels even when testing because of evaluation methods
        replacing predictions with GT for further insights into model limitations.
        :returns: a mask of the sampled boxes.
        """
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

    @abstractmethod
    def loss(
            self, class_logits: ClassLogits, box_regression: BboxRegression, proposals: BoxHeadTestProposals
    ) -> BoxHeadLossDict:
        """Compute a loss given the predicted logits and proposals with relevant ground truth fields."""
        raise NotImplementedError

    @abstractmethod
    def require_one_stage_detector(self) -> bool:
        """
        We're now implementing box heads that require having a one-stage detector.
        E.g. we don't allow reclassification, but we want to refine boxes or remove false positives.
        So we need this information to make sure that the config works.
        """
        raise NotImplementedError
