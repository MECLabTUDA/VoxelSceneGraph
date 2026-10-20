from abc import ABC, abstractmethod

import torch

from .attribute_head import BoxHeadTargets
from .backbone import FeatureMaps
from .box_head import BoxHeadTrainProposal, BoxHeadTestProposal, BoxHeadTestProposals
from .loss import LossDict
from .region_proposal import RPNProposals
from ..utils.misc import LossComputationCfg
from ...structures import BoxList


class CombinedROIHeads(torch.nn.ModuleDict, ABC):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.
    Note: can contain 0 head (no op).
    """

    @abstractmethod
    def forward(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None = None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[list[BoxList], LossDict]:
        raise NotImplementedError

    @abstractmethod
    def sample_and_predict_roi_heads(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None
    ) -> tuple[
        list[torch.Tensor | BoxList | None],
        list[torch.Tensor | BoxList | None],
        list[torch.Tensor | BoxList | None],
        list[torch.Tensor | BoxList | None]
    ]:
        """Max-memory-usage-optimized training pipeline for roi heads (except relation)."""
        raise NotImplementedError

    @abstractmethod
    def postprocess_roi_heads(
            self,
            proposals: BoxHeadTestProposal,
            box_pre_computations: list[torch.Tensor | list[BoxList]] | None,
            attr_pre_computations: list[torch.Tensor | list[BoxList]] | None,
            mask_pre_computations: list[torch.Tensor | list[BoxList]] | None,
            kp_pre_computations: list[torch.Tensor | list[BoxList]] | None,
    ) -> LossDict:
        """
        Sister method to sample_and_predict_roi_heads.
        If the required elements to compute the loss are None, then no loss should be computed.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_and_predict_relation(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[BoxHeadTestProposal, list[list | None]]:
        """Max-memory-usage-optimized training pipeline for relation prediction."""
        raise NotImplementedError

    @abstractmethod
    def postprocess_relation(
            self, proposals: BoxHeadTestProposal, pre_computations: list[list | None]
    ) -> tuple[list[BoxList], LossDict]:
        """
        Sister method to sample_and_predict_relation.
        If the required elements to compute the loss are None, then no loss should be computed.
        """
        raise NotImplementedError

    @abstractmethod
    def _add_roi_heads_gt_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]) -> BoxHeadTestProposals:
        """
        Add groundtruth boxes / masks with labels to the proposals.
        # TODO also add keypoints / attributes
        Note: useful when training a downstream relation head.
        """
        raise NotImplementedError

    @abstractmethod
    def _replace_proposals_with_gt(self, proposals: list[BoxList], targets: list[BoxList]) -> list[BoxList]:
        """
        For the Scene Graph prediction task, we may want to have a fine control over
        which parts of the prediction are replaced with groundtruth annotation during testing.
        This way, we can find out which parts of the network cause the biggest performance degradation.
        """
        raise NotImplementedError
