from abc import ABC, abstractmethod

import torch

from .attribute_head import BoxHeadTargets
from .backbone import FeatureMaps
from .box_head import BoxHeadTrainProposal, BoxHeadTestProposal
from .loss import LossDict
from .region_proposal import RPNProposals
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
            targets: BoxHeadTargets | None = None
    ) -> tuple[list[BoxList], LossDict]:
        raise NotImplementedError

    @abstractmethod
    def sample_and_predict_relation(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None,
            compute_losses: bool,
    ) -> tuple[BoxHeadTestProposal, list[list | None]]:
        """Max-memory-usage-optimized pipeline for relation."""
        raise NotImplementedError

    @abstractmethod
    def postprocess_relation(
            self, proposals: BoxHeadTestProposal, pre_computations: list[list | None]
    ) -> tuple[list[BoxList], LossDict]:
        """Sister method to sample_and_predict_relation."""
        raise NotImplementedError
