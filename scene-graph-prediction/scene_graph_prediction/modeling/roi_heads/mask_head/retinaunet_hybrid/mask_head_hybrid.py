# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.lls -l

import torch
from yacs.config import CfgNode

from ..default import ROIMaskHead
from .inference import build_roi_mask_hybrid_post_processor
from ....abstractions.backbone import AnchorStrides
from ....abstractions.box_head import BoxHeadTestProposals


class ROIMaskHeadHybrid(ROIMaskHead):
    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels, anchor_strides)
        self.post_processor = build_roi_mask_hybrid_post_processor(cfg)
        self.num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1

    def subsample(
            self,
            proposals: BoxHeadTestProposals
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals."""
        # During training, only focus on positive boxes which are not unique
        # noinspection PyTypeChecker
        return [
            torch.logical_and(proposal.LABELS > 0, proposal.LABELS <= self.num_normal_fg_classes)
            for proposal in proposals
        ]
