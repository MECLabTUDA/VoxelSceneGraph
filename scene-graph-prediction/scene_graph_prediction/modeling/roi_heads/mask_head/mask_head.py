# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from .inference import build_roi_mask_post_processor
from .loss import build_roi_mask_loss_evaluator
from .roi_mask_feature_extractors import build_roi_mask_feature_extractor
from .roi_mask_predictors import build_roi_mask_predictor
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadFeatures, BoxHeadTestProposals
from ...abstractions.loss import MaskHeadLossDict
from ...abstractions.mask_head import MaskHeadTargets, ROIMaskHead as AbstractROIMaskHead, MaskHeadProposals, MaskLogits


class ROIMaskHead(AbstractROIMaskHead):
    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)
        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM
        self.feature_extractor = build_roi_mask_feature_extractor(cfg, in_channels, anchor_strides)
        self.predictor = build_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = build_roi_mask_post_processor(cfg)
        self.loss_evaluator = build_roi_mask_loss_evaluator(cfg)
        assert self.feature_extractor.n_dim == self.n_dim
        assert self.post_processor.n_dim == self.n_dim

    def subsample(
            self,
            proposals: BoxHeadTestProposals
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals."""
        # During training, only focus on positive boxes
        return [proposal.LABELS > 0 for proposal in proposals]

    def forward(
            self,
            features: FeatureMaps | BoxHeadFeatures,
            proposals: BoxHeadTestProposals
    ) -> MaskLogits:
        """
        Note: regarding BoxList fields:
        - proposals need the "pred_labels" field during testing
        - targets need the "labels", and "masks" fields
        - during training: no field is modified
        - during testing: "pred_masks" is added
        """
        if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # In this case, features is a tensor as the pooler in the BoxHead already reduces the feature maps
            x = features
        else:
            # In this case, features is a list of tensors and will be reduced by the pooler in the extractor
            x: torch.Tensor = self.feature_extractor(features, proposals)

        mask_logits = self.predictor(x)
        return mask_logits

    def post_process_predictions(
            self,
            mask_logits: MaskLogits,
            proposals: BoxHeadTestProposals
    ) -> MaskHeadProposals:
        """Convert attribute logits to actual predictions and store them in the proposals."""
        result = self.post_processor(mask_logits, proposals)
        return result

    def loss(
            self, mask_logits: MaskLogits, proposals: BoxHeadTestProposals, targets: MaskHeadTargets
    ) -> MaskHeadLossDict:
        """Compute a loss given attribute logits and proposals with relevant ground truth fields."""
        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
        return {"loss_mask": loss_mask}
