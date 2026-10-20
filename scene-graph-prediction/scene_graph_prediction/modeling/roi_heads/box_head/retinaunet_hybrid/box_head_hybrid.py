# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.backbone import AnchorStrides
from scene_graph_prediction.modeling.abstractions.box_head import RPNProposals, BoxHeadTestProposals, \
    BoxHeadFeatures, BoxHeadTargets, ClassLogits, BboxRegression, ROIBoxHead as AbstractROIBoxHead
from scene_graph_prediction.modeling.abstractions.loss import BoxHeadLossDict
from scene_graph_prediction.modeling.utils import BoxCoder
from .inference import build_hybrid_roi_box_postprocessor
from .loss import build_roi_box_loss_evaluator_hybrid
from .sampling import build_hybrid_roi_box_samp_processor
from ..default import ROIBoxHead
from ..roi_box_feature_extractors import build_feature_extractor
from ..roi_box_predictors import build_roi_box_predictor
from scene_graph_prediction.modeling.utils.label_assignment import assign_label_to_proposals_always_match_special


class ROIBoxHeadHybrid(ROIBoxHead):
    """
    Box Head designed to work with a one-stage hybrid Retina U-Net.
    Works as a regular Box Head, but objects detected from segmentation do not get reclassified:
    - non-unique objects get a new classification score to remove false positives
    - their bounding box gets refined
    Note: ignores MODEL.CLS_AGNOSTIC_BBOX_REG.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        assert cfg.MODEL.META_ARCHITECTURE == "HybridRetinaUNet" and not cfg.MODEL.RETINANET.TWO_STAGE, "Incompatible?"
        assert not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG, "Not implemented yet"

        # We don't want to init sampler and processors from the base class
        AbstractROIBoxHead.__init__(self, cfg, in_channels)
        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM
        self.num_fg_classes = cfg.INPUT.N_OBJ_CLASSES - 1
        self.num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1

        self.feature_extractor = build_feature_extractor(
            cfg, in_channels, anchor_strides, half_out=self.cfg.MODEL.ATTRIBUTE_ON
        )

        # We'll do things differently in the post-processing / loss computation
        self.predictor = build_roi_box_predictor(cfg, self.feature_extractor.representation_size)

        box_coder = BoxCoder(weights=(1.,) * self.n_dim + (1.,) * self.n_dim, n_dim=self.n_dim)
        self.post_processor = build_hybrid_roi_box_postprocessor(cfg, box_coder)
        self.loss_evaluator = build_roi_box_loss_evaluator_hybrid(cfg, box_coder)
        self.samp_processor = build_hybrid_roi_box_samp_processor(
            cfg, box_coder, self.loss_evaluator.regression_loss.require_box_coding
        )

        if self.feature_extractor.is_mask_head_compatible:
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) if self.n_dim == 2 else torch.nn.AdaptiveAvgPool3d(1)

        assert self.feature_extractor.n_dim == self.n_dim
        assert self.predictor.n_dim == self.n_dim
        assert self.post_processor.n_dim == self.n_dim
        assert self.loss_evaluator.n_dim == self.n_dim

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
        # We don't set the PRED_LOGITS field because it would overwrite the one set by the one-stage detector
        # Instead we just update it in the post-processor
        x, results = self.post_processor(x, class_logits, box_regression, proposals)
        return x, results

    def require_one_stage_detector(self) -> bool:
        return True


class ROIRelationReadyBoxHeadHybrid(ROIBoxHeadHybrid):
    """
    Box Head designed to work with a hybrid Retina U-Net, but for relations.
    Works as a regular Box Head, but objects detected from segmentation do not get reclassified;
    only their bounding box gets refined.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        assert cfg.MODEL.REGION_PROPOSAL == "RetinaUNetHybrid" and not cfg.MODEL.RETINANET.TWO_STAGE
        super().__init__(cfg, in_channels, anchor_strides)

    def assign_label_to_proposals(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
        """We don't actually need to sample for Relation training, but we need to follow the interface."""
        # Assign labels like a hybrid Retina U-Net
        assign_label_to_proposals_always_match_special(
            proposals, targets, self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, self.num_normal_fg_classes
        )

        # Return dummy mask
        device = torch.device(self.cfg.MODEL.DEVICE)
        # noinspection PyTypeChecker
        return [torch.ones(len(prop), device=device, dtype=torch.bool) for prop in proposals]

    def post_process_predictions(
            self,
            x: BoxHeadFeatures,
            class_logits: ClassLogits,
            box_regression: BboxRegression,
            proposals: BoxHeadTestProposals
    ) -> tuple[BboxRegression, BoxHeadTestProposals]:
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            return x, proposals
        return super().post_process_predictions(x, class_logits, box_regression, proposals)

    def loss(
            self, class_logits: ClassLogits, box_regression: BboxRegression, proposals: BoxHeadTestProposals
    ) -> BoxHeadLossDict:
        raise RuntimeError("No BoxHead loss should be computed when training a RelationHead.")
