# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.backbone import AnchorStrides
from scene_graph_prediction.modeling.abstractions.box_head import FeatureMaps, RPNProposals, BoxHeadTestProposals, \
    BoxHeadFeatures, \
    BoxHeadTargets, ROIBoxHead as AbstractROIBoxHead, ClassLogits, BboxRegression
from scene_graph_prediction.modeling.abstractions.loss import BoxHeadLossDict
from scene_graph_prediction.modeling.roi_heads._utils import split_logits_for_each_image
from scene_graph_prediction.modeling.roi_heads.box_head.roi_box_feature_extractors import build_feature_extractor
from scene_graph_prediction.modeling.roi_heads.box_head.roi_box_predictors import build_roi_box_predictor
from scene_graph_prediction.structures import BoxList
from .inference import build_roi_box_postprocessor
from .loss import build_roi_box_loss_evaluator
from .sampling import build_roi_box_samp_processor


class ROIBoxHead(AbstractROIBoxHead):
    """
    Generic Box Head class.

    Note: in training mode, ground truth attribute labels (contained in fields in the targets)
           are copied to the sampled input BoxLists.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)
        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM
        self.feature_extractor = build_feature_extractor(
            cfg,
            in_channels,
            anchor_strides,
            half_out=self.cfg.MODEL.ATTRIBUTE_ON
        )
        self.predictor = build_roi_box_predictor(cfg, self.feature_extractor.representation_size)
        self.post_processor = build_roi_box_postprocessor(cfg)
        self.loss_evaluator = build_roi_box_loss_evaluator(cfg)
        self.samp_processor = build_roi_box_samp_processor(cfg, self.loss_evaluator.regression_loss.require_box_coding)

        assert self.feature_extractor.n_dim == self.n_dim
        assert self.predictor.n_dim == self.n_dim
        assert self.post_processor.n_dim == self.n_dim
        assert self.loss_evaluator.n_dim == self.n_dim

    def subsample(
            self,
            proposals: RPNProposals,
            targets: BoxHeadTargets
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals: add groundtruth fields to proposals and return sampling mask."""
        if len(proposals) == 0:
            return []

        with torch.no_grad():
            # Adds the fields "labels", "attributes" (, and "regression_targets" for head-loss computation)
            return self.samp_processor.subsample(proposals, targets)

    def forward(
            self,
            features: FeatureMaps,
            proposals: RPNProposals
    ) -> tuple[BoxHeadFeatures, ClassLogits, BboxRegression]:
        # Extract features that will be fed to the final classifier.
        # The feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # Final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        return x, class_logits, box_regression

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
        # Adds the field "pred_logits"
        split_logits_for_each_image(proposals, BoxList.PredictionField.PRED_LOGITS, class_logits)

        # Adds the fields "pred_scores", "pred_labels", "boxes_per_cls"
        x, results = self.post_processor(x, class_logits, box_regression, proposals)
        return x, results

    def loss(
            self, class_logits: ClassLogits, box_regression: BboxRegression, proposals: BoxHeadTestProposals
    ) -> BoxHeadLossDict:
        """Compute a loss given the predicted logits and proposals with relevant ground truth fields."""
        loss_classifier, loss_box_reg = self.loss_evaluator(class_logits, box_regression, proposals)
        return {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}


class ROIRelationReadyBoxHead(ROIBoxHead):
    """
    Generic Box Head class.

    Note: in training mode, ground truth labels (contained in fields in the targets)
           are copied to the sampled input BoxLists.
    """

    def subsample(
            self,
            proposals: RPNProposals,
            targets: BoxHeadTargets
    ) -> list[torch.BoolTensor]:
        """We don't actually need to sample for Relation training, but we need to follow the interface."""
        # Note: all handling of GT boxes for relation training is already done by the RPN
        #       At most, we only need to produce features

        # We need to assign labels to proposals when not using GT boxes
        # So, we call self.samp_processor.subsample to set fields, but then discard the sampling mask
        if self.training and not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.samp_processor.assign_label_to_proposals(proposals, targets)

        device = torch.device(self.cfg.MODEL.DEVICE)
        return [torch.ones(len(prop), device=device, dtype=torch.bool) for prop in proposals]

    def post_process_predictions(
            self,
            x: BoxHeadFeatures,
            class_logits: ClassLogits,
            box_regression: BboxRegression,
            proposals: BoxHeadTestProposals
    ) -> tuple[BboxRegression, BoxHeadTestProposals]:
        # Adds the field "pred_logits"
        split_logits_for_each_image(proposals, BoxList.PredictionField.PRED_LOGITS, class_logits)
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            return x, proposals

        # Post process:
        # Adds the fields "pred_scores", "pred_labels", "boxes_per_cls"
        x, result = self.post_processor(x, class_logits, box_regression, proposals)
        return x, result

    def loss(
            self, class_logits: ClassLogits, box_regression: BboxRegression, proposals: BoxHeadTestProposals
    ) -> BoxHeadLossDict:
        raise RuntimeError("No BoxHead loss should be computed when training a RelationHead.")
