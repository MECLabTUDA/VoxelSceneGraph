# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.backbone import AnchorStrides
from scene_graph_prediction.modeling.abstractions.box_head import FeatureMaps, RPNProposals, BoxHeadTestProposals, \
    BoxHeadFeatures, BoxHeadTargets, ROIBoxHead as AbstractROIBoxHead, ClassLogits, BboxRegression
from scene_graph_prediction.modeling.abstractions.loss import BoxHeadLossDict
from scene_graph_prediction.modeling.utils import BalancedSampler
from scene_graph_prediction.structures import BoxList
from .loss import build_roi_box_loss_evaluator_hybrid
from ..default.inference import build_roi_box_postprocessor
from ..roi_box_feature_extractors import build_feature_extractor
from ..roi_box_predictors import build_roi_box_predictor
from ..._utils import split_logits_for_each_image


class ROIBoxHeadHybrid(AbstractROIBoxHead):
    """
    Box Head designed to work with a one-stage hybrid Retina U-Net.
    Works as a regular Box Head, but objects detected from segmentation do not get reclassified;
    only their bounding box gets refined.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        assert cfg.MODEL.REGION_PROPOSAL == "RetinaUNetHybrid" and not cfg.MODEL.RETINANET.TWO_STAGE
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

        self.num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1

        # As to why we do not need a custom postprocessor:
        # - We assume that the score for the segmentation will be very high (so always above threshold)
        # - For unique objects, the logits for other classes (than the predicted one) will be very low
        self.post_processor = build_roi_box_postprocessor(cfg)
        # Note: since we have no sampling, we have to rely on the REGRESSION_TARGETS field set by the first detector.
        #       This also mean that we have to assume that both use the same regression loss
        #       (or that they both need the targets tp be encoded / decoded)
        self.loss_evaluator = build_roi_box_loss_evaluator_hybrid(cfg)
        self.sampler = BalancedSampler(
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        )

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

        # The matching has already been performed by the one-stage hybrid Retina U-Net, we actually only have to sample
        labels = [proposal.LABELS for proposal in proposals]
        sampled_pos_mask, sampled_neg_mask = self.fg_bg_sampler(labels)
        # Just make sure that unique objects are always sampled
        return [
            pos_mask_img | neg_mask_img | (proposal.LABELS[pos_mask_img] > self.num_normal_fg_classes)
            for pos_mask_img, neg_mask_img, proposal in zip(sampled_pos_mask, sampled_neg_mask, proposals)
        ]

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

        # Update logits for relevant boxes (logits for unique boxes must remain untouched)
        retina_logits = torch.cat([proposal.PRED_LOGITS for proposal in proposals])
        normal_boxes = torch.cat([proposal.PRED_LABELS <= self.num_normal_fg_classes for proposal in proposals])
        retina_logits[normal_boxes] += class_logits[normal_boxes]

        return x, retina_logits, box_regression

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


class ROIRelationReadyBoxHeadHybrid(ROIBoxHeadHybrid):
    """
    Box Head designed to work with a hybrid Retina U-Net, but for relations.
    Works as a regular Box Head, but objects detected from segmentation do not get reclassified;
    only their bounding box gets refined.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        assert cfg.MODEL.REGION_PROPOSAL == "RetinaUNetHybrid" and not cfg.MODEL.RETINANET.TWO_STAGE

        super().__init__(cfg, in_channels, anchor_strides)
        self.num_unique_fg_classes = cfg.INPUT.N_UNIQUE_OBJ_CLASSES
        self.num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - self.num_unique_fg_classes - 1

        self.post_processor = build_roi_box_postprocessor(cfg)

    def subsample(
            self,
            proposals: RPNProposals,
            targets: BoxHeadTargets
    ) -> list[torch.BoolTensor]:
        """We don't actually need to sample for Relation training, but we need to follow the interface."""
        # All work has already been done
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
