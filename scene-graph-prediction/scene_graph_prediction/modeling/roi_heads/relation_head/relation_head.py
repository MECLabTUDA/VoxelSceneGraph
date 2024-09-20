# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from .inference import build_roi_relation_post_processor
from .loss import build_roi_relation_loss_evaluator
from .roi_relation_feature_extractors import build_roi_relation_feature_extractor
from .roi_relation_predictors import build_roi_relation_predictor
from .sampling import build_roi_relation_samp_processor
from ..box_head.roi_box_feature_extractors import build_feature_extractor
from ...abstractions.attribute_head import AttributeHeadFeatures, AttributeLogits
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadFeatures, ClassLogits
from ...abstractions.loss import RelationHeadLossDict
from ...abstractions.relation_head import ROIRelationHead as AbstractROIRelationHead, RelationHeadTargets, \
    RelationHeadFeatures, RelationHeadProposals, RelationHeadTestProposals, RelationLogits
from ...utils import ROIHeadName


class ROIRelationHead(AbstractROIRelationHead):
    """Generic Relation Head class."""

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)
        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM
        # Same structure as the box head but with different parameters
        # These params will be trained with a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in union_feature_extractor

        self.union_feature_extractor = build_roi_relation_feature_extractor(cfg, in_channels, anchor_strides)
        if cfg.MODEL.ATTRIBUTE_ON:
            # Use half of the feature space for the box-feat extractor and the other for attr-feat extractor
            # Note: this is independent from whether the predictor supports attribute refinement
            self.box_feature_extractor = build_feature_extractor(cfg, in_channels, anchor_strides, half_out=True)
            self.att_feature_extractor = build_feature_extractor(cfg, in_channels, anchor_strides, half_out=True,
                                                                 roi_head=ROIHeadName.Attribute)
            feat_dim = self.box_feature_extractor.representation_size * 2
        else:
            self.box_feature_extractor = build_feature_extractor(cfg, in_channels, anchor_strides)
            feat_dim = self.box_feature_extractor.representation_size
        assert self.box_feature_extractor.n_dim == self.n_dim
        self.predictor = build_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = build_roi_relation_post_processor(cfg, self.predictor.supports_attribute_refinement())
        self.loss_evaluator = build_roi_relation_loss_evaluator(cfg, self.predictor.supports_attribute_refinement())
        self.samp_processor = build_roi_relation_samp_processor(cfg)

        assert self.union_feature_extractor.n_dim == self.n_dim
        assert self.box_feature_extractor.n_dim == self.n_dim

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
        # Relation sub-sampling and assign ground truth label during training
        with torch.no_grad():
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                rel_labels, rel_pair_idxs, rel_binaries = self.samp_processor.gtbox_relation_sample(proposals, targets)
            else:
                rel_labels, rel_pair_idxs, rel_binaries = self.samp_processor.detect_relation_sample(proposals, targets)

        return rel_pair_idxs, rel_labels, rel_binaries

    def prepare_relation_pairs(self, proposals: RelationHeadProposals) -> list[torch.LongTensor]:
        """
        :returns: list of relation pair indexes for each proposal.
        """
        if len(proposals) == 0:
            return []

        rel_pair_idxs = self.samp_processor.prepare_test_pairs(proposals)
        return rel_pair_idxs

    def forward(
            self,
            features: FeatureMaps,
            rel_pair_idxs: list[torch.LongTensor],
            proposals: RelationHeadProposals
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:
        """
        Note: regarding BoxList fields:
        - proposals needs the "labels" (during training), and "boxes_per_cls" fields (and "Attributes" if they are on)
        - targets needs the "labels", and "relation" fields
        - during training: no field is modified
        - during testing: "pred_labels" and "pred_scores" (and "pred_attributes") are updated with enhanced predictions,
                          "rel_pair_idxs", "pred_rel_scores", and "pred_rel_labels" are added
        Note: the RelationHead is responsible to select the proper field (predicted or gt) based on the evaluation mode.
        """
        # Use box_head to extract features that will be fed to the later predictor processing
        roi_features: BoxHeadFeatures = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features: AttributeHeadFeatures = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        union_features: RelationHeadFeatures = self.union_feature_extractor(features, proposals, rel_pair_idxs)

        # Final classifier that converts the features into predictions
        # Should correspond to all the functions and layers after the self.context class
        refined_obj_logits, relation_logits, refined_att_logits, add_losses_required = self.predictor(
            proposals, rel_pair_idxs, roi_features, union_features
        )

        return refined_obj_logits, relation_logits, refined_att_logits, add_losses_required

    def post_process_predictions(
            self,
            rel_pair_idxs: list[torch.LongTensor],
            refined_obj_logits: list[ClassLogits],
            relation_logits: list[RelationLogits],
            refined_att_logits: list[AttributeLogits] | None,
            proposals: RelationHeadProposals
    ) -> RelationHeadTestProposals:
        """Convert logits to actual predictions and store them in the proposals."""
        result = self.post_processor(relation_logits, refined_obj_logits, rel_pair_idxs, proposals, refined_att_logits)
        return result

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
        loss_relation, loss_refine_obj, loss_refine_att = self.loss_evaluator(
            proposals, rel_labels, relation_logits, refined_obj_logits, refined_att_logits
        )

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL or \
                self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            # Ignore this loss
            with torch.no_grad():
                loss_refine_obj.fill_(0)

        output_losses = {
            "loss_rel": loss_relation,
            "loss_refine_obj": loss_refine_obj,
            **self.predictor.extra_losses(add_losses_required, rel_binaries, rel_labels)
        }

        if self.cfg.MODEL.ATTRIBUTE_ON:
            output_losses["loss_refine_att"] = loss_refine_att

        return output_losses
