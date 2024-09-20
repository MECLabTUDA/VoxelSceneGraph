# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Mapping

import torch
from yacs.config import CfgNode

from .attribute_head import build_roi_attribute_head
from .box_head import build_roi_box_head
from .keypoint_head import build_roi_keypoint_head
from .mask_head import build_roi_mask_head
from .relation_head import build_roi_relation_head
from ..abstractions.attribute_head import BoxHeadTargets
from ..abstractions.backbone import FeatureMaps, AnchorStrides
from ..abstractions.box_head import BoxHeadTrainProposal, BoxHeadTestProposal
from ..abstractions.loss import LossDict
from ..abstractions.region_proposal import RPNProposals
from ..abstractions.roi_heads import CombinedROIHeads as AbstractCombinedROIHeads
from ...structures import BoxList


class CombinedROIHeads(AbstractCombinedROIHeads):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.
    Note: can contain 0 head (no op).
    """

    def __init__(
            self,
            cfg: CfgNode,
            heads: Mapping[str, torch.nn.Module]
    ):
        super().__init__(heads)
        self.cfg = cfg

        if not hasattr(self, "box"):
            assert not cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR, "No box head; cannot share features."
            assert not cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR, "No box head; cannot share features."

        if hasattr(self, "mask") and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # Check that pooler configs are identical
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION == cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH == cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_DEPTH
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO == cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            self.mask.feature_extractor = self.box.feature_extractor

        if hasattr(self, "keypoint") and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # Check that pooler configs are identical
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION_DEPTH
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None = None,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        if self.training:
            assert targets is not None
            return self.standard_forward_train(features, proposals, targets)
        return self.standard_forward_test(features, proposals, targets, loss_during_testing=loss_during_testing)

    def standard_forward_train(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets
    ) -> tuple[list[BoxList], LossDict]:
        losses = {}

        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            # Here, we need to call self.box.subsample to set some fields (even for relations)
            keep = self.box.subsample(proposals, targets)
            # If RELATION_ON, proposals == box_head_train_boxes
            box_head_train_boxes = proposals[keep]
            x, class_logits, box_regression = self.box(features, box_head_train_boxes)

            if not self.cfg.MODEL.RELATION_ON:
                # No loss during relation training
                loss_box = self.box.loss(class_logits, box_regression, box_head_train_boxes)
                losses.update(loss_box)
            else:
                # We only need to run the full pipeline for relations during training
                x, proposals = self.box.post_process_predictions(x, class_logits, box_regression, box_head_train_boxes)
        else:
            # One stage models
            # Note: sharing the feature extractor is not supported in this case
            x, box_head_train_boxes = None, proposals

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, box_head_train_boxes)
            if not self.cfg.MODEL.RELATION_ON:
                loss_attribute = self.attribute(attribute_logits, box_head_train_boxes)
                losses.update(loss_attribute)
            else:
                # Attributes are only reused in the relation head
                proposals = self.attribute.post_process_predictions(attribute_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError("Cannot share box extractor for the mask head with a one-stage detector.")
                mask_features = x
            else:
                mask_features = features

            if not self.cfg.MODEL.RELATION_ON:
                keep = self.mask.subsample(proposals)
                mask_head_train_boxes = proposals[keep]
            else:
                mask_head_train_boxes = proposals

            mask_logits = self.mask(mask_features, mask_head_train_boxes)

            if not self.cfg.MODEL.RELATION_ON:
                loss_mask = self.mask.loss(mask_logits, mask_head_train_boxes, targets)
                losses.update(loss_mask)
            else:
                proposals = self.mask.post_process_predictions(mask_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError("Cannot share box extractor for the keypoint head with a one-stage detector.")
                keypoint_features = x
            else:
                keypoint_features = features

            if not self.cfg.MODEL.RELATION_ON:
                keep = self.keypoint.subsample(proposals)
                kp_head_train_boxes = proposals[keep]
            else:
                kp_head_train_boxes = proposals

            kp_logits = self.keypoint(keypoint_features, kp_head_train_boxes)

            if not self.cfg.MODEL.RELATION_ON:
                loss_keypoint = self.keypoint.loss(kp_logits, kp_head_train_boxes)
                losses.update(loss_keypoint)
            else:
                proposals = self.keypoint.post_process_predictions(kp_logits, kp_head_train_boxes)

        # ==============================================================================================================
        if hasattr(self, "relation"):
            rel_pair_idxs, rel_labels, rel_binaries = self.relation.subsample_relation_pairs(proposals, targets)
            refined_obj_logits, relation_logits, refined_att_logits, add_losses_required = \
                self.relation(features, rel_pair_idxs, proposals)
            loss_relation = self.relation.loss(
                refined_obj_logits,
                relation_logits,
                refined_att_logits,
                add_losses_required,
                proposals,
                rel_binaries,
                rel_labels
            )
            losses.update(loss_relation)

        return proposals, losses

    def standard_forward_test(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        if loss_during_testing:
            assert targets is not None
        losses = {}

        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            x, class_logits, box_regression = self.box(features, proposals)

            if not self.cfg.MODEL.RELATION_ON:
                if loss_during_testing:
                    # No loss during relation training
                    keep = self.box.subsample(proposals, targets)
                    loss_box = self.box.loss(class_logits[keep], box_regression[keep], proposals[keep])
                    losses.update(loss_box)

            x, proposals = self.box.post_process_predictions(x, class_logits, box_regression, proposals)

        else:
            # One stage models
            # Note: sharing the feature extractor is not supported in this case
            x = None

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, proposals)

            if not self.cfg.MODEL.RELATION_ON:
                if loss_during_testing:
                    keep = self.box.subsample(proposals, targets)
                    loss_attribute = self.attribute(attribute_logits[keep], proposals[keep])
                    losses.update(loss_attribute)

            proposals = self.attribute.post_process_predictions(attribute_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError("Cannot share box extractor for the mask head with a one-stage detector.")
                mask_features = x
            else:
                mask_features = features

            mask_logits = self.mask(mask_features, proposals)

            if not self.cfg.MODEL.RELATION_ON:
                if loss_during_testing:
                    keep = self.mask.subsample(proposals)
                    loss_mask = self.mask.loss(mask_logits[keep], proposals[keep], targets)
                    losses.update(loss_mask)

            proposals = self.mask.post_process_predictions(mask_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError("Cannot share box extractor for the keypoint head with a one-stage detector.")
                keypoint_features = x
            else:
                keypoint_features = features

            kp_logits = self.keypoint(keypoint_features, proposals)

            if not self.cfg.MODEL.RELATION_ON:
                if loss_during_testing:
                    keep = self.keypoint.subsample(proposals)
                    loss_keypoint = self.keypoint.loss(kp_logits[keep], proposals[keep])
                    losses.update(loss_keypoint)

            proposals = self.keypoint.post_process_predictions(kp_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "relation"):
            rel_pair_idxs = self.relation.prepare_relation_pairs(proposals)

            # We need to filter samples with no sampled relations
            # But we also need to output them still (for evaluation purposes)
            kept_proposals = [prop for idxs, prop in zip(rel_pair_idxs, proposals) if idxs.numel() > 0]
            kept_rel_pair_idxs = [idxs for idxs in rel_pair_idxs if idxs.numel() > 0]

            # Check if we have anything to predict
            if kept_rel_pair_idxs:
                refined_obj_logits, relation_logits, refined_att_logits, add_losses_required = \
                    self.relation(features, kept_rel_pair_idxs, kept_proposals)
                kept_proposals = self.relation.post_process_predictions(
                    kept_rel_pair_idxs, refined_obj_logits, relation_logits, refined_att_logits, kept_proposals
                )

                if loss_during_testing:
                    # Because of the complex sampling, we cannot factorize much computation for the relation loss
                    rel_pair_idxs, rel_labels, rel_binaries = self.relation.subsample_relation_pairs(
                        kept_proposals, targets
                    )
                    refined_obj_logits, relation_logits, refined_att_logits, add_losses_required = \
                        self.relation(features, kept_rel_pair_idxs, kept_proposals)
                    loss_relation = self.relation.loss(
                        refined_obj_logits,
                        relation_logits,
                        refined_att_logits,
                        add_losses_required,
                        kept_proposals,
                        rel_binaries,
                        rel_labels
                    )
                    losses.update(loss_relation)

            # Finally we have to insert the proposals which have no sampled relations (with initialized fields)
            idx_kept_prop = 0
            for global_idx, idxs in enumerate(rel_pair_idxs):
                if idxs.numel() == 0:
                    # Init field
                    proposals[global_idx].REL_PAIR_IDXS = torch.empty(0, 2)
                else:
                    # Replace from kept proposals and update counter for that list
                    proposals[global_idx] = kept_proposals[idx_kept_prop]
                    idx_kept_prop += 1

        return proposals, losses

    def sample_and_predict_relation(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None,
            compute_losses: bool,
    ) -> tuple[BoxHeadTestProposal, list[list | None]]:
        """Max-memory-usage-optimized pipeline for relation."""

        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            x, class_logits, box_regression = self.box(features, proposals)
            x, proposals = self.box.post_process_predictions(x, class_logits, box_regression, proposals)
        else:
            # One stage models
            # Note: sharing the feature extractor is not supported in this case
            x = None

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, proposals)
            proposals = self.attribute.post_process_predictions(attribute_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError("Cannot share box extractor for the mask head with a one-stage detector.")
                mask_features = x
            else:
                mask_features = features

            mask_logits = self.mask(mask_features, proposals)
            proposals = self.mask.post_process_predictions(mask_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if x is None:
                    raise RuntimeError(
                        "Cannot share box extractor for the keypoint head with a one-stage detector.")
                keypoint_features = x
            else:
                keypoint_features = features

            kp_logits = self.keypoint(keypoint_features, proposals)
            proposals = self.keypoint.post_process_predictions(kp_logits, proposals)

        # ==============================================================================================================
        # Relation stuff
        if self.training or not self.training and compute_losses:
            rel_pair_idxs_train, rel_labels_train, rel_binaries_train = \
                self.relation.subsample_relation_pairs(proposals, targets)
            refined_obj_logits_train, relation_logits_train, refined_att_logits_train, add_losses_required_train = \
                self.relation(features, rel_pair_idxs_train, proposals)
        else:
            rel_pair_idxs_train = None
            rel_labels_train = None
            rel_binaries_train = None
            refined_obj_logits_train = None
            relation_logits_train = None
            refined_att_logits_train = None
            add_losses_required_train = None

        if not self.training:
            rel_pair_idxs_test = self.relation.prepare_relation_pairs(proposals)
            refined_obj_logits_test, relation_logits_test, refined_att_logits_test, add_losses_required_test = \
                self.relation(features, rel_pair_idxs_test, proposals)
        else:
            rel_pair_idxs_test = None
            refined_obj_logits_test = None
            relation_logits_test = None
            refined_att_logits_test = None
            add_losses_required_test = None

        # Slightly ugly but eh...
        return proposals, [
            rel_pair_idxs_train,
            rel_labels_train,
            rel_binaries_train,
            refined_obj_logits_train,
            relation_logits_train,
            refined_att_logits_train,
            add_losses_required_train,
            rel_pair_idxs_test,
            refined_obj_logits_test,
            relation_logits_test,
            refined_att_logits_test,
            add_losses_required_test
        ]

    def postprocess_relation(
            self, proposals: BoxHeadTestProposal, pre_computations: list[list | None]
    ) -> tuple[list[BoxList], LossDict]:
        """Sister method to sample_and_predict_relation."""
        # Slightly ugly but eh...
        rel_pair_idxs_train, \
            rel_labels_train, \
            rel_binaries_train, \
            refined_obj_logits_train, \
            relation_logits_train, \
            refined_att_logits_train, \
            add_losses_required_train, \
            rel_pair_idxs_test, \
            refined_obj_logits_test, \
            relation_logits_test, \
            refined_att_logits_test, \
            add_losses_required_test = pre_computations

        if rel_pair_idxs_train is not None:
            loss_relation = self.relation.loss(
                refined_obj_logits_train,
                relation_logits_train,
                refined_att_logits_train,
                add_losses_required_train,
                proposals,
                rel_binaries_train,
                rel_labels_train
            )
        else:
            loss_relation = {}

        if rel_pair_idxs_test is not None:
            proposals = self.relation.post_process_predictions(
                rel_pair_idxs_test, refined_obj_logits_test, relation_logits_test, refined_att_logits_test, proposals
            )

        return proposals, loss_relation


def build_roi_heads(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides,
        is_rpn_only: bool = False,
        has_boxes: bool = True,
        has_masks: bool = False,
        has_keypoints: bool = False,
        has_attributes: bool = False,
        has_relations: bool = False,
) -> CombinedROIHeads:
    """
    :param cfg:
    :param in_channels:
    :param anchor_strides: strides for the Poolers.
    :param is_rpn_only: we only wish to evaluate region proposals: no ROI heads needed.
    :param has_boxes: should only ever be False if is_rpn_only is True,
                      OR the model is a one-stage model (e.g. RetinaNet).
    :param has_masks:
    :param has_keypoints:
    :param has_attributes:
    :param has_relations:
    :return:
    """

    # Individually create the heads, that will be combined afterward
    roi_heads: list[tuple[str, torch.nn.Module]] = []

    if not is_rpn_only:
        if has_boxes:
            roi_heads.append(("box", build_roi_box_head(cfg, in_channels, anchor_strides)))
        if has_masks:
            roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels, anchor_strides)))
        if has_keypoints:
            roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels, anchor_strides)))
        if has_attributes:
            roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels, anchor_strides)))
        if has_relations:
            roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels, anchor_strides)))

    # Combine individual heads in a single module
    # Linter going haywire
    # noinspection PyTypeChecker
    return CombinedROIHeads(cfg, roi_heads)
