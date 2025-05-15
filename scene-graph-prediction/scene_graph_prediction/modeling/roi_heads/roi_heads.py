# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Mapping

import torch
from yacs.config import CfgNode

from scene_graph_api.tensor_structures import BoxListOps
from .attribute_head import build_roi_attribute_head
from .box_head import build_roi_box_head
from .keypoint_head import build_roi_keypoint_head
from .mask_head import build_roi_mask_head
from .relation_head import build_roi_relation_head
from ..abstractions.attribute_head import BoxHeadTargets
from ..abstractions.backbone import FeatureMaps, AnchorStrides
from ..abstractions.box_head import BoxHeadTrainProposal, BoxHeadTestProposal, BoxHeadTestProposals
from ..abstractions.loss import LossDict
from ..abstractions.region_proposal import RPNProposals
from ..abstractions.roi_heads import CombinedROIHeads as AbstractCombinedROIHeads
from ..utils.misc import LossComputationCfg
from ...structures import BoxList


class CombinedROIHeads(AbstractCombinedROIHeads):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.
    Note: can contain 0 head (no op).
    """

    def __init__(
            self,
            cfg: CfgNode,
            heads: Mapping[str, torch.nn.Module],
            detector_is_one_stage: bool
    ):
        super().__init__(heads)
        self.cfg = cfg

        if not hasattr(self, "box"):
            assert not cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR, "No box head; cannot share features."
            assert not cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR, "No box head; cannot share features."

        if hasattr(self, "mask") and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # Check that pooler configs are identical
            assert not detector_is_one_stage, "Cannot share box extractor for the mask head with a one-stage detector."
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION == cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH == cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_DEPTH
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO == cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            self.mask.feature_extractor = self.box.feature_extractor

        if hasattr(self, "keypoint") and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # Check that pooler configs are identical
            assert not detector_is_one_stage, "Cannot share box extractor for the kp head with a one-stage detector."
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION_DEPTH
            assert cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO == cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None = None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[list[BoxList], LossDict]:
        if self.training:
            assert targets is not None
            return self.standard_forward_train(features, proposals, targets, compute_loss)
        return self.standard_forward_test(features, proposals, targets, compute_loss)

    def standard_forward_train(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets,
            compute_loss: LossComputationCfg
    ) -> tuple[list[BoxList], LossDict]:
        losses = {}

        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            # Note if RELATION_on then box_head_train_boxes is proposals
            keep = self.box.assign_label_to_proposals(proposals, targets)
            box_head_train_boxes = [p[k] for p, k in zip(proposals, keep)]
            x, class_logits, box_regression = self.box(features, box_head_train_boxes)

            if compute_loss.compute_roi_heads_loss:
                # No loss during relation training
                loss_box = self.box.loss(class_logits, box_regression, box_head_train_boxes)
                losses.update(loss_box)

            if self.cfg.MODEL.RELATION_ON:
                # We only need to run the full pipeline for relations during training
                x, box_head_train_boxes = self.box.post_process_predictions(
                    x, class_logits, box_regression, box_head_train_boxes
                )
        else:
            # One stage models
            # Note: sharing the feature extractor is not supported in this case
            x, box_head_train_boxes = None, proposals

        if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
            box_head_train_boxes = self._add_roi_heads_gt_to_proposals(box_head_train_boxes, targets)  # With labels

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, box_head_train_boxes)
            if compute_loss.compute_roi_heads_loss:
                loss_attribute = self.attribute(attribute_logits, box_head_train_boxes)
                losses.update(loss_attribute)

            if self.cfg.MODEL.RELATION_ON:
                # Attributes are only reused in the relation head
                box_head_train_boxes = self.attribute.post_process_predictions(attribute_logits, box_head_train_boxes)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                    raise NotImplementedError(
                        "We currently do not support shared feature extraction with addition of GT boxes. "
                        "We would need to also generate the appropriate box features for the GT boxes."
                    )
                mask_features = x
            else:
                mask_features = features

            if not self.cfg.MODEL.RELATION_ON:
                keep = self.mask.subsample(box_head_train_boxes)
                mask_head_train_boxes = [p[k] for p, k in zip(box_head_train_boxes, keep)]
            else:
                mask_head_train_boxes = box_head_train_boxes

            mask_logits = self.mask(mask_features, mask_head_train_boxes)

            if compute_loss.compute_roi_heads_loss:
                loss_mask = self.mask.loss(mask_logits, mask_head_train_boxes, targets)
                losses.update(loss_mask)

            if self.cfg.MODEL.RELATION_ON:
                box_head_train_boxes = self.mask.post_process_predictions(mask_logits, box_head_train_boxes)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                    raise NotImplementedError(
                        "We currently do not support shared feature extraction with addition of GT boxes. "
                        "We would need to also generate the appropriate box features for the GT boxes."
                    )
                keypoint_features = x
            else:
                keypoint_features = features

            if not self.cfg.MODEL.RELATION_ON:
                keep = self.keypoint.subsample(box_head_train_boxes)
                kp_head_train_boxes = [p[k] for p, k in zip(box_head_train_boxes, keep)]
            else:
                kp_head_train_boxes = box_head_train_boxes

            kp_logits = self.keypoint(keypoint_features, kp_head_train_boxes)

            if compute_loss.compute_roi_heads_loss:
                loss_keypoint = self.keypoint.loss(kp_logits, kp_head_train_boxes)
                losses.update(loss_keypoint)

            # FIXME we need to add kp results to the existing proposals
            #  (the earlier version assumes that this is our only endpoint...)
            # if self.cfg.MODEL.RELATION_ON:
            #     proposals = self.keypoint.post_process_predictions(kp_logits, kp_head_train_boxes)

        # ==============================================================================================================
        if hasattr(self, "relation"):
            # If there is a box head, we need to assign the final labels
            # Otherwise, we're using a one-stage detector which assigned this earlier
            if hasattr(self, "box"):
                self.box.assign_label_to_proposals(box_head_train_boxes, targets)

            rel_pair_idxs, rel_labels, rel_binaries = self.relation.subsample_relation_pairs(
                box_head_train_boxes, targets
            )
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
            if compute_loss.compute_rel_heads_loss:
                losses.update(loss_relation)

        return proposals, losses

    def standard_forward_test(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets,
            compute_loss: LossComputationCfg
    ) -> tuple[list[BoxList], LossDict]:
        losses = {}

        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            x, class_logits, box_regression = self.box(features, proposals)

            if compute_loss.compute_roi_heads_loss:
                # No loss during relation training
                keep = self.box.assign_label_to_proposals(proposals, targets)
                box_props = [p[k] for p, k in zip(proposals, keep)]
                loss_box = self.box.loss(class_logits[keep], box_regression[keep], box_props)
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

            if compute_loss.compute_roi_heads_loss:
                keep = self.box.assign_label_to_proposals(proposals, targets)
                attr_props = [p[k] for p, k in zip(proposals, keep)]
                loss_attribute = self.attribute(attribute_logits[keep], attr_props)
                losses.update(loss_attribute)

            proposals = self.attribute.post_process_predictions(attribute_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            else:
                mask_features = features

            mask_logits = self.mask(mask_features, proposals)

            if compute_loss.compute_roi_heads_loss:
                keep = self.mask.subsample(proposals)
                mask_props = [p[k] for p, k in zip(proposals, keep)]
                loss_mask = self.mask.loss(mask_logits[keep], mask_props, targets)
                losses.update(loss_mask)

            proposals = self.mask.post_process_predictions(mask_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                keypoint_features = x
            else:
                keypoint_features = features

            kp_logits = self.keypoint(keypoint_features, proposals)

            if compute_loss.compute_roi_heads_loss:
                keep = self.keypoint.subsample(proposals)
                kp_props = [p[k] for p, k in zip(proposals, keep)]
                loss_keypoint = self.keypoint.loss(kp_logits[keep], kp_props)
                losses.update(loss_keypoint)

            proposals = self.keypoint.post_process_predictions(kp_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "relation"):
            # If there is a box head, we need to assign the final labels
            # Otherwise, we're using a one-stage detector which assigned this earlier
            if hasattr(self, "box") and targets is not None:
                self.box.assign_label_to_proposals(proposals, targets)

            # Optionally, filter/clean predictions based on the targets (and the configuration)
            proposals = self._replace_proposals_with_gt(proposals, targets)

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

                if compute_loss.compute_rel_heads_loss:
                    # Because of the complex sampling, we cannot factorize much computation for the relation loss
                    rel_pair_idxs, rel_labels, rel_binaries = self.relation.subsample_relation_pairs(
                        kept_proposals, targets
                    )
                    refined_obj_logits, relation_logits, refined_att_logits, add_losses_required = \
                        self.relation(features, rel_pair_idxs, kept_proposals)
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
        """Max-memory-usage-optimized pipeline for roi heads (except relation)."""
        # ==============================================================================================================
        if hasattr(self, "box"):
            # Two stage models
            keep = self.box.assign_label_to_proposals(proposals, targets)
            box_head_train_boxes = [p[k] for p, k in zip(proposals, keep)]
            x, class_logits, box_regression = self.box(features, box_head_train_boxes)
            box_pre_computations = [class_logits, box_regression, box_head_train_boxes]
        else:
            # One stage models
            x, box_head_train_boxes = None, proposals
            box_pre_computations = [None]

        if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
            box_head_train_boxes = self._add_roi_heads_gt_to_proposals(box_head_train_boxes, targets)  # With labels

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, box_head_train_boxes)
            attr_pre_computations = [attribute_logits, box_head_train_boxes]
        else:
            attr_pre_computations = [None]

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                    raise NotImplementedError(
                        "We currently do not support shared feature extraction with addition of GT boxes. "
                        "We would need to also generate the appropriate box features for the GT boxes."
                    )
                mask_features = x
            else:
                mask_features = features

            keep = self.mask.subsample(box_head_train_boxes)
            mask_head_train_boxes = [p[k] for p, k in zip(box_head_train_boxes, keep)]
            mask_logits = self.mask(mask_features, mask_head_train_boxes)
            mask_pre_computations = [mask_logits, mask_head_train_boxes, targets]
        else:
            mask_pre_computations = [None]

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                    raise NotImplementedError(
                        "We currently do not support shared feature extraction with addition of GT boxes. "
                        "We would need to also generate the appropriate box features for the GT boxes."
                    )
                keypoint_features = x
            else:
                keypoint_features = features

            keep = self.keypoint.subsample(box_head_train_boxes)
            kp_head_train_boxes = [p[k] for p, k in zip(box_head_train_boxes, keep)]
            kp_logits = self.keypoint(keypoint_features, kp_head_train_boxes)
            kp_pre_computations = [kp_logits, kp_head_train_boxes]
        else:
            kp_pre_computations = [None]

        return box_pre_computations, attr_pre_computations, mask_pre_computations, kp_pre_computations

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
        losses = {}

        # ==============================================================================================================
        if hasattr(self, "box"):
            class_logits, box_regression, box_head_train_boxes = box_pre_computations
            loss_box = self.box.loss(class_logits, box_regression, box_head_train_boxes)
            losses.update(loss_box)

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            attribute_logits, box_head_train_boxes = attr_pre_computations
            loss_attribute = self.attribute(attribute_logits, box_head_train_boxes)
            losses.update(loss_attribute)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            mask_logits, mask_head_train_boxes, targets = mask_pre_computations
            loss_mask = self.mask.loss(mask_logits, mask_head_train_boxes, targets)
            losses.update(loss_mask)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            kp_logits, kp_head_train_boxes = kp_pre_computations
            loss_keypoint = self.keypoint.loss(kp_logits, kp_head_train_boxes)
            losses.update(loss_keypoint)

        return losses

    def sample_and_predict_relation(
            self,
            features: FeatureMaps,
            proposals: RPNProposals | BoxHeadTrainProposal | BoxHeadTestProposal,
            targets: BoxHeadTargets | None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[BoxHeadTestProposal, list[list | None]]:
        """Max-memory-usage-optimized training pipeline for relation prediction."""

        # ==============================================================================================================
        if hasattr(self, "box"):
            x, class_logits, box_regression = self.box(features, proposals)
            x, proposals = self.box.post_process_predictions(x, class_logits, box_regression, proposals)
        else:
            # One stage models
            # Note: sharing the feature extractor is not supported in this case
            x = None

        if self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
            proposals = self._add_roi_heads_gt_to_proposals(proposals, targets)  # With labels

        # ==============================================================================================================
        if hasattr(self, "attribute"):
            # We reuse the sampling of the box head here
            attribute_logits = self.attribute(features, proposals)
            proposals = self.attribute.post_process_predictions(attribute_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "mask"):
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            else:
                mask_features = features

            mask_logits = self.mask(mask_features, proposals)
            proposals = self.mask.post_process_predictions(mask_logits, proposals)

        # ==============================================================================================================
        if hasattr(self, "keypoint"):
            if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                keypoint_features = x
            else:
                keypoint_features = features

            kp_logits = self.keypoint(keypoint_features, proposals)
            proposals = self.keypoint.post_process_predictions(kp_logits, proposals)

        # ==============================================================================================================
        # Relation stuff

        # If there is a box head, we need to assign the final labels
        # Otherwise, we're using a one-stage detector which assigned this earlier
        if hasattr(self, "box"):
            self.box.assign_label_to_proposals(proposals, targets)

        # Optionally, filter/clean predictions based on the targets (and the configuration)
        proposals = self._replace_proposals_with_gt(proposals, targets)

        if compute_loss.compute_rel_heads_loss:
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

        # Note: we keep this bit of code in case we want to make this pipeline compatible with testing
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
        # noinspection PyTypeChecker
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

        # Note: we keep this bit of code in case we want to make this pipeline compatible with testing
        if rel_pair_idxs_test is not None:
            proposals = self.relation.post_process_predictions(
                rel_pair_idxs_test, refined_obj_logits_test, relation_logits_test, refined_att_logits_test, proposals
            )

        return proposals, loss_relation

    def _add_roi_heads_gt_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]) -> BoxHeadTestProposals:
        """
        Add groundtruth boxes / masks with labels to the proposals.
        # TODO also add keypoints / attributes
        Note: useful when training a downstream relation head.
        """
        if len(proposals) == 0:
            return proposals

            # We don't want to copy any field except LABELS; otherwise the BoxList concatenation will fail
        INF = 10
        gt_boxes = [target.copy() for target in targets]
        device = torch.device(self.cfg.MODEL.DEVICE)
        n_dim = self.cfg.INPUT.N_DIM

        # Later cat of bbox requires all fields to be present for all bbox,
        # So we need to add dummy fields that are missing
        for gt_box, target, proposal in zip(gt_boxes, targets, proposals):
            gt_box.PRED_SCORES = torch.ones(len(gt_box), device=device)
            gt_box.PRED_LABELS = target.LABELS.long()

            # We need to add the pred_logits to match the set of fields from the proposals
            # Note: in particular here, we're checking whether we're a 1-stage or 2-stage detector
            num_classes = proposal.BOXES_PER_CLS.shape[1] // (2 * n_dim)
            logits = torch.zeros((len(gt_box), num_classes), dtype=torch.float32, device=device)
            logits[:, gt_box.PRED_LABELS] = INF
            gt_box.PRED_LOGITS = logits
            gt_box.BOXES_PER_CLS = torch.tile(gt_box.boxes, (1, num_classes))

            # Note: as it's the box head's job to add the labels field,
            #       one stage object detectors will already add this field
            #       In comparison, two-stage detectors will add it later.
            #       This ensures that each detector remains master of the matching algorithm.
            if proposal.has_field(BoxList.AnnotationField.LABELS):
                gt_box.LABELS = target.LABELS

            if proposal.has_field(BoxList.PredictionField.PRED_SEGMENTATION):
                gt_box.PRED_SEGMENTATION = proposal.PRED_SEGMENTATION

            if proposal.has_field(BoxList.PredictionField.PRED_MASKS):
                gt_box.PRED_MASKS = target.MASKS

            if proposal.has_field(BoxList.PredictionField.MATCHED_IDXS):
                # Matching is easy at least
                gt_box.MATCHED_IDXS = torch.arange(0, len(target), 1, device=device)

        return [BoxListOps.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

    def _replace_proposals_with_gt(self, proposals: list[BoxList], targets: list[BoxList]) -> list[BoxList]:
        """
        For the Scene Graph prediction task, we may want to have a fine control over
        which parts of the prediction are replaced with groundtruth annotation during testing.
        This way, we can find out which parts of the network cause the biggest performance degradation.
        WARNING: this method requires GT labels to be assigned to work.
        """
        if not (self.cfg.TEST.RELATION.REPLACE_SEGMENTATION or
                self.cfg.TEST.RELATION.REMOVE_FALSE_POSITIVES or
                self.cfg.TEST.RELATION.REPLACE_MATCHED_BOXES):
            return proposals

        assert targets is not None
        assert len(proposals) == len(targets)
        for idx in range(len(proposals)):
            proposal, target = proposals[idx], targets[idx]

            # Replace the semantic segmentation / binary masks
            if self.cfg.TEST.RELATION.REPLACE_SEGMENTATION:
                if proposal.has_field(BoxList.PredictionField.PRED_SEGMENTATION):
                    proposal.PRED_SEGMENTATION = target.SEGMENTATION
                if proposal.has_field(BoxList.PredictionField.PRED_MASKS):
                    proposal.PRED_MASKS = target.MASKS

            # Remove objects with no groundtruth match
            if self.cfg.TEST.RELATION.REMOVE_FALSE_POSITIVES:
                proposals[idx] = proposal[proposal.LABELS > 0]

            # Coordinates of predicted objects having a match with a GT object are replaced with GT coordinates
            if self.cfg.TEST.RELATION.REPLACE_MATCHED_BOXES:
                proposal.boxes = target.boxes[proposal.MATCHED_IDXS.clamp(min=0)]

        return proposals


def build_roi_heads(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides,
        detector_is_one_stage: bool,
        is_rpn_only: bool = False,
        has_boxes: bool = True,
        has_masks: bool = False,
        has_keypoints: bool = False,
        has_attributes: bool = False,
        has_relations: bool = False
) -> CombinedROIHeads:
    """
    :param cfg:
    :param in_channels:
    :param anchor_strides: strides for the Poolers.
    :param detector_is_one_stage: whether the RPN is actually a one-stage detector
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
    return CombinedROIHeads(cfg, roi_heads, detector_is_one_stage)
