# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Implements the Generalized R-CNN framework."""
from __future__ import annotations

from abc import ABC
from functools import reduce

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import ImageList, BoxList, BoxListOps
from ..abstractions.backbone import Backbone, FeatureMaps
from ..abstractions.box_head import BoxHeadTestProposals
from ..abstractions.detector import AbstractDetector
from ..abstractions.loss import LossDict
from ..abstractions.region_proposal import RPNProposals, RPN
from ..abstractions.roi_heads import CombinedROIHeads


class BaseDetector(AbstractDetector, ABC):
    """
    Main class for Generalized R-CNN. Currently, supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    def __init__(
            self,
            cfg: CfgNode,
            backbone: Backbone,
            rpn: RPN,
            roi_heads: CombinedROIHeads
    ):
        # Some config checks
        if cfg.MODEL.RPN.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN:
            assert not cfg.MODEL.RELATION_ON, ("Cannot add RPN GT boxes when training a relation detector. "
                                               "Use MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN instead.")
        if cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN:
            assert not cfg.MODEL.RPN_ONLY, ("Cannot add box head GT boxes when training a relation detector. "
                                            "Use MODEL.RPN.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN instead.")
        if cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE:
            assert cfg.MODEL.RELATION_ON, "The optimized pipeline is only available for relations."

        super().__init__(cfg, backbone, rpn, roi_heads)

    def forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        """
        :param images: images to be processed
        :param targets: ground-truth boxes present in the image (optional)
        :param loss_during_testing: whether to compute the loss for relevant modules even when evaluating.

        :returns: The output from the model.
                  During training, it returns a dict[Tensor] which contains the losses.
                  During testing, it returns list[BoxList] contains additional fields
                  like `pred_scores`, `pred_labels` and `pred_masks` (for Mask R-CNN models).
        """
        if self.cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE and self.training:
            # This algorithm cannot be used for testing because we filter out images with no sampled relation
            # The filtering messes up the prediction ordering if we have a batch size > 1
            proposals, all_losses = self.roi_heads_optimized_forward(images, targets, loss_during_testing)
        else:
            proposals, all_losses = self.standard_forward(images, targets, loss_during_testing)

        if targets is not None and not self.training:
            # Add affine from targets (annoying to get it from the image directly)
            for r, t in zip(proposals, targets):  # type: BoxList, BoxList
                if t.has_field(BoxList.AnnotationField.AFFINE_MATRIX):
                    r.AFFINE_MATRIX = t.AFFINE_MATRIX

        return proposals, all_losses

    def standard_forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        """Standard pipeline where all images are handled at once."""
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_list = ImageList.to_image_list(images, self.n_dim)
        features, proposals, all_losses = self._prepare_rpn_proposals(image_list, targets, loss_during_testing)

        if self.roi_heads:
            proposals, roi_head_losses = self.roi_heads(features, proposals, targets)
            all_losses.update(roi_head_losses)

        return proposals, all_losses

    def roi_heads_optimized_forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        """
        Pipeline where the feature maps are computed for one image at a time and only pooled features are kept.
        This allows training on multiple images, while mitigating the maximum memory footprint.
        WARNING: only available for relation training.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        assert self.cfg.MODEL.RELATION_ON

        device = torch.device(self.cfg.MODEL.DEVICE)
        images = ImageList.to_image_list(images, self.n_dim)
        compute_losses = self.training or loss_during_testing

        # Per-image, per-head intermediate results which can be used to compute a loss / post-process predictions
        all_proposals = []
        all_pre_computations = []
        for idx, cur_image in enumerate(images):
            # We need to get the ith image to ImageList with one image and then set the device
            cur_image = cur_image.to(device)
            cur_target = [targets[idx].to(device)] if targets is not None else None

            # We assume that the RPN is not trained with the optimized pipeline
            features, proposals, _ = self._prepare_rpn_proposals(cur_image, cur_target, False)

            proposals, pre_computations = self.roi_heads.sample_and_predict_relation(
                features, proposals, cur_target, compute_losses
            )

            all_proposals.append(proposals)
            all_pre_computations.append(pre_computations)

            # Delete whatever is not needed anymore
            del cur_image
            del cur_target
            del features

        # Aggregate intermediate results
        all_proposals = reduce(lambda a, b: a + b, all_proposals)
        pre_computations_transposed = list(zip(*all_pre_computations))
        all_pre_computations = [
            None if None in comp_list else reduce(lambda a, b: a + b, comp_list)
            for comp_list in pre_computations_transposed
        ]

        # Aggregate or compute losses
        all_proposals, all_losses = self.roi_heads.postprocess_relation(all_proposals, all_pre_computations)
        return all_proposals, all_losses

    def _prepare_rpn_proposals(
            self,
            image_list: ImageList,
            targets: list[BoxList] | None = None,
            loss_during_testing: bool = False
    ) -> tuple[FeatureMaps, BoxHeadTestProposals, LossDict]:
        """
        Handle the RPN prediction pipeline and proposals preparation for all scenarios, e.g.
        one-stage vs two-stage object detector, w/ vs w/o relation training.
        """
        rpn_losses = {}
        features = self.backbone(image_list.tensors)
        compute_losses = self.training or loss_during_testing

        if not self.cfg.MODEL.RELATION_ON or not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            # Normal computation pipeline
            raw_rpn_predictions = self.rpn(image_list, features)
            proposals = self.rpn.post_process_predictions(raw_rpn_predictions, targets=targets)

            # Prevent any RPN to produce a loss when it shouldn't
            if not (self.cfg.MODEL.ROI_HEADS_ONLY or self.cfg.MODEL.RELATION_ON) and compute_losses:
                rpn_losses.update(self.rpn.loss(raw_rpn_predictions, targets=targets))

            # If gradients are required, the grad_context will still have a reference, otherwise we may free the memory
            del raw_rpn_predictions

            # Check whether to add GT annotation to predictions:
            if self.cfg.MODEL.RPN.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                proposals = self._add_rpn_gt_to_proposals(proposals, targets)
            elif self.cfg.MODEL.ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                proposals = self._add_box_head_gt_to_proposals(proposals, targets)

            # Check whether the RPN is a one-stage detector, i.e. we need to spoof the OBJECTNESS field in the proposals
            if self.cfg.MODEL.ROI_HEADS_ONLY:
                for proposal in proposals:
                    if not proposal.has_field(BoxList.PredictionField.OBJECTNESS):
                        proposal.OBJECTNESS = proposal.PRED_SCORES
                    # TODO to make one-stage detectors compatible with ROI heads (other than box),
                    #  we need to compute the MATCHED_IDXS field.
                    #  (The one deleted here was from the RPN matcher, which is not compatible with all options here)
                    proposal.del_field(BoxList.PredictionField.MATCHED_IDXS)

            # Optionally, filter/clean predictions based on the targets (and the configuration)
            if self.cfg.MODEL.RELATION_ON and not self.training:
                proposals = self._replace_proposals_with_gt(proposals, targets)

        else:
            # For relation training with GT boxes, we only need to prepare proposals from the targets
            # Note: if we're not using the GT labels, then it's the BOxHead's role to add the predicted logits
            #       One-stage detector do not support this configuration and no extra processing is required.
            proposals = [target.copy_with_all_fields() for target in targets]

        return features, proposals, rpn_losses

    @staticmethod
    def _add_rpn_gt_to_proposals(proposals: list[BoxList], targets: list[BoxList]) -> RPNProposals:
        """
        Add groundtruth boxes to the proposals.
        Note: useful when training a downstream box head.
        """
        if len(proposals) == 0:
            return proposals

        # We don't want to copy any field; otherwise the cat will fail
        gt_boxes = [target.copy() for target in targets]

        # Get the device we're operating on
        device = proposals[0].boxes.device

        # Later cat of bbox requires all fields to be present for all bbox,
        # So we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.OBJECTNESS = torch.ones(len(gt_box), device=device)

        return [BoxListOps.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

    def _add_box_head_gt_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]) -> BoxHeadTestProposals:
        """
        Add groundtruth boxes with labels to the proposals.
        Note: useful when training a downstream relation head.
        """
        if len(proposals) == 0:
            return proposals

        # We don't want to copy any field except LABELS; otherwise the BoxList concatenation will fail
        INF = 10
        gt_boxes = [target.copy() for target in targets]
        device = torch.device(self.cfg.MODEL.DEVICE)

        # Later cat of bbox requires all fields to be present for all bbox,
        # So we need to add dummy fields that are missing
        for gt_box, target, proposal in zip(gt_boxes, targets, proposals):
            gt_box.PRED_SCORES = torch.ones(len(gt_box), device=device)
            gt_box.PRED_LABELS = target.LABELS.long()

            # We need to add the pred_logits to match the set of fields from the proposals
            # Note: in particular here, we're checking whether we're a 1-stage or 2-stage detector
            num_classes = proposal.BOXES_PER_CLS.shape[1] // (2 * self.n_dim)
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

            # TODO add an option to add GT masks
            if proposal.has_field(BoxList.PredictionField.PRED_SEGMENTATION):
                gt_box.PRED_SEGMENTATION = proposal.PRED_SEGMENTATION

        return [BoxListOps.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

    def _replace_proposals_with_gt(self, proposals: list[BoxList], targets: list[BoxList]) -> list[BoxList]:
        """
        For the Scene Graph prediction task, we may want to have a fine control over
        which parts of the prediction are replaced with groundtruth annotation during testing.
        This way, we can find out which parts of the network cause the biggest performance degradation.
        """
        if not (self.cfg.TEST.RELATION.REPLACE_SEGMENTATION or
                self.cfg.TEST.RELATION.REMOVE_FALSE_POSITIVES or
                self.cfg.TEST.RELATION.REPLACE_MATCHED_BOXES):
            return proposals

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
