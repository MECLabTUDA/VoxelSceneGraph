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
from ..utils.misc import LossComputationCfg


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

        # Some more compatibility checks
        if hasattr(roi_heads, "box"):
            assert roi_heads.box.require_one_stage_detector() and rpn.is_one_stage_detector(), \
                "The box head requires a one-stage detector, but only an RPN has been configured."
            # We can easily convert predictions from a one-stage detector to the output of an RPN
            self.simulate_two_stage = not roi_heads.box.require_one_stage_detector() and rpn.is_one_stage_detector()
        else:
            self.simulate_two_stage = False

        super().__init__(cfg, backbone, rpn, roi_heads)

    def forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[list[BoxList], LossDict]:
        """
        :param images: images to be processed
        :param targets: ground-truth boxes present in the image (optional)
        :param compute_loss: which loss should be computed (even when evaluating).

        :returns: The output from the model.
                  During training, it returns a dict[Tensor] which contains the losses.
                  During testing, it returns list[BoxList] contains additional fields
                  like `pred_scores`, `pred_labels` and `pred_masks` (for Mask R-CNN models).
        """
        if self.cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE and self.training:
            # This algorithm cannot be used for testing because we filter out images with no sampled relation
            # The filtering messes up the prediction ordering if we have a batch size > 1
            # Also it's not really relevant for testing...
            if self.cfg.MODEL.RELATION_ON:
                proposals, all_losses = self.rel_head_optimized_forward(images, targets, compute_loss)
            else:
                proposals, all_losses = self.roi_heads_optimized_forward(images, targets, compute_loss)
        else:
            proposals, all_losses = self.standard_forward(images, targets, compute_loss)

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
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[list[BoxList], LossDict]:
        """Standard pipeline where all images are handled at once."""
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_list = ImageList.to_image_list(images, self.n_dim)
        features, proposals, all_losses = self._prepare_rpn_proposals(image_list, targets, compute_loss)

        proposals, roi_head_losses = self.roi_heads(features, proposals, targets, compute_loss)
        all_losses.update(roi_head_losses)

        return proposals, all_losses

    def roi_heads_optimized_forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> LossDict:
        """
        Pipeline where the feature maps are computed for one image at a time and only pooled features are kept.
        This allows training on multiple images, while mitigating the maximum memory footprint.
        WARNING: only available for training roi heads (except relation). The implementation is very much no compatible.
        WARNING: we need a batch size of at least 1 (duh...).
        """
        assert not self.cfg.MODEL.RELATION_ON
        assert not compute_loss.compute_rpn_loss
        assert compute_loss.compute_roi_heads_loss
        assert not compute_loss.compute_rel_heads_loss

        device = torch.device(self.cfg.MODEL.DEVICE)
        images = ImageList.to_image_list(images, self.n_dim)
        assert len(images) > 0

        # Per-image, per-head intermediate results which can be used to compute a loss / post-process predictions
        all_proposals = []
        all_box_pre_computations = []
        all_attr_pre_computations = []
        all_mask_pre_computations = []
        all_kp_pre_computations = []
        for idx, cur_image in enumerate(images):
            # We need to get the ith image to ImageList with one image and then set the device
            cur_image = cur_image.to(device)
            cur_target = [targets[idx].to(device)] if targets is not None else None

            # We assume that the RPN is not trained with the optimized pipeline
            features, proposals, _ = self._prepare_rpn_proposals(cur_image, cur_target, compute_loss)

            box_pre_computations, attr_pre_computations, mask_pre_computations, kp_pre_computations = (
                self.roi_heads.sample_and_predict_roi_heads(features, proposals, cur_target)
            )

            all_proposals.append(proposals)
            all_box_pre_computations.append(box_pre_computations)
            all_attr_pre_computations.append(attr_pre_computations)
            all_mask_pre_computations.append(mask_pre_computations)
            all_kp_pre_computations.append(kp_pre_computations)

            # Delete whatever is not needed anymore
            del cur_image
            del cur_target
            del features

        # Aggregate intermediate results
        all_proposals = reduce(lambda a, b: a + b, all_proposals)

        def transpose_computations(pre_computations):
            return [
                None
                if None in comp_list
                else (  # Check whether we concatenate tensors or BoxLists
                    torch.cat(comp_list)
                    if isinstance(comp_list[0], torch.Tensor)
                    else reduce(lambda a, b: a + b, comp_list)  # Tuple of list of BoxLists to list of BoxLists
                )
                for comp_list in list(zip(*pre_computations))
            ]

        all_box_pre_computations = transpose_computations(all_box_pre_computations)
        all_attr_pre_computations = transpose_computations(all_attr_pre_computations)
        all_mask_pre_computations = transpose_computations(all_mask_pre_computations)
        all_kp_pre_computations = transpose_computations(all_kp_pre_computations)

        # Compute losses (we don't care about predictions)
        # noinspection PyTypeChecker
        all_losses = self.roi_heads.postprocess_roi_heads(
            all_proposals,
            all_box_pre_computations,
            all_attr_pre_computations,
            all_mask_pre_computations,
            all_kp_pre_computations
        )
        return all_proposals, all_losses

    def rel_head_optimized_forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[list[BoxList], LossDict]:
        """
        Pipeline where the feature maps are computed for one image at a time and only pooled features are kept.
        This allows training on multiple images, while mitigating the maximum memory footprint.
        WARNING: only available for non-relation training.
        TODO: improve implementation by assuming that we're always training
        """
        assert self.cfg.MODEL.RELATION_ON
        assert not compute_loss.compute_rpn_loss
        assert not compute_loss.compute_roi_heads_loss
        assert compute_loss.compute_rel_heads_loss

        device = torch.device(self.cfg.MODEL.DEVICE)
        images = ImageList.to_image_list(images, self.n_dim)

        # Per-image, per-head intermediate results which can be used to compute a loss / post-process predictions
        all_proposals = []
        all_pre_computations = []
        for idx, cur_image in enumerate(images):
            # We need to get the ith image to ImageList with one image and then set the device
            cur_image = cur_image.to(device)
            cur_target = [targets[idx].to(device)] if targets is not None else None

            # We assume that the RPN is not trained with the optimized pipeline
            features, proposals, _ = self._prepare_rpn_proposals(cur_image, cur_target, compute_loss)

            proposals, pre_computations = self.roi_heads.sample_and_predict_relation(
                features, proposals, cur_target, compute_loss
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
            compute_loss: LossComputationCfg = LossComputationCfg.none()
    ) -> tuple[FeatureMaps, BoxHeadTestProposals, LossDict]:
        """
        Handle the RPN prediction pipeline and proposals preparation for all scenarios, e.g.
        one-stage vs two-stage object detector, w/ vs w/o relation training.
        """
        rpn_losses = {}
        features = self.backbone(image_list.tensors)

        if not (self.cfg.MODEL.RELATION_ON and self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX):
            # Normal computation pipeline
            raw_rpn_predictions = self.rpn(image_list, features)
            proposals = self.rpn.post_process_predictions(raw_rpn_predictions, targets=targets)

            # Prevent any RPN to produce a loss when it shouldn't
            if compute_loss.compute_rpn_loss:
                rpn_losses.update(self.rpn.loss(raw_rpn_predictions, targets=targets))

            # If gradients are required, the grad_context will still have a reference, otherwise we may free the memory
            del raw_rpn_predictions

            # Check whether the RPN is a one-stage detector, i.e. we need to spoof the OBJECTNESS field in the proposals
            if self.simulate_two_stage:
                for proposal in proposals:
                    if not proposal.has_field(BoxList.PredictionField.OBJECTNESS):
                        proposal.OBJECTNESS = proposal.PRED_SCORES
                    # Then we need to delete fields that are otherwise not predicted
                    proposal.del_field(BoxList.PredictionField.PRED_SCORES)
                    proposal.del_field(BoxList.PredictionField.PRED_LOGITS)
                    proposal.del_field(BoxList.PredictionField.PRED_SEGMENTATION_LOGITS)
                    proposal.del_field(BoxList.PredictionField.PRED_LABELS)

            # Check whether to add GT annotation to predictions
            # Note: if we're simulating a two-stage decoder with a one-stage decoder,
            # then we can only call this method once we have simulated the OBJECTNESS field
            if self.cfg.MODEL.RPN.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN and self.training:
                proposals = self._add_rpn_gt_to_proposals(proposals, targets)  # No labels, with objectness

            # Check whether we need the one-stage detector to assign labels for ROI heads,
            # i.e. we need to compute losses in some heads, but there is no box head to assign the labels
            if (compute_loss.compute_roi_heads_loss or compute_loss.compute_rel_heads_loss) and \
                    (self.cfg.MODEL.BOX_ON or self.cfg.MODEL.ATTRIBUTE_ON or self.cfg.MODEL.MASK_ON or
                     self.cfg.MODEL.KEYPOINT_ON or self.cfg.MODEL.RELATION_ON) and \
                    not hasattr(self.roi_heads, "box"):
                self.rpn.assign_label_to_proposals(proposals, targets)

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
        WARNING: requires the proposals to have the OBJECTNESS field.
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
