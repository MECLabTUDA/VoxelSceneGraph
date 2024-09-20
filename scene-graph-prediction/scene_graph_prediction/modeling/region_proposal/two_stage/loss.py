# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""This file contains specific functions for computing losses on the RPN file."""
from abc import abstractmethod, ABC
from typing import Callable

import torch
# noinspection PyPep8Naming
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.matcher import Matcher
from scene_graph_prediction.modeling.abstractions.region_proposal import ImageAnchors
from scene_graph_prediction.modeling.utils import BoxCoder, IoUMatcher, ATSSMatcher, HardNegativeSampler
from scene_graph_prediction.structures import BoxList, BoxListOps
from .._common.utils import concat_box_prediction_layers
from ...abstractions.sampler import Sampler


class RPNLossComputationBase(ABC):
    """
    This class computes the RPN loss.

    Note: adds a "matched_idxs" field to BoxLists for *local* use ONLY
    (BoxLists still go through the RPNModule code though).
    It's only other use is in the generate_rpn_labels function below, for loss computation.
    """

    def __init__(
            self,
            proposal_matcher: Matcher,
            box_coder: BoxCoder,
            generate_labels_func: Callable[[BoxList], torch.Tensor],
            weighted_training: bool
    ):
        """
        :param proposal_matcher: a matching algorithm for anchors and ground truth objects. See ...utils.matching.
        :param box_coder: a BoxCoder to convert int box coordinates to a more usable range.
        :param generate_labels_func: a function that given a BoxList returns a Tensor with labels,
                                     given the MATCHED_IDXS and LABELS fields.
        :param weighted_training: whether the ground truth objects are weighted unequally during training.
        """
        super().__init__()
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.generate_labels_func = generate_labels_func
        self.weighted_training = weighted_training

    def _match_targets_to_anchors(
            self,
            anchors: BoxList,
            target: BoxList,
            fields_to_copy: list
    ) -> tuple[BoxList, torch.LongTensor]:
        # We need to add weights if weighted training is enabled
        if self.weighted_training:
            fields_to_copy.append(BoxList.AnnotationField.IMPORTANCE)

        if len(target) == 0:
            device = anchors.boxes.device
            matched_targets = BoxList(torch.zeros_like(anchors.boxes), anchors.size)
            for field in fields_to_copy:
                matched_targets.add_field(
                    field,
                    torch.zeros(len(anchors), device=device),
                    target.fields_indexing_power[field]
                )
            return matched_targets, torch.full((len(anchors),), -1, device=device).long()

        matched_indexes = self.proposal_matcher(target, anchors)

        # RPN doesn't need any fields from target for creating the labels, so clear them all
        # RetinaNet needs the "labels" field however
        target = target.copy_with_fields(fields_to_copy)
        # Get the targets corresponding GT for each anchor
        # Note: need to clamp the indices because we can have a single GT in the image,
        # and matched_indexes can be -2, which goes out of bounds
        matched_targets = target[matched_indexes.clamp(min=0)]
        return matched_targets, matched_indexes

    def _prepare_targets(
            self,
            anchors: ImageAnchors,
            targets: list[BoxList],
            discard_occluded: bool,
            discard_in_between: bool,
            fields_to_copy: list,
            encode_targets: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Utility function used to do matching and preprocessing.
        :param anchors:
        :param targets:
        :param discard_occluded: Whether to discard anchors with VISIBILITY field 0.
        :param discard_in_between: Whether to discard anchors that are between IoU thresholds.
        :param fields_to_copy: Fields to copy during matching and that are needed to generate labels.
        :param encode_targets: Whether to encode targets or keep them as bounding box.
        :return: labels, targets, box weights
        """
        labels = []
        final_targets = []
        weights = []
        # Iterate for images in batch
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets, matched_idxs_per_image = self._match_targets_to_anchors(
                anchors_per_image,
                targets_per_image,
                fields_to_copy
            )
            matched_targets.MATCHED_IDXS = matched_idxs_per_image
            labels_per_image = self.generate_labels_func(matched_targets).long()

            # Discard anchors that go out of the boundaries of the image
            if discard_occluded:
                occluded = ~anchors_per_image.get_field(BoxList.PredictionField.VISIBILITY)
                labels_per_image[occluded] = Sampler.IGNORE

            # Discard indices that are between thresholds
            if discard_in_between:
                in_between = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[in_between] = Sampler.IGNORE

            if encode_targets:
                # Compute regression targets
                regression_targets_per_image = self.box_coder.encode(matched_targets.boxes, anchors_per_image.boxes)
                final_targets.append(regression_targets_per_image)
            else:
                final_targets.append(matched_targets.boxes)

            if self.weighted_training:
                # Need to set the importance to a normal level for objects that have not been matched
                importance = matched_targets.IMPORTANCE
                importance[matched_idxs_per_image < 0] = 1.
                # Ignore any object with a weight of 0
                labels_per_image[torch.abs(importance) < 1e-5] = Sampler.IGNORE
                weights.append(importance)
            else:
                # Default to a plain average computation
                weights.append(torch.ones(len(matched_targets), dtype=torch.float32, device=labels_per_image.device))

            labels.append(labels_per_image)

        return labels, final_targets, weights

    @abstractmethod
    def __call__(
            self,
            anchors: list[ImageAnchors],
            objectness: list[torch.Tensor],
            box_regression: list[torch.Tensor],
            targets: list[BoxList]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RPNLossComputation(RPNLossComputationBase):
    """This class computes the RPN loss."""

    def __init__(
            self,
            proposal_matcher: Matcher,
            fg_bg_sampler: Sampler,
            box_coder: BoxCoder,
            generate_labels_func: Callable[[BoxList], torch.Tensor],
            weighted_training: bool
    ):
        super().__init__(proposal_matcher, box_coder, generate_labels_func, weighted_training)
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(
            self,
            anchors: list[ImageAnchors],
            objectness: list[torch.Tensor],
            box_regression: list[torch.Tensor],
            targets: list[BoxList]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat_anchors = [BoxListOps.cat(anchors_per_image) for anchors_per_image in anchors]

        cat_objectness, cat_box_regression = concat_box_prediction_layers(objectness, box_regression)
        cat_objectness = cat_objectness.squeeze()

        labels, regression_targets, weights = self._prepare_targets(cat_anchors, targets, True, True, [])
        sampled_pos_masks, sampled_neg_masks = self.fg_bg_sampler(labels, cat_objectness)
        # Note: need binary masks so that we can cat them between images
        sampled_pos_masks = torch.cat(sampled_pos_masks, dim=0)
        sampled_neg_masks = torch.cat(sampled_neg_masks, dim=0)

        with torch.no_grad():
            cat_labels = torch.cat(labels, dim=0).float()
            cat_regression_targets = torch.cat(regression_targets, dim=0)
            cat_weights = torch.cat(weights, dim=0)

        num_pos = sampled_pos_masks.sum()
        if num_pos > 0:
            box_loss = torch.sum(
                torch.nn.functional.smooth_l1_loss(
                    cat_box_regression[sampled_pos_masks],
                    cat_regression_targets[sampled_pos_masks],
                    beta=1 / 9,
                    reduction="none"
                ) * cat_weights[sampled_pos_masks, None]) / \
                       (torch.sum(cat_weights[sampled_pos_masks]) * cat_box_regression.shape[1])
        else:
            box_loss = 0.

        # Assert to avoid indexing errors caused by a misconfiguration
        assert cat_labels.shape[0] == cat_box_regression.shape[0], "Shape mismatch. Anchors are misconfigured..."

        all_masks = torch.logical_or(sampled_pos_masks, sampled_neg_masks)
        if torch.any(all_masks):
            objectness_loss = torch.sum(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    cat_objectness[all_masks],
                    cat_labels[all_masks],
                    reduction="none"
                ) * cat_weights[all_masks, None]) / (torch.sum(cat_weights[all_masks]) * cat_labels.shape[1])
        else:
            objectness_loss = torch.tensor(0., device=sampled_pos_masks.device, requires_grad=True)

        return objectness_loss, box_loss


def build_rpn_loss_evaluator(
        cfg: CfgNode,
        box_coder: BoxCoder,
        num_anchors_per_lvl: int
) -> RPNLossComputation:
    # This function should be overwritten in RetinaNet
    # Here we generate binary labels
    def generate_rpn_labels(matched_targets: BoxList) -> torch.Tensor:
        matched_indexes = matched_targets.MATCHED_IDXS
        return matched_indexes >= 0

    match cfg.MODEL.RPN.MATCHER:
        case "IoUMatcher":
            matcher = IoUMatcher(
                cfg.MODEL.RPN.FG_IOU_THRESHOLD,
                cfg.MODEL.RPN.BG_IOU_THRESHOLD,
                always_keep_best_match=True,
            )
        case "ATSSMatcher":
            matcher = ATSSMatcher(
                num_anchors_per_lvl=num_anchors_per_lvl,
                num_candidates=cfg.MODEL.RPN.ATSS_NUM_CANDIDATES
            )
        case _:
            raise ValueError(f"Unknown matcher class (\"{cfg.MODEL.RPN.MATCHER}\")")

    fg_bg_sampler = HardNegativeSampler(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)

    return RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels,
        cfg.MODEL.WEIGHTED_BOX_TRAINING
    )
