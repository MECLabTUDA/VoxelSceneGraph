# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets, RPNProposals
from scene_graph_prediction.modeling.abstractions.matcher import Matcher
from scene_graph_prediction.modeling.abstractions.sampler import Sampler
from scene_graph_prediction.modeling.utils import BoxCoder, IoUMatcher
from scene_graph_prediction.modeling.utils.sampling import BalancedSampler
from scene_graph_prediction.structures import BoxList


class FastRCNNSampling:
    """
    Sampling RoIs.

    Note: adds a "regression_targets" field to BoxLists for *local* use ONLY
    (BoxLists still go through the BoxHead code though).
    It's only other use is for the loss computation.
    """

    def __init__(
            self,
            proposal_matcher: Matcher,
            fg_bg_sampler: BalancedSampler,
            box_coder: BoxCoder,
            encode_targets: bool,
            attribute_on: bool
    ):
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.encode_targets = encode_targets
        self.attribute_on = attribute_on

    def _match_targets_to_proposals(self, proposal: BoxList, target: BoxList) -> tuple[BoxList, torch.LongTensor]:
        """
        :param proposal: a BoxList
        :param target: a BoxHeadTarget
        :returns: A BoxList with fields "labels", "attributes"
        """
        matched_idxs = self.proposal_matcher(target, proposal)

        # Fast RCNN only need "labels" field for selecting the targets
        # Get the targets corresponding GT for each proposal
        # Note: need to clamp the indices because we can have a single GT in the image, 
        # and matched_idxs can be -2, which goes out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    def _prepare_targets(
            self,
            proposals: RPNProposals,
            targets: BoxHeadTargets
    ) -> tuple[list[torch.LongTensor], list[torch.LongTensor | None], list[torch.Tensor], list[torch.LongTensor]]:
        """
        Takes a list of proposals and targets for a batch of images.
        Performs some matching between proposals and targets.
        Also adds the field "matched_idxs" to the proposals to avoid
        having to match proposals with targets in other ROI heads.
        :returns: labels, and attributes as LongTensors; regression_targets as encoded boxes
        """
        labels = []
        attributes = []
        regression_targets = []
        matched_idxs = []

        # Iterate over batch of images
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # Handle empty proposals
            if len(proposals_per_image) == 0:
                device = proposals_per_image.boxes.device
                labels.append(torch.tensor([], dtype=torch.long, device=device))
                matched_idxs.append(torch.tensor([], dtype=torch.long, device=device))
                regression_targets.append(torch.tensor([], dtype=torch.long, device=device))
                attributes.append(torch.tensor([], dtype=torch.long, device=device))
                continue

            matched_targets, matched_idxs_per_image = self._match_targets_to_proposals(proposals_per_image,
                                                                                       targets_per_image)
            labels_per_image = matched_targets.LABELS.long()

            # Label background (below the low threshold)
            bg_indexes = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indexes] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_indexes = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_indexes] = Sampler.IGNORE

            # Compute regression targets
            if self.encode_targets:
                regression_targets_per_image = self.box_coder.encode(matched_targets.boxes, proposals_per_image.boxes)
            else:
                regression_targets_per_image = matched_targets.boxes

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)

            if self.attribute_on:
                attributes_per_image = matched_targets.ATTRIBUTES.long()
                attributes_per_image[bg_indexes] = 0
                attributes_per_image[ignore_indexes] = Sampler.IGNORE
                attributes.append(attributes_per_image)
            else:
                attributes.append(None)

        return labels, attributes, regression_targets, matched_idxs

    def subsample(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
        """
        Add groundtruth fields to proposals (LABELS, REGRESSION_TARGETS, MATCHED_IDXS, and optionally ATTRIBUTES).
        Perform the positive/negative sampling, and return the sampling mask.
        """
        labels, attributes, regression_targets, matched_idxs = self._prepare_targets(proposals, targets)

        proposals = list(proposals)
        # Iterate over images in batch
        # Add corresponding label and regression_targets information to the bounding boxes
        for labels, attributes, regression_targets, matched_idxs, proposal in \
                zip(labels, attributes, regression_targets, matched_idxs, proposals):
            proposal.LABELS = labels
            # Used for loss computation...
            proposal.REGRESSION_TARGETS = regression_targets
            # ...and this one also in other ROI heads
            proposal.MATCHED_IDXS = matched_idxs
            if self.attribute_on:
                proposal.ATTRIBUTES = attributes

        sampled_pos_mask, sampled_neg_mask = self.fg_bg_sampler(labels)
        return [
            pos_mask_img | neg_mask_img
            for pos_mask_img, neg_mask_img in zip(sampled_pos_mask, sampled_neg_mask)
        ]

    def assign_label_to_proposals(self, proposals: RPNProposals, targets: BoxHeadTargets):
        """
        Update the "labels" and "attributes" fields of the proposals after matching with GT.
        I.e. converts RPNProposals to BoxHeadTargets.
        A 0 in these tensor means that there was no match.
        Note: this is a light-weight version of self.subsample for relation detection.
        """

        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            matched_idxs = self.proposal_matcher(target, proposal)
            matched_targets: BoxList = target[matched_idxs.clamp(min=0)]

            labels_per_image = matched_targets.LABELS.long()
            labels_per_image[matched_idxs < 0] = 0
            proposals[img_idx].LABELS = labels_per_image

            if self.attribute_on:
                attributes_per_image = matched_targets.ATTRIBUTES.long()
                attributes_per_image[matched_idxs < 0, :] = 0
                proposals[img_idx].ATTRIBUTES = attributes_per_image


def build_roi_box_samp_processor(cfg: CfgNode, encode_targets: bool) -> FastRCNNSampling:
    matcher = IoUMatcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        always_keep_best_match=False
    )

    fg_bg_sampler = BalancedSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    return FastRCNNSampling(
        matcher,
        fg_bg_sampler,
        BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS, n_dim=cfg.INPUT.N_DIM),
        encode_targets,
        cfg.MODEL.ATTRIBUTE_ON
    )
