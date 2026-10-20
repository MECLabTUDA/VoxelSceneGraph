# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets, RPNProposals
from scene_graph_prediction.modeling.abstractions.matcher import Matcher
from scene_graph_prediction.modeling.abstractions.sampler import Sampler
from scene_graph_prediction.modeling.utils.sampling import AllSampler
from ..default.sampling import FastRCNNSampling
from scene_graph_prediction.modeling.utils import BoxCoder, IoUMatcher
from scene_graph_prediction.structures import BoxList, BoxListOps


class HybridDetectorSampling(FastRCNNSampling):
    """
    Sampling RoIs.

    Note: adds a "regression_targets" field to BoxLists for *local* use ONLY
    (BoxLists still go through the BoxHead code though).
    It's only other use is for the loss computation.
    """

    def __init__(
            self,
            proposal_matcher: Matcher,
            fg_bg_sampler: Sampler,
            box_coder: BoxCoder,
            encode_targets: bool,
            attribute_on: bool,
            num_normal_fg_classes: int
    ):
        super().__init__(
            proposal_matcher=proposal_matcher,
            fg_bg_sampler=fg_bg_sampler,
            box_coder=box_coder,
            encode_targets=encode_targets,
            attribute_on=attribute_on
        )
        self.num_normal_fg_classes = num_normal_fg_classes

    def subsample(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
        """
        Add groundtruth fields to proposals (LABELS, REGRESSION_TARGETS, MATCHED_IDXS,
        and optionally ATTRIBUTES, IMPORTANCE).
        Perform the positive/negative sampling, and return the sampling mask.
        """
        # We need to enforce a normal importance for unique objects since the KnowledgeGuidedSampler does not do it
        # Note: previously, it wasn't an issue because, we detected unique objects from segmentation
        if targets:
            for target in targets:
                if target.has_field(BoxList.AnnotationField.IMPORTANCE):
                    target.IMPORTANCE[target.LABELS > self.num_normal_fg_classes] = 1.
        return super().subsample(proposals, targets)

    def _match_targets_to_proposals(self, proposal: BoxList, target: BoxList) -> tuple[BoxList, torch.LongTensor]:
        """
        :param proposal: a BoxList
        :param target: a BoxHeadTarget
        :returns: A BoxList with fields "labels", "attributes"
        """
        # To do class-wise matching, we use the same trick as for class-wise NMS
        max_size = proposal.size[0]
        BoxListOps.offset_classwise(proposal, max_size, BoxList.PredictionField.PRED_LABELS)
        BoxListOps.offset_classwise(target, max_size, BoxList.AnnotationField.LABELS)

        # We need to make sure that non-unique objects are never matched with unique objects and vice versa
        matched_idxs = self.proposal_matcher(target, proposal)

        # Then we need to offset back
        BoxListOps.offset_classwise(proposal, -max_size, BoxList.PredictionField.PRED_LABELS)
        BoxListOps.offset_classwise(target, -max_size, BoxList.AnnotationField.LABELS)

        # Fast RCNN only needs "labels" field for selecting the targets
        # Get the targets corresponding GT for each proposal
        # Note: need to clamp the indices because we can have a single GT in the image,
        # and matched_idxs can be -2, which goes out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    # Since we use the AllSampler, everything is kept anyway
    # def subsample(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
    #     """
    #     Add groundtruth fields to proposals (LABELS, REGRESSION_TARGETS, MATCHED_IDXS, and optionally ATTRIBUTES).
    #     Perform the positive/negative sampling, and return the sampling mask.
    #     """
    #     # The sampler takes care of selecting all properly matched objects
    #     keep = super().subsample(proposals, targets)
    #     # Just make sure that unique objects are always sampled even when the match is BAD
    #     # (which would cause the object to be ignored and not picked by the AllSampler)
    #     return [
    #         torch.logical_or(mask, proposal.LABELS > self.num_normal_fg_classes)
    #         for mask, proposal in zip(keep, proposals)
    #     ]


def build_hybrid_roi_box_samp_processor(
        cfg: CfgNode,
        box_coder: BoxCoder,
        encode_targets: bool
) -> HybridDetectorSampling:
    matcher = IoUMatcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        always_keep_best_match=False
    )

    # Note the sampler is actually not used
    fg_bg_sampler = AllSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    return HybridDetectorSampling(
        proposal_matcher=matcher,
        fg_bg_sampler=fg_bg_sampler,
        box_coder=box_coder,
        encode_targets=encode_targets,
        attribute_on=cfg.MODEL.ATTRIBUTE_ON,
        num_normal_fg_classes=cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1
    )
