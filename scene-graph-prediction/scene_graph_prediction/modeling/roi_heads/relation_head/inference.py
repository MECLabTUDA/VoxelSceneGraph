# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from ._utils.relation import obj_prediction_nms
from ...abstractions.box_head import ClassLogits
from ...abstractions.relation_head import RelationLogits, RelationHeadProposals, RelationHeadTestProposals


class RelationPostProcessor(torch.nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the final results.
    """

    def __init__(
            self,
            use_gt_box: bool = False,
            use_gt_box_label: bool = False,
            later_nms_pred_thres: float = 0.3,  # NMS threshold for after reclassification
            attribute_refinement_on: bool = False,
            disable_reclassification: bool = False
    ):
        super().__init__()
        if use_gt_box_label:
            assert use_gt_box, "Cannot use GT object labels without GT boxes..."
        self.use_gt_box = use_gt_box
        self.use_gt_box_label = use_gt_box_label
        self.later_nms_pred_thres = later_nms_pred_thres
        self.attribute_refinement_on = attribute_refinement_on
        self.disable_reclassification = disable_reclassification

    def forward(
            self,
            relation_logits: RelationLogits,
            refined_obj_logits: ClassLogits,
            rel_pair_idxs: list[torch.LongTensor],
            proposals: RelationHeadProposals,
            refined_att_logits: list[torch.Tensor] | None
    ) -> RelationHeadTestProposals:
        """
        First we handle the refined object detection:
        - apply NMS if necessary
        - adapt boxes if a reclassification occurred
        - set PRED_SCORES and PRED_LABELS fields
        Then we handle relation:
        - compute a score for each relation pair as the product of the scores in the triplet
        - set the REL_PAIR_IDXS, PRED_REL_CLS_SCORES, and PRED_REL_LABELS fields
        Finally handle attributes if activated.

        :param relation_logits: the relation logits
        :param refined_obj_logits: the  fine-tuned object logits from the relation model
        :param rel_pair_idxs: subject and object indices of each relation, the size of tensor is (num_rel, 2)
        :param proposals: bounding boxes that are used as reference, one for ech image
        :param refined_att_logits: the  fine-tuned object logits from the relation model

        :returns: one BoxList for each image, containing the extra fields pred_labels, pred_scores
                  (, pred_attributes), rel_pair_idxs, pred_rel_scores, pred_rel_labels
        """
        results = []
        # Loop over batch of images
        for i, (rel_logit, obj_logit, rel_pair_idx, proposal) in \
                enumerate(zip(relation_logits, refined_obj_logits, rel_pair_idxs, proposals)):
            # Compute the classification probability for each bbox
            obj_class_prob = torch.nn.functional.softmax(obj_logit, -1)
            # obj_class_prob[:, 0] = 0  # Set background score to 0
            n_bbox = obj_class_prob.shape[0]
            n_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box or self.disable_reclassification:
                # Just use the correct probability
                bbox_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred += 1
            else:
                # Apply NMS first
                obj_pred = obj_prediction_nms(
                    proposal.BOXES_PER_CLS.view(proposal.BOXES_PER_CLS.shape[0], obj_logit.shape[1], -1),
                    obj_logit,
                    self.later_nms_pred_thre
                )
                obj_score_ind = torch.arange(n_bbox, device=obj_logit.device) * n_obj_class + obj_pred
                bbox_scores = obj_class_prob.view(-1)[obj_score_ind]

            # Note: sanity check
            assert bbox_scores.shape[0] == n_bbox

            if not self.use_gt_box and not self.disable_reclassification:
                # mode==sgdet
                # Refine box prediction if gt not used (in case the classification changed)
                # Apply regression based on fine-tuned object class
                proposal.boxes = proposal.BOXES_PER_CLS[
                    torch.arange(obj_pred.shape[0], device=obj_pred.device),
                    obj_pred
                ]

            # Set the final PRED_LABELS and PRED_SCORES fields
            if self.use_gt_box_label:
                proposal.PRED_LABELS = proposal.LABELS  # (#obj, )
                proposal.PRED_SCORES = torch.ones_like(proposal.LABELS).float()  # (#obj, )
            elif not self.disable_reclassification:
                # If reclassification is not disabled, overwrite box head predictions
                proposal.PRED_LABELS = obj_pred  # (#obj, )
                proposal.PRED_SCORES = bbox_scores  # (#obj, )

            # Note: filtering out objects that are now predicted as background would mess up the indices
            # Sorting triples according to score production
            subj_scores = proposal.PRED_SCORES[rel_pair_idx[:, 0]]
            obj_scores = proposal.PRED_SCORES[rel_pair_idx[:, 1]]
            rel_class_prob = torch.nn.functional.softmax(rel_logit, -1)

            # Note: we sort based on the max foreground probability...
            rel_scores, _ = rel_class_prob[:, 1:].max(dim=1)
            #       ...however if the overall max prob is for background, then the predicted label is background
            _, rel_class = rel_class_prob.max(dim=1)
            # Note: we cannot filter out relations predicted as background
            #       because it would break non-graph-constraint metrics

            # Note: how about using weighted sum here?
            triple_scores = rel_scores * subj_scores * obj_scores
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)

            proposal.REL_PAIR_IDXS = rel_pair_idx[sorting_idx]  # (#rel, 2)
            proposal.PRED_REL_CLS_SCORES = rel_class_prob[sorting_idx]  # (#rel, #rel_class)
            proposal.PRED_REL_LABELS = rel_class[sorting_idx]  # (#rel, )

            # Update attributes if enabled
            if self.attribute_refinement_on:
                proposal.PRED_ATTRIBUTES = torch.sigmoid(refined_att_logits[i])

            # Should have fields : REL_PAIR_IDXS, PRED_REL_CLS_SCORES, PRED_REL_LABELS, PRED_LABELS, PRED_SCORES
            results.append(proposal)
        return results


def build_roi_relation_post_processor(cfg: CfgNode, predictor_supports_attr_refine: bool) -> RelationPostProcessor:
    return RelationPostProcessor(
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL,
        cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES,
        cfg.MODEL.ATTRIBUTE_ON and predictor_supports_attr_refine,
        cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION
    )
