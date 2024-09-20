# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets, BoxHeadTestProposal, \
    BoxHeadTestProposals
from scene_graph_prediction.modeling.utils import BoxCoder
from scene_graph_prediction.structures import BoxList, BoxListOps

_SIZE_T = tuple[int, ...]


class PostProcessor(torch.nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the final results.
    """

    def __init__(
            self,
            box_coder: BoxCoder,
            score_thresh: float = 0.05,
            nms: float = 0.5,
            post_nms_per_cls_topn: int = 300,
            nms_filter_duplicates: bool = False,
            detections_per_img: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            bbox_aug_enabled: bool = False
    ):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.post_nms_per_cls_topn = post_nms_per_cls_topn
        self.nms_filter_duplicates = nms_filter_duplicates
        self.detections_per_img = detections_per_img
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.n_dim = box_coder.n_dim

    # noinspection DuplicatedCode
    def forward(
            self,
            features: torch.Tensor,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,
            proposals: BoxHeadTargets
    ) -> tuple[torch.Tensor, BoxHeadTestProposals]:
        """
        Given extracted features, class logits and box_regression:
        - perform NMS
        - add "pred_labels", and "pred_scores" field to the BoxLists corresponding to the prediction
        - keep only features for the selected boxes
        Note: proposals need to have the PRED_LOGITS field.
        Note: applies NMS and sorts the predicted boxes by score.

        :returns:
            features for selected boxes
            one BoxList for each image, containing the extra fields labels and scores.
        """
        class_prob = torch.nn.functional.softmax(class_logits, -1)

        image_shapes = [box.size for box in proposals]
        boxes_per_image = [len(box) for box in proposals]
        concat_boxes = torch.cat([a.boxes for a in proposals], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -2 * self.n_dim:]
        # Add rpn regression offset to the original proposals
        # tensor of size (num_box, 2 * n_dim * num_cls)
        proposals_boxes = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)

        if self.cls_agnostic_bbox_reg:
            proposals_boxes = proposals_boxes.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        features = features.split(boxes_per_image, dim=0)
        proposals_boxes = proposals_boxes.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        nms_features = []
        for i, (prob, boxes_per_img, image_shape) in enumerate(zip(class_prob, proposals_boxes, image_shapes)):
            # Converts the Tensor to a BoxList with field pred_scores
            boxlist = self._prepare_boxlist(boxes_per_img, prob, image_shape)
            # Cannot remove empty, otherwise we don't have a box for each class, and it messes up the reshaping later
            boxlist = BoxListOps.clip_to_image(boxlist, remove_empty=False)

            # We used to filter only when not bbox_aug, but we can actually do it multiple times just fine
            boxlist, orig_indexes, boxes_per_cls = self.filter_results(boxlist, num_classes)
            boxlist = self._add_fields_from_pre_processing_boxlist(
                proposals[i],
                orig_indexes,
                boxlist,
                boxes_per_cls
            )
            nms_features.append(features[i][orig_indexes])

            results.append(boxlist)

        nms_features = torch.cat(nms_features, dim=0)
        return nms_features, results

    # noinspection DuplicatedCode
    def filter_results(
            self,
            boxlist: BoxList,
            num_classes: int
    ) -> tuple[BoxHeadTestProposal, torch.LongTensor, torch.FloatTensor]:
        """
        Return bounding-box detection results by thresholding on scores, applying NMS and sort by score.
        Require the "pred_cls_scores" field.
        Returns a BoxList with the fields "pred_scores", and "pred_labels".
        :returns:
            result: BoxList with predicted positive matches (score > self.score_thresh), maybe after NMS
            orig_clswise_match_indexes: indexes of matched predictions
            boxes_per_cls: per-class predicted bounding boxes of matched predictions (#boxes, #num classes, #2 * n_dim)
        """
        # Unwrap the boxlist to avoid additional overhead.
        # If we had multi-class NMS, we could perform this directly on the boxlist
        boxes_per_cls = boxlist.boxes.reshape(-1, num_classes, 2 * self.n_dim)
        cls_scores = boxlist.PRED_CLS_SCORES.reshape(-1, num_classes)

        device = cls_scores.device
        results_for_classes: list[BoxList] = []  # List of BoxList containing positive results for each class
        orig_clswise_match_indexes = []

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        positive_cls_matches = cls_scores > self.score_thresh
        for j in range(1, num_classes):
            indexes = positive_cls_matches[:, j].nonzero().squeeze(1)

            boxlist_for_class = boxlist[indexes]
            # Select for jth class
            boxlist_for_class.boxes = boxes_per_cls[indexes, j]
            boxlist_for_class.PRED_SCORES = cls_scores[indexes, j]

            if len(boxlist) > 1:
                boxlist_for_class, keep = BoxListOps.nms(
                    boxlist_for_class,
                    self.nms,
                    max_proposals=self.post_nms_per_cls_topn,
                    score_field=BoxList.PredictionField.PRED_SCORES
                )
                indexes = indexes[keep]

            num_labels = len(boxlist_for_class)
            boxlist_for_class.PRED_LABELS = torch.full((num_labels,), j, dtype=torch.int64, device=device)

            results_for_classes.append(boxlist_for_class)
            orig_clswise_match_indexes.append(indexes)

        # Note: according to Neural-MOTIFS, we need remove duplicate bbox i.e. bbox with multiple scores above thr
        if self.nms_filter_duplicates:
            assert len(orig_clswise_match_indexes) == num_classes - 1

            # Update positive_cls_matches with boxes kept after NMS
            # Set all bg to zero
            positive_cls_matches[:, 0] = 0
            for j in range(1, num_classes):
                positive_cls_matches[:, j] = 0
                orig_idx = orig_clswise_match_indexes[j - 1]
                positive_cls_matches[orig_idx, j] = 1

            # Keep only the label with maximum score
            dist_scores = cls_scores * positive_cls_matches.float()
            scores_pre, labels_pre = dist_scores.max(1)
            final_indexes = scores_pre.nonzero()
            assert final_indexes.dim() != 0
            final_indexes = final_indexes.squeeze(1)

            # Note: it doesn't matter that we create a new BoXlist here since all fields will be copied later
            result = BoxList(boxes_per_cls[final_indexes, labels_pre], boxlist.size, mode=BoxList.Mode.zyxzyx)
            result.PRED_SCORES = scores_pre
            result.PRED_LABELS = labels_pre.long()
            result.PRED_CLS_SCORES = cls_scores

            orig_clswise_match_indexes = final_indexes
        else:
            # No filter: just cat the list of class-wise indexes to tensor
            result = BoxListOps.cat(results_for_classes)
            orig_clswise_match_indexes = torch.cat(orig_clswise_match_indexes, dim=0)

        # Sort predictions by confidence
        _, sort_ind = result.PRED_SCORES.view(-1).sort(dim=0, descending=True)
        orig_clswise_match_indexes = orig_clswise_match_indexes[sort_ind]
        result = result[sort_ind]

        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            result = result[: self.detections_per_img]
            orig_clswise_match_indexes = orig_clswise_match_indexes[: self.detections_per_img]
        return result, orig_clswise_match_indexes, boxes_per_cls[orig_clswise_match_indexes]

    def _prepare_boxlist(
            self,
            boxes: torch.Tensor,
            scores: torch.Tensor,
            image_shape: _SIZE_T,
    ) -> BoxList:
        """
        Returns BoxList from `boxes` and adds probability scores information as an extra field.
        The detections in each row originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list of object detection confidence
        scores for each of the object classes in the dataset (including the background class).
        `scores[i, j]`` corresponds to the box at `boxes[i, j * 2 * n_dim:(j + 1) * 2 * n_dim]`.
        Note: we cannot use a copy of the proposal here, since we get  n_obj times as many boxes
        """
        boxes = boxes.reshape(-1, 2 * self.n_dim)
        boxlist = BoxList(boxes, image_shape, mode=BoxList.Mode.zyxzyx)
        boxlist.PRED_CLS_SCORES = scores
        return boxlist

    @staticmethod
    def _add_fields_from_pre_processing_boxlist(
            pre_proc_proposal: BoxList,
            orig_indexes: torch.LongTensor,
            post_proc_proposal: BoxHeadTestProposal,
            boxes_per_cls: torch.Tensor
    ) -> BoxList:
        """
        Since the pre-processing boxlist cannot be copied, we have to copy some fields after bbox-sampling.
        """
        post_proc_proposal_with_fields = pre_proc_proposal[orig_indexes]
        # Overwrite boxes
        post_proc_proposal_with_fields.boxes = post_proc_proposal.boxes
        # Copy all fields ("pred_cls_scores", "pred_labels", "pred_scores")
        for field in post_proc_proposal.fields():
            post_proc_proposal_with_fields.add_field(field, post_proc_proposal.get_field(field))
        # Add the BOXES_PER_CLS field
        post_proc_proposal_with_fields.BOXES_PER_CLS = boxes_per_cls

        return post_proc_proposal_with_fields


# noinspection DuplicatedCode
def build_roi_box_postprocessor(cfg: CfgNode) -> PostProcessor:
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights, n_dim=cfg.INPUT.N_DIM)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    post_nms_per_cls_topn = cfg.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN
    nms_filter_duplicates = cfg.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES

    return PostProcessor(
        box_coder,
        score_thresh,
        nms_thresh,
        post_nms_per_cls_topn,
        nms_filter_duplicates,
        detections_per_img,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
