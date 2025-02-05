# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets, BoxHeadTestProposal, \
    BoxHeadTestProposals
from scene_graph_prediction.modeling.utils import BoxCoder
from scene_graph_prediction.structures import BoxList, BoxListOps

_SIZE_T = tuple[int, ...]


class HybridPostProcessor:
    """See ROIBoxHeadHybrid."""
    def __init__(
            self,
            box_coder: BoxCoder,
            num_fg_classes: int,
            num_normal_fg_classes: int,  # Non-unique classes
            score_thresh: float = 0.05,
            nms: float = 0.5,
            detections_per_img: int = 100,
            bbox_aug_enabled: bool = False
    ):
        self.num_fg_classes = num_fg_classes
        self.num_normal_fg_classes = num_normal_fg_classes

        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.box_coder = box_coder
        self.bbox_aug_enabled = bbox_aug_enabled
        self.n_dim = box_coder.n_dim

    def __call__(
            self,
            features: torch.Tensor,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,
            proposals: BoxHeadTargets
    ) -> tuple[torch.Tensor, BoxHeadTestProposals]:
        """
        Given extracted features, class logits and box_regression:
        - Replace the classification score of non-unique objects
        - Update the box of all objects
        - Perform NMS
        - Add "pred_labels", and "pred_scores" field to the BoxLists corresponding to the prediction
        - Keep only features for the selected boxes
        Note: applies NMS and sorts the predicted boxes by score.

        :returns:
            features for selected boxes
            one BoxList for each image, containing the extra fields labels and scores.
        """
        # TODO handle the BOXES_PER_CLS field
        image_shapes = [box.size for box in proposals]
        boxes_per_image = [len(box) for box in proposals]
        cat_boxes = torch.cat([a.boxes for a in proposals], dim=0)

        # Add rpn regression offset to the original proposals
        # tensor of size (num_box, 2 * n_dim * num_cls)
        proposals_boxes = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), cat_boxes)

        features = features.split(boxes_per_image, dim=0)
        proposals_boxes = proposals_boxes.split(boxes_per_image, dim=0)
        class_logits = class_logits.split(boxes_per_image, dim=0)

        results = []
        nms_features = []
        for proposal, feat, logits, boxes_per_img, image_shape in \
                zip(proposals, features, class_logits, proposals_boxes, image_shapes):
            # One-stage predicted labels
            orig_pred_labels = proposal.PRED_LABELS

            # Extract the corresponding new score and new box
            index = torch.tensor(list(range(logits.shape[0])))
            new_pred_logits = logits[index, orig_pred_labels]
            map_indexes = proposal.n_dim * 2 * orig_pred_labels[:, None] + \
                          torch.tensor(list(range(proposal.n_dim * 2)), device=orig_pred_labels.device)
            new_boxes = boxes_per_img[index[:, None], map_indexes]

            # Update boxes of all objects
            proposal.boxes = new_boxes
            # Cannot remove empty to avoid messing up the feature selection at the end
            proposal = BoxListOps.clip_to_image(proposal, remove_empty=False)

            # Update the score and relevant logit of all normal (non-unique) objects
            normal_objects = orig_pred_labels <= self.num_normal_fg_classes
            proposal.PRED_SCORES[normal_objects] = new_pred_logits[normal_objects].sigmoid()
            proposal.PRED_LOGITS[index, orig_pred_labels][normal_objects] = new_pred_logits[normal_objects]

            # Filter-out false positives and empty boxes
            proposal, kept_features = self.filter_results(proposal, feat)
            nms_features.append(kept_features)
            results.append(proposal)

        nms_features = torch.cat(nms_features, dim=0)
        return nms_features, results

    # noinspection DuplicatedCode
    def filter_results(
            self,
            boxlist: BoxList,
            features: torch.FloatTensor
    ) -> tuple[BoxHeadTestProposal, torch.FloatTensor]:
        """
        Return bounding-box detection results by thresholding on scores, applying NMS and sort by score.
        Empty boxes are also removed.
        :returns:
            result: BoxList with predicted positive matches (score > self.score_thresh), maybe after NMS
            features: corresponding to the selected boxes
        """
        # Keep by score
        keep_score = boxlist.PRED_SCORES > self.score_thresh
        boxlist = boxlist[keep_score]
        features = features[keep_score]

        # Keep not empty
        keep_not_empty = BoxListOps.volume(boxlist) > 1
        boxlist = boxlist[keep_not_empty]
        features = features[keep_not_empty]

        # Do classwise NMS
        boxlist, keep = BoxListOps.nms_classwise(
            boxlist,
            self.nms,
            max_proposals=self.detections_per_img,
            score_field=BoxList.PredictionField.PRED_SCORES
        )
        features = features[keep]

        # Sort predictions by confidence
        _, sort_ind = boxlist.PRED_SCORES.sort(dim=0, descending=True)
        boxlist = boxlist[sort_ind]
        features = features[sort_ind]

        # noinspection PyTypeChecker
        return boxlist, features


# noinspection DuplicatedCode
def build_hybrid_roi_box_postprocessor(cfg: CfgNode, box_coder: BoxCoder) -> HybridPostProcessor:
    return HybridPostProcessor(
        box_coder=box_coder,
        num_fg_classes=cfg.INPUT.N_OBJ_CLASSES - 1,
        num_normal_fg_classes=cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1,
        score_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH,
        nms=cfg.MODEL.ROI_HEADS.NMS,
        detections_per_img=cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG,
        bbox_aug_enabled=cfg.TEST.BBOX_AUG.ENABLED
    )
