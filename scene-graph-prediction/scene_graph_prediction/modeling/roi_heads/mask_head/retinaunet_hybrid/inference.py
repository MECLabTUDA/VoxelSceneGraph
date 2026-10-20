# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.mask_head import MaskLogits, MaskHeadTargets
from ..default.inference import Masker, MaskPostProcessor


class MaskPostProcessorHybrid(MaskPostProcessor):
    """
    From the results of the CNN, post-process the masks by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output by the CNN)
    and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally project the masks in the image
    according to the locations in boxes.
    """

    def __init__(self, n_dim: int, masker: Masker, num_normal_fg_classes: int):
        super().__init__(n_dim, masker)
        self.num_normal_fg_classes = num_normal_fg_classes

    def __call__(self, x: MaskLogits, proposals: BoxHeadTestProposals) -> MaskHeadTargets:
        proposals = super().__call__(x, proposals)

        for proposal in proposals:
            # Find unique objects
            pred_labels = proposal.PRED_LABELS
            unique_obj_idxs = torch.nonzero(pred_labels > self.num_normal_fg_classes)
            # Replace the predicted mask with one obtained from the predicted semantic seg
            # Note: the semantic segmentation is of the size of the padded image and needs to be sliced
            slicer = tuple(slice(0, s) for s in proposal.size)
            for idx in unique_obj_idxs:
                # noinspection PyUnresolvedReferences
                proposal.PRED_MASKS.masks[idx.item()] = (proposal.PRED_SEGMENTATION == pred_labels[idx])[slicer]

        return proposals


def build_roi_mask_hybrid_post_processor(cfg: CfgNode) -> MaskPostProcessorHybrid:
    return MaskPostProcessorHybrid(
        cfg.INPUT.N_DIM,
        Masker(n_dim=cfg.INPUT.N_DIM, threshold=cfg.MODEL.ROI_MASK_HEAD.SCORE_THRESH, padding=1),
        cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1
    )
