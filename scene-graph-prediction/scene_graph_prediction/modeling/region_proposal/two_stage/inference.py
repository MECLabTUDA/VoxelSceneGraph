# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, BoxListOps
from .._common.utils import permute_and_flatten
from ...abstractions.region_proposal import ImageAnchors, Objectness, BoundingBoxRegression, FeatureMapsObjectness, \
    FeatureMapsBoundingBoxRegression, RPNProposals
from ...utils import BoxCoder


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the proposals to the heads.
    Note: class needs to be a Module to automatically know hen we're training or testing.
    """

    def __init__(
            self,
            pre_nms_top_n_train: int,
            post_nms_top_n_train: int,
            pre_nms_top_n_test: int,
            post_nms_top_n_test: int,
            nms_thresh: float,
            n_dim: int,
            box_coder: BoxCoder,
            fpn_post_nms_per_batch: bool = True
    ):
        super().__init__()
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.post_nms_top_n_train = post_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.n_dim = n_dim
        self.box_coder = box_coder
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def forward(
            self,
            anchors: list[ImageAnchors],
            objectness: FeatureMapsObjectness,
            box_regression: FeatureMapsBoundingBoxRegression,
            targets: list[BoxList] | None = None
    ) -> RPNProposals:
        """Returns the post-processed anchors, after applying box decoding and NMS."""
        sampled_boxes = []  # List (over image batch) of list (over maps) of BoxList
        num_levels = len(objectness)
        # From [img: [feat lvl: [anchors]]] to [feat lvl: [img: [anchors]]]; same format as objectness/box_regression
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            # Linter issues with zip for some reason
            sampled_boxes.append(self._forward_for_single_feature_map(a, o, b))

        # From [feat lvl: [img: [anchors]]] to [img: [feat lvl: [anchors]]]
        # Then concatenate across feature maps for each image
        boxlists = [BoxListOps.cat(maps_boxlists) for maps_boxlists in zip(*sampled_boxes)]

        if num_levels > 1:
            boxlists = self._select_over_all_levels(boxlists)

        return boxlists

    def _forward_for_single_feature_map(
            self,
            anchors: list[BoxList],
            objectness: Objectness,
            box_regression: BoundingBoxRegression
    ) -> RPNProposals:
        """
        TopK for RPN then NMS.
        :param anchors: List of BoxLists (for an image batch) at a specific feature level
        :param objectness: Tensor of size N, A, *zyx_lengths (H, W in 2D)
        :param box_regression: Tensor of size N, A * 2 * n_dim, *lengths_zyx
        """
        n, a, *zyx_lengths = objectness.shape

        # Put in the same format as anchors
        flat_objectness = permute_and_flatten(objectness, n, a, 1, *zyx_lengths).view(n, -1)
        flat_box_regression = permute_and_flatten(box_regression, n, a, 2 * self.n_dim, *zyx_lengths)
        num_anchors = a * np.prod(zyx_lengths)

        pre_nms_top_n = self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
        pre_nms_top_n = min(pre_nms_top_n, num_anchors)
        topk_objectness, topk_idx = flat_objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        # Compute the sigmoid only after the topk call as large objectness may get all flattened to 1.
        topk_objectness = topk_objectness.sigmoid()

        batch_idx = torch.arange(n, device=objectness.device)[:, None]
        topk_box_regression = flat_box_regression[batch_idx, topk_idx]

        concat_anchors = torch.cat([a.boxes for a in anchors], dim=0)
        topk_concat_anchors = concat_anchors.reshape(n, -1, 2 * self.n_dim)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(topk_box_regression.view(-1, 2 * self.n_dim),
                                          topk_concat_anchors.view(-1, 2 * self.n_dim))
        proposals = proposals.view(n, pre_nms_top_n, 2 * self.n_dim)

        result = []
        # Iterate over images in batch and keep only relevant boxes
        image_shapes = [box.size for box in anchors]
        post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test
        for proposal, score, im_shape in zip(proposals, topk_objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode=BoxList.Mode.zyxzyx)
            boxlist.OBJECTNESS = score
            boxlist = BoxListOps.clip_to_image(boxlist, remove_empty=True)
            boxlist, _ = BoxListOps.nms(
                boxlist,
                self.nms_thresh,
                max_proposals=post_nms_top_n,
                score_field=BoxList.PredictionField.OBJECTNESS
            )
            result.append(boxlist)

        return result

    def _select_over_all_levels(self, boxlists: RPNProposals) -> RPNProposals:
        """
        Select topK for FPN but no box NMS.
        Different behavior during training and during testing:
          - During training, post_nms_top_n is over *all* (over the whole batch) the proposals combined
          - During testing, it is over the proposals for each image
        Note: it should be per image, and not per batch.
        However, to be consistent with Detectron, the default is per batch.
        :param boxlists: BoxList containing boxes across feature levels for each image in a batch.
        """
        post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test

        # fpn_post_nms_per_batch seems to only be True during training
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat([boxlist.OBJECTNESS for boxlist in boxlists], dim=0)
            box_sizes = [len(boxlist) for boxlist in boxlists]

            post_nms_top_n = min(post_nms_top_n, len(objectness))
            _, indexes_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)

            indexes_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            indexes_mask[indexes_sorted] = 1
            indexes_mask = indexes_mask.split(box_sizes)

            for i in range(len(boxlists)):
                boxlists[i] = boxlists[i][indexes_mask[i]]

        else:
            for i in range(len(boxlists)):
                objectness = boxlists[i].OBJECTNESS
                post_nms_top_n = min(post_nms_top_n, len(objectness))
                _, indexes_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
                boxlists[i] = boxlists[i][indexes_sorted]

        return boxlists


def build_rpn_postprocessor(cfg: CfgNode, rpn_box_coder: BoxCoder) -> RPNPostProcessor:
    return RPNPostProcessor(
        pre_nms_top_n_train=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN,
        post_nms_top_n_train=cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN,
        pre_nms_top_n_test=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST,
        post_nms_top_n_test=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST,
        nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        n_dim=cfg.INPUT.N_DIM,
        box_coder=rpn_box_coder,
        fpn_post_nms_per_batch=cfg.MODEL.FPN.POST_NMS_PER_BATCH
    )
