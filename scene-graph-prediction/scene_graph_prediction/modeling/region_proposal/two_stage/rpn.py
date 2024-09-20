# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from scene_graph_prediction.modeling.region_proposal.two_stage.inference import build_rpn_postprocessor
from scene_graph_prediction.modeling.region_proposal.two_stage.loss import build_rpn_loss_evaluator
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import BoxCoder
from scene_graph_prediction.structures import ImageList, BoxList, BoxListOps
from .._common.anchor_generator import build_anchor_generator
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import ImageAnchors, FeatureMapsObjectness, FeatureMapsBoundingBoxRegression, \
    RPN, RPNProposals


class RPNModule(RPN[tuple[list[ImageAnchors], FeatureMapsObjectness, FeatureMapsBoundingBoxRegression]]):
    """
    Module for RPN computation.
    Takes feature maps from the backbone and RPN proposals and losses.
    Works for both FPN and non-FPN.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__()

        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM

        anchor_generator = build_anchor_generator(cfg, anchor_strides)
        rpn_head_class = RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head_class(
            cfg,
            in_channels,
            cfg.MODEL.RPN.RPN_MID_CHANNEL,
            anchor_generator.num_anchors_per_level()
        )
        assert head.n_dim == self.n_dim

        if cfg.MODEL.MASK_ON:
            assert not cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION, \
                "Masks are on: the RPN requires individual segmentation masks."

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0) * self.n_dim, n_dim=self.n_dim)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector = build_rpn_postprocessor(cfg, rpn_box_coder)
        self.loss_evaluator = build_rpn_loss_evaluator(cfg, rpn_box_coder, anchor_generator.num_anchors_per_level())

    def forward(self, images: ImageList, features: FeatureMaps):
        """
        :param images: Images for which we want to compute the predictions.
        :param features: Features computed from the images that are used for computing the predictions.
                          Each tensor in the list correspond to different feature levels

        :returns:
            anchors: a per-level list of anchors for the batch
            objectness: a per-level list of objectness for the batch
            box_regression: a per-level list of regressions for the batch
        """
        assert images.n_dim == self.n_dim
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        return anchors, objectness, rpn_box_regression

    def post_process_predictions(self, args, targets: list[BoxList] | None = None) -> RPNProposals:
        """
        :param args: see self.forward().
        :param targets: Ground truth boxes present in the image (optional). Used to add GT objects during training.
        :return: The predicted boxes from the RPN, one BoxList per image.
        """
        anchors, objectness, rpn_box_regression = args
        if self.training and self.cfg.MODEL.RPN_ONLY:
            # In RPN_ONLY mode, there is no further training than the RPN, so we skip any post-processing.
            # Note: in RPN_ONLY mode these BoxLists don't have the OBJECTNESS field,
            #       but it should not matter given the downstream code.
            boxes = [BoxListOps.cat(anc) for anc in anchors]
        else:
            with torch.no_grad():
                boxes: RPNProposals = self.box_selector(anchors, objectness, rpn_box_regression, targets)
        return boxes

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        """
        :param args: see self.forward().
        :param targets: Ground truth boxes present in the image.
        :returns: The losses for the model.
        """
        anchors, objectness, rpn_box_regression = args
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
        return {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}
