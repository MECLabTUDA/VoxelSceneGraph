# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .loss import build_roi_attribute_loss_evaluator
from .roi_attribute_predictors import build_roi_attribute_predictor
from .._utils import split_logits_for_each_image
from ..box_head.roi_box_feature_extractors import build_feature_extractor
from ...abstractions.attribute_head import AttributeHeadFeatures, AttributeHeadProposals, \
    ROIAttributeHead as AbstractROIAttributeHead, AttributeLogits
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import AttributeHeadLossDict
from ...utils import ROIHeadName


class ROIAttributeHead(AbstractROIAttributeHead):
    """Generic Attribute Head class."""

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)

        self.cfg = cfg.clone()
        # Feature extractors for attributes were (before the refactoring) just a copy of the ones for the box head
        self.feature_extractor = build_feature_extractor(
            cfg, in_channels, anchor_strides,
            half_out=self.cfg.MODEL.ATTRIBUTE_ON,
            roi_head=ROIHeadName.Attribute
        )
        self.predictor = build_roi_attribute_predictor(cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = build_roi_attribute_loss_evaluator(cfg)

    def forward(self, features: FeatureMaps, proposals: BoxHeadTestProposals) -> AttributeLogits:
        """
        Note: Attribute head is fixed when we train the relation head.
        Note: the sampling/proposal-gt matching has already been handled by the box head. It also handled fields.
        Note: regarding BoxList fields:
        - proposals needs the "attributes" field
        - the field "attribute_logits" is added, no existing field is erased
        :param features: extracted from RPN
        :param proposals: extracted from box_head
        """
        x: AttributeHeadFeatures = self.feature_extractor(features, proposals)
        attribute_logits = self.predictor(x)
        return attribute_logits

    def post_process_predictions(
            self,
            attribute_logits: AttributeLogits,
            proposals: BoxHeadTestProposals
    ) -> AttributeHeadProposals:
        # Adds field "attribute_logits"
        split_logits_for_each_image(proposals, BoxList.PredictionField.ATTRIBUTE_LOGITS, attribute_logits)
        return proposals

    def loss(self, attribute_logits: AttributeLogits, proposals: BoxHeadTestProposals) -> AttributeHeadLossDict:
        loss_attribute = self.loss_evaluator(proposals, attribute_logits)
        return {"loss_attribute": loss_attribute}
