# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import ROIAlign, ROIAlign3D
from scene_graph_prediction.modeling.abstractions.backbone import AnchorStrides, FeatureMaps
from scene_graph_prediction.modeling.abstractions.box_head import RPNProposals, BoxHeadTestProposals, \
    BoxHeadFeatures, BoxHeadTargets, ClassLogits, BboxRegression
from scene_graph_prediction.modeling.abstractions.loss import BoxHeadLossDict
from scene_graph_prediction.modeling.utils.label_assignment import assign_label_to_proposals_always_match_special
from ..retinaunet_hybrid import ROIBoxHeadHybrid
from scene_graph_prediction.modeling.roi_heads.box_head.roi_mask_feature_extractors import \
    build_mask_feature_extractor


class ROIBoxHeadHybridSegGrounded(ROIBoxHeadHybrid):
    """
    Box Head designed to work with a one-stage hybrid Retina U-Net.
    Works as a regular Box Head, but objects detected from segmentation do not get reclassified:
    - non-unique objects get a new classification score to remove false positives
    - their bounding box gets refined
    Note: ignores MODEL.CLS_AGNOSTIC_BBOX_REG.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels, anchor_strides)
        # Init mask feature extractor
        self.mask_feature_extractor = build_mask_feature_extractor(cfg, in_channels=1, out_channels=in_channels)

        # Used to interpolate binary masks of objects to the correct shape for feature extraction
        if self.n_dim == 2:
            self.mask_align = ROIAlign(
                self.mask_feature_extractor.input_size(),
                spatial_scale=1.,
                sampling_ratio=0
            )
        else:
            self.mask_align = ROIAlign3D(
                self.mask_feature_extractor.input_size(),
                spatial_scale=1.,
                spatial_scale_depth=1.,
                sampling_ratio=0
            )

    def forward(
            self, features: FeatureMaps, proposals: RPNProposals
    ) -> tuple[BoxHeadFeatures, ClassLogits, BboxRegression]:
        # 0. Perform ROIALign on visual feature
        vis_pooled = self.feature_extractor.pooler(features, proposals)

        # Extract features from the semantic segmentation
        # 1. We need to convert the semantic seg to binary mask by keeping only voxels for the box's class
        # 2. Perform ROIAlign by class
        masks = []
        for proposal in proposals:
            seg = proposal.PRED_SEGMENTATION
            # noinspection PyTypeChecker
            bin_masks = torch.stack([seg == c for c in range(1, self.num_fg_classes + 1)])[:, None].float()  # Bx1xDxHxW
            rois = torch.cat([proposal.PRED_LABELS[:, None] - 1, proposal.boxes], dim=1)
            interp_bin_masks = self.mask_align(bin_masks, rois)
            masks.append(interp_bin_masks)
        masks = torch.cat(masks, dim=0)

        # 3. Compute mask features
        mask_pooled = self.mask_feature_extractor(masks)

        # Extract features that will be fed to the final classifier.
        # The feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor.forward_without_pool(vis_pooled + mask_pooled)

        if self.feature_extractor.is_mask_head_compatible:
            # Need to convert and flatten features for the predictor
            # However, we cannot return the flattened features as they are not compatible with the mask head
            x_pred = self.avg_pool(x).squeeze()
        else:
            x_pred = x

        # Final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x_pred)

        return x, class_logits, box_regression


class ROIRelationReadyBoxHeadHybridSegGrounded(ROIBoxHeadHybridSegGrounded):
    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        assert cfg.MODEL.REGION_PROPOSAL == "RetinaUNetHybrid" and not cfg.MODEL.RETINANET.TWO_STAGE
        super().__init__(cfg, in_channels, anchor_strides)

    def assign_label_to_proposals(self, proposals: RPNProposals, targets: BoxHeadTargets) -> list[torch.BoolTensor]:
        """We don't actually need to sample for Relation training, but we need to follow the interface."""
        # Assign labels like a hybrid Retina U-Net
        assign_label_to_proposals_always_match_special(
            proposals, targets, self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, self.num_normal_fg_classes
        )

        # Return dummy mask
        device = torch.device(self.cfg.MODEL.DEVICE)
        # noinspection PyTypeChecker
        return [torch.ones(len(prop), device=device, dtype=torch.bool) for prop in proposals]

    def post_process_predictions(
            self,
            x: BoxHeadFeatures,
            class_logits: ClassLogits,
            box_regression: BboxRegression,
            proposals: BoxHeadTestProposals
    ) -> tuple[BboxRegression, BoxHeadTestProposals]:
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            return x, proposals
        return super().post_process_predictions(x, class_logits, box_regression, proposals)

    def loss(
            self, class_logits: ClassLogits, box_regression: BboxRegression, proposals: BoxHeadTestProposals
    ) -> BoxHeadLossDict:
        raise RuntimeError("No BoxHead loss should be computed when training a RelationHead.")
