# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from functools import reduce

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import ROIAlign, ROIAlign3D
from scene_graph_prediction.structures import BoxList, BoxListOps
from scene_graph_prediction.utils.miscellaneous import get_pred_masks, get_gt_masks
from .roi_relation_mask_feature_extractors import build_relation_mask_feature_extractor
from ..box_head.roi_box_feature_extractors import build_feature_extractor
from ...abstractions.backbone import AnchorStrides, FeatureMaps
from ...abstractions.relation_head import ROIRelationFeatureExtractor, RelationHeadFeatures
from ...registries import ROI_RELATION_FEATURE_EXTRACTORS
from ...utils import ROIHeadName, cat


@ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(ROIRelationFeatureExtractor):
    """
    Heads for Motifs for relation triplet classification.
    Note: supports 2D and 3D.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)
        self.cfg = cfg
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        # Note: might get ignored by the box feature extractor
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = build_feature_extractor(
                cfg, in_channels, anchor_strides, half_out=True,
                cat_all_levels=pool_all_levels,
                roi_head=ROIHeadName.BoundingBox
            )
            self.att_feature_extractor = build_feature_extractor(
                cfg, in_channels, anchor_strides, half_out=True,
                cat_all_levels=pool_all_levels,
                roi_head=ROIHeadName.Attribute
            )
            self.representation_size = self.feature_extractor.representation_size * 2
        else:
            self.feature_extractor = build_feature_extractor(
                cfg, in_channels, anchor_strides,
                cat_all_levels=pool_all_levels,
                roi_head=ROIHeadName.BoundingBox
            )
            self.representation_size = self.feature_extractor.representation_size

        # Union rectangle size
        # Note: we also want to exploit finer details in the masks, so we use a size 4 times greater than the pooler's,
        #        and reduce it through the convolutions / pooling
        self.rect_conv = build_relation_mask_feature_extractor(cfg, in_channels)
        assert self.n_dim == self.rect_conv.n_dim

        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_MASKS:
            # Used to interpolate binary masks of objects to the correct shape for feature extraction
            if self.n_dim == 2:
                self.mask_align = ROIAlign(
                    self.rect_conv.get_orig_rect_size(),
                    spatial_scale=1.,
                    sampling_ratio=0
                )
            else:
                self.mask_align = ROIAlign3D(
                    self.rect_conv.get_orig_rect_size(),
                    spatial_scale=1.,
                    spatial_scale_depth=1.,
                    sampling_ratio=0
                )
        else:
            self.mask_align = None

    def forward(
            self,
            x: FeatureMaps,
            proposals: list[BoxList],
            rel_pair_idxs: list[torch.LongTensor]
    ) -> RelationHeadFeatures:
        """
        For each image, we do:
        - create mask tensors of the same shape as pooler outputs
        - resize boxes to this shape
        - compute the resized binary masks
        - concatenate them
        - use the box feature extractor to generate the missing channels
        - add these box features to the raw pooled features of the visual union of objects
        - compute final features with the extractor
        """
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            # Get the box for subject and object for the relation batch
            subj_proposal: BoxList = proposal[rel_pair_idx[:, 0]]
            obj_proposal: BoxList = proposal[rel_pair_idx[:, 1]]
            # Compute the union box
            union_proposal = BoxListOps.union(subj_proposal, obj_proposal)
            union_proposals.append(union_proposal)

            if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_MASKS:
                # Use predicted binary masks
                subj_masks, obj_masks = self._prepare_binary_masks(proposal, rel_pair_idx, union_proposal)
            else:
                # Compute rectangular masks for subjects and objects
                subj_masks, obj_masks = self._prepare_rect_masks(
                    subj_proposal, obj_proposal, union_proposal, x[0].device
                )

            rect_input = torch.stack((subj_masks, obj_masks), dim=1)  # (num_rel, 2, *rect_size)
            rect_inputs.append(rect_input)

        # Rectangle features size (total_num_rel, in_channels, *rect_size)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        # Rectangle features size (total_num_rel, in_channels, *rect_size), just like the output of a pooler
        rect_features = self.rect_conv(rect_inputs)

        # Union visual feature size (total_num_rel, in_channels, *rect_size)
        union_vis_pooled = self.feature_extractor.pooler(x, union_proposals)

        # (total_num_rel, representation_siwe)
        union_features = self.feature_extractor.forward_without_pool(union_vis_pooled + rect_features)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)

        return union_features

    def _prepare_binary_masks(
            self,
            proposal: BoxList,
            rel_pair_idx: torch.LongTensor,
            union_proposal: BoxList
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Given proposals and relation indexes, resize and sample binary masks (to #N_RELxDxHxW)."""
        # TODO add tests for this
        # Convert any sort of predicted segmentation to binary masks format
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            bin_masks = get_gt_masks(proposal)
        else:
            bin_masks = get_pred_masks(proposal)
        # Compute inverse mapping to know which union boxes need to be sampled for each object
        box_idx_to_rel_idxs = {}
        for box_idx in range(len(proposal)):
            # noinspection PyTypeChecker
            box_idx_to_rel_idxs[box_idx] = torch.nonzero(torch.any(rel_pair_idx == box_idx, 1)).view(-1)
        # Then we compute the boxes and batch ids
        concat_boxes = cat([union_proposal.boxes[matches] for matches in box_idx_to_rel_idxs.values()])
        ids = cat([
            torch.full((len(matches), 1), i, dtype=union_proposal.boxes.dtype, device=union_proposal.boxes.device)
            for i, matches in box_idx_to_rel_idxs.items()
        ])
        rois = torch.cat([ids, concat_boxes], dim=1)
        interp_bin_masks = self.mask_align(bin_masks.float(), rois)

        # Finally, we only have to index the sampled masks,
        # such that we have a list for all subjects (and one for all objects)
        interp_bin_masks_by_box = interp_bin_masks.split([len(matches) for matches in box_idx_to_rel_idxs.values()])

        subj_bin_masks = []
        for rel_idx, subj_idx in enumerate(rel_pair_idx[:, 0]):
            subj_bin_masks.append(
                interp_bin_masks_by_box[subj_idx.cpu().item()][box_idx_to_rel_idxs[subj_idx.cpu().item()] == rel_idx]
            )
        subj_bin_masks = cat(subj_bin_masks)

        obj_bin_masks = []
        for rel_idx, obj_idx in enumerate(rel_pair_idx[:, 1]):
            obj_bin_masks.append(
                interp_bin_masks_by_box[obj_idx.cpu().item()][box_idx_to_rel_idxs[obj_idx.cpu().item()] == rel_idx]
            )
        obj_bin_masks = cat(obj_bin_masks)

        # Note: remove channel dim because the torch.stack later adds it back
        return subj_bin_masks[:, 0], obj_bin_masks[:, 0]

    def _prepare_rect_masks(
            self,
            subj_proposal: BoxList,
            obj_proposal: BoxList,
            union_proposal: BoxList,
            device: torch.device
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Given subject and object proposals, compute rectangular masks matching the resized boxes."""

        def create_rect_mask(boxes) -> torch.FloatTensor:
            # Compute rectangular subject and object masks in resized shape
            # Note: the rect_size is so small that doing this instead of indexing is completely fine
            # noinspection PyUnresolvedReferences
            return reduce(
                lambda a, b: a & b,
                [
                    dummy_ranges[dim] >= (
                            (boxes[:, dim] - union_proposal.boxes[:, dim]) /
                            (union_proposal.boxes[:, self.n_dim + dim] - union_proposal.boxes[:, dim]) *
                            rect_size[dim]
                    ).floor().view(-1, *n_dim_ones).long()
                    for dim in range(self.n_dim)
                ] +
                [
                    dummy_ranges[dim] <= (
                            (boxes[:, self.n_dim + dim] - union_proposal.boxes[:, dim]) /
                            (union_proposal.boxes[:, self.n_dim + dim] - union_proposal.boxes[:, dim]) *
                            rect_size[dim]
                    ).ceil().view(-1, *n_dim_ones).long()
                    for dim in range(self.n_dim)
                ]
            ).float()

        # TODO add tests for this
        n_dim_ones = (1,) * self.n_dim
        num_rel = len(subj_proposal)
        subj_proposal = subj_proposal.convert(BoxList.Mode.zyxzyx)
        obj_proposal = obj_proposal.convert(BoxList.Mode.zyxzyx)
        rect_size = self.rect_conv.get_orig_rect_size()

        # Compute per-axis positional embedding
        # Note: we need this ugly indexing to expand in the right direction
        dummy_ranges = [
            torch.arange(rect_size[dim], device=device)
            .view((1,) * (dim + 1) + (-1,) + (1,) * (self.n_dim - dim - 1)).expand(num_rel, *rect_size)
            for dim in range(self.n_dim)
        ]
        return create_rect_mask(subj_proposal.boxes), create_rect_mask(obj_proposal.boxes)


def build_roi_relation_feature_extractor(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides
) -> ROIRelationFeatureExtractor:
    relation_feature_extractor = ROI_RELATION_FEATURE_EXTRACTORS[cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR]
    return relation_feature_extractor(cfg, in_channels, anchor_strides)
