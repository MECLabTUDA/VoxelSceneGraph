# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import ROIAlign3D
from scene_graph_prediction.structures import BoxList, BoxListOps
from ..box_head.roi_box_feature_extractors import DeepConvFeatureExtractor
from ...abstractions.backbone import FeatureMaps
from ...abstractions.relation_head import ROIRelationFeatureExtractor, RelationHeadFeatures
from ...utils import cat
from ...utils.pooler import Pooler


class RelationFeatureExtractorICH(ROIRelationFeatureExtractor):
    """
    Heads for Motifs for relation triplet classification.
    Note: supports 2D and 3D.
    """

    def __init__(self, cfg: CfgNode):
        in_channels = cfg.INPUT.N_CHANNELS
        super().__init__(cfg, in_channels)
        self.cfg = cfg
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim == 3
        anchor_strides = (1, 1, 1,),

        self.feature_extractor = DeepConvFeatureExtractor(
            # +2 for the two binary masks
            cfg, in_channels + 2, anchor_strides, half_out=False, cat_all_levels=True
        )
        self.representation_size = self.feature_extractor.representation_size

        # Used to interpolate binary masks of objects to the correct shape for feature extraction
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        resolution_depth = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH
        self.roi_align = ROIAlign3D(
            (resolution_depth, resolution, resolution),
            spatial_scale=1.,
            spatial_scale_depth=1.,
            sampling_ratio=0
        )

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
            if rel_pair_idx.numel() == 0:
                # Empty BoxList
                union_proposals.append(proposal[rel_pair_idx[:, 0]])
                rect_inputs.append(torch.zeros((0, 2, *self.roi_align.output_size), device=proposal.boxes.device))
                continue

            # Get the box for subject and object for the relation batch
            subj_proposal: BoxList = proposal[rel_pair_idx[:, 0]]
            obj_proposal: BoxList = proposal[rel_pair_idx[:, 1]]

            # Compute the union box
            # We first compute the normal union between subject and object
            union_proposal = BoxListOps.union(subj_proposal, obj_proposal)

            # Then we compute the union box between ventricle system and midline
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                union_structures = BoxListOps.union(
                    proposal[torch.nonzero(proposal.LABELS == 2)[0]],
                    proposal[torch.nonzero(proposal.LABELS == 3)[0]],
                )
            else:
                union_structures = BoxListOps.union(
                    proposal[torch.nonzero(proposal.PRED_LABELS == 2)[0]],
                    proposal[torch.nonzero(proposal.PRED_LABELS == 3)[0]],
                )
            # We repeat these boxes to match the number of relations
            union_structures.boxes = torch.tile(union_structures.boxes, (rel_pair_idx.shape[0], 1))

            # Finally we do the union between both sets
            union_proposal = BoxListOps.union(union_proposal, union_structures)
            union_proposals.append(union_proposal)

            assert self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_MASKS
            # Use predicted binary masks
            subj_masks, obj_masks = self._prepare_binary_masks(proposal, rel_pair_idx, union_proposal)

            rect_input = torch.stack((subj_masks, obj_masks), dim=1)  # (num_rel, 2, *rect_size)
            rect_inputs.append(rect_input)

        # Rectangle features size (total_num_rel, 2, *rect_size)
        rect_inputs = torch.cat(rect_inputs, dim=0)

        # Union visual feature size (total_num_rel, in_channels, *rect_size)
        # We skip the pooler, as we replace it with a single ROIAlign on the single scale
        # Note: this means that we also have to select the correct feature map (at the reight scale)
        img_inputs = self.roi_align(x[0], Pooler._convert_to_roi_format(union_proposals))

        # Combine the aligned image with the two aligned masks (total_num_rel, in_channels + 2, *rect_size)
        combined_inputs = torch.cat((img_inputs, rect_inputs), dim=1)

        # (total_num_rel, representation_size)
        union_features = self.feature_extractor.forward_without_pool(combined_inputs)

        return union_features

    def _prepare_binary_masks(
            self,
            proposal: BoxList,
            rel_pair_idx: torch.LongTensor,
            union_proposal: BoxList
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Given proposals and relation indexes, resize and sample binary masks (to #N_RELxDxHxW)."""
        # For this method to work, we need these fields explicitly.
        # Converting from a semantic segmentation would undo any improvement.
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            bin_masks = proposal.MASKS
        else:
            bin_masks = proposal.PRED_MASKS
        bin_masks = bin_masks.get_mask_tensor()[:, None]  # Convert from AbstractMaskList and add a channel dim

        # Compute inverse mapping to know which union boxes need to be sampled for each object
        # Note: we typically use ROIAlign layers with a batch of images.
        #       Here an "image" is an object mask that get resampled for different bboxes (one per sampled relation).
        box_idx_to_rel_idxs = {}
        # For each bbox in a BoxList, find all relation indices that contain this bbox
        for box_idx in range(len(proposal)):
            # noinspection PyTypeChecker
            box_idx_to_rel_idxs[box_idx] = torch.nonzero(torch.any(rel_pair_idx == box_idx, 1)).view(-1)
        # TODO Optimally, we don't want to resize twice if two relations have the exact same union box
        #  E.g. this happens because for each bleeding, we have the exact same bbox with the ventricle system and the midline

        # Then we compute the boxes and batch ids
        concat_boxes = cat([union_proposal.boxes[matches] for matches in box_idx_to_rel_idxs.values()])
        ids = cat([
            torch.full((len(matches), 1), i, dtype=union_proposal.boxes.dtype, device=union_proposal.boxes.device)
            for i, matches in box_idx_to_rel_idxs.items()
        ])
        rois = torch.cat([ids, concat_boxes], dim=1)
        interp_bin_masks = self.roi_align(bin_masks.float(), rois)

        # Finally, we only have to index the sampled masks,
        # such that we have a list for all subjects (and one for all objects)
        interp_bin_masks_by_box = interp_bin_masks.split([len(matches) for matches in box_idx_to_rel_idxs.values()])

        # For each subject/object, we need to find the correct cropping
        # So we use the subject/object id to find the correct set of croppings in interp_bin_masks_by_box
        # Then we need to find the one for this specific relation.
        # So we just use the list of all relation indices for this object and
        # create a binary mask with one match to index these croppings.
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


def build_roi_relation_feature_extractor(cfg: CfgNode) -> ROIRelationFeatureExtractor:
    return RelationFeatureExtractorICH(cfg)
