# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import ROIAlign, ROIAlign3D
from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.mask_head import MaskHeadTargets, MaskLogits
from scene_graph_prediction.modeling.utils import cat


class MaskRCNNLossComputation(torch.nn.Module):
    def __init__(self, discretization_size: tuple[int, ...]):
        super().__init__()
        if len(discretization_size) == 2:
            self.mask_align = ROIAlign(
                discretization_size,
                spatial_scale=1.,
                sampling_ratio=0
            )
        else:
            self.mask_align = ROIAlign3D(
                discretization_size,
                spatial_scale=1.,
                spatial_scale_depth=1.,
                sampling_ratio=0
            )

    def _prepare_targets(
            self,
            proposals: BoxHeadTestProposals,
            targets: MaskHeadTargets
    ) -> tuple[list[torch.LongTensor], list[torch.Tensor]]:
        """
        Takes a list of proposals and targets for a batch of images.
        Performs some matching between proposals and targets.
        """

        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(proposals_per_image) == 0:
                continue

            matched_idxs = proposals_per_image.MATCHED_IDXS
            segmentation_masks = targets_per_image.MASKS.get_mask_tensor()

            # Create targets using ROIAlign
            rois = torch.cat([matched_idxs[:, None], proposals_per_image.boxes], dim=1)
            masks_per_image = self.mask_align(segmentation_masks[:, None], rois)[:, 0]  # Need a channel temporary dim

            labels.append(targets_per_image.LABELS[matched_idxs])
            masks.append(masks_per_image)

        return labels, masks

    def forward(
            self,
            proposals: BoxHeadTestProposals,
            mask_logits: MaskLogits,
            targets: MaskHeadTargets
    ) -> torch.Tensor:
        labels, mask_targets = self._prepare_targets(proposals, targets)

        if len(labels) == 0:
            # No image with positive examples
            return torch.tensor(0., device=mask_logits.device, requires_grad=True)

        cat_labels = cat(labels, dim=0).long()
        cat_gt_seg = cat(mask_targets, dim=0)

        index = torch.arange(cat_labels.shape[0], device=cat_labels.device)
        cat_seg_logits = mask_logits[index, cat_labels]

        # Compute the Dice loss
        seg_loss_dice = self._squared_dice_loss(cat_seg_logits, cat_gt_seg)

        # # Compute the cross-entropy loss on the groundtruth foreground and predicted false positives
        # Note: nnDetection does not apply any non-linearity before loss call
        flat_logits = cat_seg_logits.view(cat_seg_logits.size(0), -1)
        pred_seg = torch.argmax(torch.cat([mask_logits[:, 0][:, None], cat_seg_logits[:, None]], dim=1), dim=1)
        gt_seg = cat_gt_seg.view(cat_gt_seg.size(0), -1).round().long()
        fg_union_mask = (pred_seg.view(cat_gt_seg.size(0), -1) + gt_seg) > 0
        # Manually compute the mean over all images
        ce_losses = []
        for flat_logits_image, gt_seg_image, fg_union_mask_image in zip(flat_logits, gt_seg, fg_union_mask):
            ce_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                flat_logits_image[fg_union_mask_image][None], gt_seg_image[fg_union_mask_image][None].float()
            ))
        seg_loss_ce = torch.mean(torch.stack(ce_losses))

        return (seg_loss_ce + seg_loss_dice) / 2

    @staticmethod
    def _squared_dice_loss(cat_seg_logits: torch.Tensor, cat_gt_seg: torch.Tensor) -> torch.FloatTensor:
        """Compute the squared dice loss for the binary masks. The average overall masks is computed."""
        flat_sigmoid_logits = cat_seg_logits.sigmoid().view(cat_seg_logits.size(0), -1)
        cat_flat_gt_seg = cat_gt_seg.view(cat_gt_seg.size(0), -1)
        intersection = (flat_sigmoid_logits * cat_flat_gt_seg).sum(1)
        denominator = (flat_sigmoid_logits ** 2 + cat_flat_gt_seg ** 2).sum(1)
        eps = 1e-6
        # noinspection PyTypeChecker
        return torch.mean(1. - (2 * intersection + eps) / (denominator + eps))


def build_roi_mask_loss_evaluator(
        cfg: CfgNode,
        predicted_mask_size: tuple[int, ...]
) -> MaskRCNNLossComputation:
    n_dim = cfg.INPUT.N_DIM
    assert len(predicted_mask_size) == n_dim + 1  # channel + n_dim
    return MaskRCNNLossComputation(predicted_mask_size[1:])
