# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.mask_head import MaskHeadTargets, MaskLogits
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList, AbstractMaskList


class MaskRCNNLossComputation(torch.nn.Module):
    def __init__(self, discretization_size: tuple[int, ...]):
        super().__init__()
        self.discretization_size = discretization_size

    def _project_masks_on_boxes(
            self,
            segmentation_masks: AbstractMaskList,
            proposal: BoxList
    ) -> torch.Tensor:
        """
        Given segmentation masks and the bounding boxes corresponding to the location of the masks in the image,
        this function crops and resizes the masks in the position defined by the boxes.
        This prepares the masks for them to be fed to the loss computation as the targets.
        """
        masks = []
        device = proposal.boxes.device
        proposal = proposal.convert(BoxList.Mode.zyxzyx)
        assert segmentation_masks.size == proposal.size, f"{segmentation_masks.size}, {proposal.size}"

        # Note: CPU computation bottleneck, this should be parallelized
        proposal_boxes = proposal.boxes.to(torch.device("cpu"))
        for segmentation_mask, box in zip(segmentation_masks, proposal_boxes):  # type: AbstractMaskList, torch.Tensor
            # Crop the masks, resize them to the desired resolution and then convert them to the tensor representation.
            cropped_mask = segmentation_mask.crop(box.tolist())
            scaled_mask = cropped_mask.resize(self.discretization_size)
            mask = scaled_mask.get_mask_tensor()
            masks.append(mask)

        if len(masks) == 0:
            return torch.tensor(0., device=device, requires_grad=True)

        return torch.stack(masks, dim=0).to(device, dtype=torch.float32)

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
            # noinspection DuplicatedCode
            matched_idxs = proposals_per_image.MATCHED_IDXS
            matched_targets = targets_per_image[matched_idxs.clamp(min=0)]
            labels_per_image = matched_targets.LABELS.long()

            # Mask scores are only computed on positive samples
            positive_indexes = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.MASKS
            segmentation_masks = segmentation_masks[positive_indexes]
            positive_proposals = proposals_per_image[positive_indexes]

            masks_per_image = self._project_masks_on_boxes(segmentation_masks, positive_proposals)

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def forward(
            self,
            proposals: BoxHeadTestProposals,
            mask_logits: MaskLogits,
            targets: MaskHeadTargets
    ) -> torch.Tensor:
        labels, mask_targets = self._prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        # noinspection PyTypeChecker
        positive_mask = labels > 0

        # torch.mean (in binary_cross_entropy_with_logits) doesn't accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mask_logits[positive_mask, labels[positive_mask]], mask_targets
        )

        return mask_loss


def build_roi_mask_loss_evaluator(cfg: CfgNode) -> MaskRCNNLossComputation:
    n_dim = cfg.INPUT.N_DIM
    res: int = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
    res_depth: int = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_DEPTH
    if n_dim == 2:
        return MaskRCNNLossComputation((res, res))
    elif n_dim == 3:
        return MaskRCNNLossComputation((res_depth, res, res))
    raise NotImplementedError
