# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from functools import reduce

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.mask_head import MaskLogits, MaskHeadTargets
from scene_graph_prediction.structures import BoxList, BinaryMaskList


class Masker:
    """
    Projects a set of masks in an image on the locations specified by the bounding boxes.
    Note: masks are zero-padded to improve the interpolation results.
    """

    def __init__(self, n_dim: int, threshold: float = 0.5, padding: int = 1):
        self.n_dim = n_dim
        self.threshold = threshold
        self.padding = padding

    def _expand_masks(self, mask: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        """Zero-pad a 1x(Dx)HxW mask and returns the new scale along each axis."""
        scales = [(s + self.padding * 2) / s for s in mask.shape[-self.n_dim:]]
        return torch.nn.functional.pad(mask, (1, 1) * self.n_dim, "constant", 0), scales

    def _expand_box(self, box: torch.Tensor, scales: list[float]) -> torch.Tensor:
        """Since the mask is padded, they contain a box slightly larger. So, we also have to update the boxes."""
        half_lengths = [(box[self.n_dim + dim] - box[dim] + 1) * .5 for dim in range(self.n_dim)]
        centers = [(box[self.n_dim + dim] + box[dim]) * .5 for dim in range(self.n_dim)]
        scaled_half_lengths = [length * scale for length, scale in zip(half_lengths, scales)]

        box_exp = torch.empty_like(box)
        for dim in range(self.n_dim):
            box_exp[dim] = centers[dim] - scaled_half_lengths[dim]
            box_exp[self.n_dim + dim] = centers[dim] + scaled_half_lengths[dim]

        return box_exp

    def _paste_mask_in_image(
            self,
            mask: torch.Tensor,
            box: torch.Tensor,
            dhw_image_size: tuple[int, ...],
    ) -> torch.Tensor:
        """Interpolates a mask to box shape and pastes it in a mask of the shape of the image."""
        mask, scales = self._expand_masks(mask.float())
        box = self._expand_box(box.float(), scales).round().to(dtype=torch.int32)

        lengths = [int(box[self.n_dim + dim] - box[dim] + 1) for dim in range(self.n_dim)]
        # Safeguard again empty boxes
        lengths = tuple(max(length, 1) for length in lengths)

        # Resize mask
        mask = torch.nn.functional.interpolate(
            mask[None],  # Reshape to Nx1xDxHxW from 1xDxHxW
            size=lengths,
            mode="bilinear" if self.n_dim == 2 else "trilinear",
            align_corners=False
        )[0, 0]

        mask = mask > self.threshold

        # Paste the interpolated mask into a tensor of the same shape as the image
        # Note: The padding is:
        #       - Before: the first coordinate of the box
        #       - After: the size of the image minus the last coordinate of the box minus 1
        #                (because this coordinate is still included in the box)
        # Note: The padding can make the generated mask larger than the image.
        #       However, allowing the padding to be negative causes an automatic crop to fix this issue.
        img_padding = tuple(
            (box[self.n_dim - dim].item(), (dhw_image_size[-dim] - box[2 * self.n_dim + -dim] - 1).item())
            for dim in range(1, self.n_dim + 1)
        )
        img_padding = reduce(lambda a, b: a + b, img_padding)
        im_mask = torch.nn.functional.pad(mask.to(torch.uint8), img_padding, "constant", 0)

        return im_mask

    def _forward_single_image(self, masks: torch.Tensor, proposal: BoxList) -> torch.Tensor:
        if len(proposal) == 0:
            # Return empty tensor of the right shape
            return masks.new_empty((0, 1) + tuple(masks.shape[-dim] for dim in range(self.n_dim, 0, -1)))

        proposal = proposal.convert(BoxList.Mode.zyxzyx)
        res = [self._paste_mask_in_image(mask, box, proposal.size) for mask, box in zip(masks, proposal.boxes)]
        return torch.stack(res, dim=0)[:, None]

    def __call__(self, masks: list[torch.Tensor], boxes: list[BoxList]) -> list[torch.Tensor]:
        """
        :param masks: list of Nx1(xD)xHxW per image.
        :param boxes:
        :return:
        """
        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        results = []
        for im_masks, im_boxes in zip(masks, boxes):
            assert im_masks.shape[0] == len(im_boxes), "Number of objects should be the same."
            result = self._forward_single_image(im_masks, im_boxes)
            results.append(result)

        return results


class MaskPostProcessor:
    """
    From the results of the CNN, post-process the masks by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output by the CNN)
    and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally project the masks in the image
    according to the locations in boxes.
    """

    def __init__(self, n_dim: int, masker: Masker):
        assert masker.n_dim == n_dim
        self.n_dim = n_dim
        self.masker = masker

    def __call__(self, x: MaskLogits, proposals: BoxHeadTestProposals) -> MaskHeadTargets:
        mask_prob = x.sigmoid()

        # Select masks corresponding to the predicted classes
        labels = torch.cat([bbox.PRED_LABELS for bbox in proposals])
        index = torch.arange(mask_prob.shape[0], device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        proposals_per_image = [len(box) for box in proposals]
        mask_prob = mask_prob.split(proposals_per_image, dim=0)
        mask_prob = self.masker(mask_prob, proposals)

        for prob, proposal in zip(mask_prob, proposals):  # type: torch.Tensor, BoxList
            proposal.PRED_MASKS = BinaryMaskList(prob[:, 0], proposal.size)  # Need to remove the channel dim

        return proposals


def build_roi_mask_post_processor(cfg: CfgNode) -> MaskPostProcessor:
    return MaskPostProcessor(
        cfg.INPUT.N_DIM,
        Masker(n_dim=cfg.INPUT.N_DIM, threshold=cfg.MODEL.ROI_MASK_HEAD.SCORE_THRESH, padding=1)
    )
