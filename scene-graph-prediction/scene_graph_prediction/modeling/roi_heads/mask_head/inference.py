# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.mask_head import MaskLogits, MaskHeadTargets
from scene_graph_prediction.structures import BoxList


class Masker:
    """Projects a set of masks in an image on the locations specified by the bounding boxes."""

    # TODO adapt code for 3D where masks may not be MxMxM
    def __init__(self, n_dim: int, threshold: float = 0.5, padding: int = 1):
        super().__init__()
        self.n_dim = n_dim
        self.threshold = threshold
        self.padding = padding

    def _expand_masks(self, mask: torch.Tensor, padding: int) -> tuple[torch.Tensor, float]:
        """Zero-pad MxMxM boxes and returns the new scale."""
        n = mask.shape[0]
        m = mask.shape[-1]
        pad2 = 2 * padding
        scale = (m + pad2) / m
        padded_mask = mask.new_zeros((n, 1,) + tuple(mask.shape[dim + 2] + pad2 for dim in range(1, mask.dim())))

        slicer = (slice(None), slice(None)) + tuple(slice(padding, -padding) for _ in range(self.n_dim))
        padded_mask[slicer] = mask
        return padded_mask, scale

    def _expand_boxes(self, boxes: torch.Tensor, scale: float) -> torch.Tensor:
        """Rescales the boxes around their center based on the scale computed when padding masks."""
        half_lengths = [(boxes[:, self.n_dim + dim] - boxes[:, dim]) * .5 for dim in range(self.n_dim)]
        centers = [(boxes[:, self.n_dim + dim] + boxes[:, dim]) * .5 for dim in range(self.n_dim)]
        scaled_half_lengths = [length * scale for length in half_lengths]

        boxes_exp = torch.zeros_like(boxes)
        for dim in range(self.n_dim):
            boxes_exp[:, dim] = centers[dim] - scaled_half_lengths[dim]
            boxes_exp[:, self.n_dim + dim] = centers[dim] + scaled_half_lengths[dim]

        return boxes_exp

    def _paste_mask_in_image(
            self,
            mask: torch.Tensor,
            box: torch.Tensor,
            zyx_image_size: tuple[int, ...],
            thresh: float = 0.5,
            padding: int = 1
    ) -> torch.Tensor:
        """Interpolates a mask to box shape and pastes it in a mask of the shape of the image."""
        # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
        mask = mask.float()
        box = box.float()

        # FIXME what's the point of padding masks? we only do interpolation anyway...
        padded_mask, scale = self._expand_masks(mask, padding=padding)
        mask = padded_mask[0, 0]
        box = self._expand_boxes(box[None], scale)[0].to(dtype=torch.int32)

        lengths = [int(box[self.n_dim + dim] - box[dim] + 1) for dim in range(self.n_dim)]
        lengths = tuple(max(length, 1) for length in lengths)

        # Resize mask
        mask = torch.nn.functional.interpolate(
            mask[None, None],  # Reshape to NxCxDxHxW from DxHxW
            size=tuple(lengths),
            mode="bilinear" if self.n_dim == 2 else "trilinear",
            align_corners=False
        )[0, 0]

        if thresh >= 0:
            mask = mask > thresh

        im_mask = torch.zeros(tuple(zyx_image_size), dtype=torch.uint8)

        starts = [max(box[dim], 0) for dim in range(self.n_dim)]
        ends = [min(box[self.n_dim + dim], zyx_image_size[dim]) for dim in range(self.n_dim)]

        im_slicer = tuple(slice(starts[dim], ends[dim]) for dim in range(self.n_dim))
        mask_slicer = tuple(slice(starts[dim] - box[dim], ends[dim] - box[dim]) for dim in range(self.n_dim))

        im_mask[im_slicer] = mask[mask_slicer]
        return im_mask

    def _forward_single_image(self, masks: torch.Tensor, boxes: BoxList) -> torch.Tensor:
        boxes = boxes.convert(BoxList.Mode.zyxzyx)
        res = [
            self._paste_mask_in_image(mask, box, boxes.size, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.boxes)
        ]
        if len(res) > 0:
            return torch.stack(res, dim=0)[:, None]
        # Return empty tensor of the right shape
        return masks.new_empty((0, 1), tuple(masks.shape[-dim] for dim in range(self.n_dim, 0, -1)))

    def __call__(self, masks: list[torch.Tensor], boxes: list[BoxList]) -> list[torch.Tensor]:
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

    def __init__(self, n_dim: int, masker: Masker | None = None):
        super().__init__()
        if masker:
            assert masker.n_dim == n_dim
        self.n_dim = n_dim
        self.masker = masker

    def __call__(self, x: MaskLogits, boxes: BoxHeadTestProposals) -> MaskHeadTargets:
        mask_prob = x.sigmoid()

        # Select masks corresponding to the predicted classes
        labels = torch.cat([bbox.PRED_LABELS for bbox in boxes])
        num_masks = x.shape[0]
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        results = []
        for prob, box in zip(mask_prob, boxes):  # type: torch.Tensor, BoxList
            bbox = box.copy_with_all_fields()
            bbox.PRED_MASKS = prob
            results.append(bbox)

        return results


def build_roi_mask_post_processor(cfg: CfgNode) -> MaskPostProcessor:
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(n_dim=cfg.INPUT.N_DIM, threshold=mask_threshold, padding=1)
    else:
        masker = None

    return MaskPostProcessor(cfg.INPUT.N_DIM, masker)
