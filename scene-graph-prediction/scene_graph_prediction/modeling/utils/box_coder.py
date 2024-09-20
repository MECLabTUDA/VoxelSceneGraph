# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into the representation used for training the regressors.
    Note: supports ND.
    """

    def __init__(self, weights: tuple[float, ...], n_dim: int, bbox_xform_clip: float = math.log(1024. / 16)):
        """
        :param weights: zyxdhw ordered.
        :param n_dim:
        :param bbox_xform_clip:
        """
        assert len(weights) == 2 * n_dim
        self.weights = weights
        self.n_dim = n_dim
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, target_boxes: torch.Tensor, anchors: torch.Tensor):
        """
        Encodes a set of proposals with respect to some reference boxes.
        Note: self.decode(self.encode(reference_boxes, proposals), proposals) == reference_boxes
        :param target_boxes: Reference zyxzyx boxes.
        :param anchors: zyxzyx boxes to be encoded relative to the reference boxes (targets).
        :returns: encoded zyxdhw boxes
        """
        assert target_boxes.shape[1] == self.n_dim * 2
        assert anchors.shape[1] == self.n_dim * 2

        anc_lengths, anc_centers = self._get_lengths_centers(anchors)
        tgt_lengths, tgt_centers = self._get_lengths_centers(target_boxes)

        targets_dx = tuple(
            self.weights[dim] * (tgt_centers[dim] - anc_centers[dim]) / anc_lengths[dim]
            for dim in range(self.n_dim)
        )
        targets_dw = tuple(
            self.weights[dim + self.n_dim] * torch.log(tgt_lengths[dim] / anc_lengths[dim])
            for dim in range(self.n_dim)
        )

        return torch.stack(targets_dx + targets_dw, dim=1)

    def decode(self, rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes.

        :param rel_codes: encoded zyxdhw boxes.
        :param anchors: reference zyxzyx boxes.
        """
        assert rel_codes.shape[1] % (self.n_dim * 2) == 0
        assert anchors.shape[1] == self.n_dim * 2

        boxes = anchors.to(rel_codes.dtype)

        lengths, centers = self._get_lengths_centers(boxes)

        # dim::(2 * self.n_dim) is crucial to also support decoding of class-wise rel_codes from a box head
        dx = [rel_codes[:, dim::(2 * self.n_dim)] / self.weights[dim] for dim in range(self.n_dim)]
        dw = [rel_codes[:, dim::(2 * self.n_dim)] / self.weights[dim] for dim in range(self.n_dim, 2 * self.n_dim)]

        # Prevent sending too large values into torch.exp()
        dw = [torch.clamp(dw[dim], max=self.bbox_xform_clip) for dim in range(self.n_dim)]

        # Because of the dim::(2 * self.n_dim) above, the [:, None] has to be here
        pred_centers = [dx[dim] * lengths[dim][:, None] + centers[dim][:, None] for dim in range(self.n_dim)]
        pred_lengths = [torch.exp(dw[dim]) * lengths[dim][:, None] for dim in range(self.n_dim)]

        pred_boxes = torch.zeros_like(rel_codes)

        for dim in range(self.n_dim):
            # Because of the dim::(2 * self.n_dim) above, it is again required here
            pred_boxes[:, dim::(2 * self.n_dim)] = pred_centers[dim] - 0.5 * pred_lengths[dim]
            # Note: "- 1" is correct; don't be fooled by the asymmetry
            pred_boxes[:, dim + self.n_dim::(2 * self.n_dim)] = pred_centers[dim] + 0.5 * pred_lengths[dim] - 1

        return pred_boxes

    def _get_lengths_centers(self, boxes: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        lengths = tuple(boxes[:, self.n_dim + dim] - boxes[:, dim] + 1 for dim in range(self.n_dim))
        centers = tuple(boxes[:, dim] + 0.5 * lengths[dim] for dim in range(self.n_dim))
        return lengths, centers
