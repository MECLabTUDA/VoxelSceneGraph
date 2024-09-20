from __future__ import annotations

from typing import Hashable

import torch
from scene_graph_api.tensor_structures import BoxListOps as _BoxListOps
from scene_graph_api.tensor_structures.box_list_ops import BoxListBase

from scene_graph_prediction.layers import nms, nms_3d

_SIZE_T = tuple[int, ...]


class BoxListOps(_BoxListOps):
    """Some additional operations that are specific to Scene Graph prediction."""

    @staticmethod
    def nms(
            boxlist: BoxListBase,
            nms_iou_thresh: float,
            max_proposals: int = -1,
            score_field: Hashable = None
    ) -> tuple[BoxListBase, torch.LongTensor]:
        """
        Perform non-maximum suppression (based on IoU), with scores specified in the field score_field.
        Note: boxes are sorted by descending score.
        :param boxlist: BoxList.
        :param nms_iou_thresh: threshold for box overlap
        :param max_proposals: if > 0, then only the top max_proposals are kept
        :param score_field
        :return top boxes, keep (if thresh > 0 else empty tensor) as idx tensor
        """
        if score_field is None:
            score_field = boxlist.PredictionField.PRED_SCORES

        if nms_iou_thresh <= 0:
            # We only need to sort by descending score
            _, sort_ind = boxlist.get_field(score_field).sort(descending=True)
            return boxlist[sort_ind], torch.empty()
        mode = boxlist.mode
        boxlist = boxlist.convert(boxlist.Mode.zyxzyx)

        scores = boxlist.get_field(score_field)
        if boxlist.n_dim == 2:
            keep = nms(boxlist.boxes, scores, nms_iou_thresh)
        elif boxlist.n_dim == 3:
            keep = nms_3d(boxlist.boxes, scores, nms_iou_thresh).to(device=boxlist.boxes.device)
        else:
            raise RuntimeError(f"Nms is not implemented for {boxlist.n_dim}D.")

        # Sort kept boxes by score
        _, sort_ind = scores[keep].sort(descending=True)
        sorted_keep = keep[sort_ind]

        # Keep up to max proposals
        if max_proposals > 0:
            sorted_keep = sorted_keep[:max_proposals]

        return boxlist[sorted_keep].convert(mode), sorted_keep

    @staticmethod
    def nms_classwise(
            boxlist: BoxListBase,
            nms_iou_thresh: float,
            max_proposals: int = -1,
            score_field: Hashable = None,
            label_field: Hashable = None
    ) -> tuple[BoxListBase, torch.LongTensor]:
        """
        Perform class-wise non-maximum suppression (based on IoU), with scores specified in the field score_field
        and labels specified in the label_field.
        Note: boxes are sorted by descending score.
        Note: works using a single NMS call by offsetting boxes:
              - based on their label
              - with an offset larger than any box
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param nms_iou_thresh: threshold for box overlap
        :param max_proposals: if > 0, then only the top max_proposals are kept
        :param score_field: field with score for NMS.
        :param label_field: field used to distinguish boxes with different classes for NMS.
        :return top boxes, keep (if thresh > 0 else empty tensor) as idx tensor
        """
        if score_field is None:
            score_field = boxlist.PredictionField.PRED_SCORES
        if label_field is None:
            score_field = boxlist.PredictionField.PRED_LABELS

        if nms_iou_thresh <= 0:
            # We only need to sort by descending score
            _, sort_ind = boxlist.get_field(score_field).sort(descending=True)
            return boxlist[sort_ind], torch.empty()

        if len(boxlist) == 0:
            return boxlist, torch.empty(0, dtype=torch.int64, device=boxlist.boxes.device)

        # Find out the max size for dim 0
        labels = boxlist.get_field(label_field, raise_missing=True)
        max_size = torch.max(boxlist.boxes[:, boxlist.n_dim] - boxlist.boxes[:, 0]) + 1
        # Offset boxes
        offset = labels * max_size
        boxlist.boxes[:, 0] += offset
        if boxlist.mode == boxlist.Mode.zyxzyx:
            # No need if zyxdhw encoded
            boxlist.boxes[:, boxlist.n_dim] += offset

        # Perform NMS
        _, keep = BoxListOps.nms(boxlist, nms_iou_thresh, max_proposals=max_proposals, score_field=score_field)

        # Since we have shallow copies, we need to remove the offset again...
        boxlist.boxes[:, 0] -= offset
        if boxlist.mode == boxlist.Mode.zyxzyx:
            # No need if zyxdhw encoded
            boxlist.boxes[:, boxlist.n_dim] -= offset

        return boxlist[keep], keep

    @staticmethod
    def _one_to_one_iou(
            boxlist: BoxListBase, other_boxlist: BoxListBase
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the IoU (one to one) of two set of boxes of the SAME length.
        Note: only used for a loss computation, where there is a one-to-one matching,
              unlike where the IoU matrix is used to compute matches.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :returns iou, intersection volume, union volume
        """
        assert len(boxlist) == len(other_boxlist) and boxlist.size == other_boxlist.size

        boxlist = boxlist.convert(boxlist.Mode.zyxzyx)
        other_boxlist = other_boxlist.convert(other_boxlist.Mode.zyxzyx)

        inter_vol = BoxListOps.volume(BoxListOps.intersection(boxlist, other_boxlist, filter_empty=False))
        union_vol = BoxListOps.volume(boxlist) + BoxListOps.volume(other_boxlist) - inter_vol

        return inter_vol / (union_vol + BoxListOps.EPS), inter_vol, union_vol

    @staticmethod
    def generalized_iou(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """
        Compute the Generalized IoU (one to one) of two set of boxes of the SAME length.
        https://giou.stanford.edu/
        Note: only used for a loss computation, where there is a one-to-one matching,
              unlike where the IoU matrix is used to compute matches.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :return: generalized iou sized [N]
        """
        boxlist = boxlist.convert(boxlist.Mode.zyxzyx)
        other_boxlist = other_boxlist.convert(other_boxlist.Mode.zyxzyx)

        iou, _, union_vol = BoxListOps._one_to_one_iou(boxlist, other_boxlist)
        full_union_vol = BoxListOps.volume(BoxListOps.union(boxlist, other_boxlist))

        return iou - (full_union_vol - union_vol) / (full_union_vol + BoxListOps.EPS)

    @staticmethod
    def _distance_iou_penalty(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """Penalty term for the DIoU loss."""
        # Corner distance
        full_union = BoxListOps.union(boxlist, other_boxlist).convert(boxlist.Mode.zyxdhw)
        corner_dist = (full_union.boxes[:, boxlist.n_dim:] ** 2).sum(-1)
        # Center distance
        boxlist_center = (boxlist.boxes[:, boxlist.n_dim:] + boxlist.boxes[:, :boxlist.n_dim] + 1) / 2
        other_boxlist_center = (other_boxlist.boxes[:, boxlist.n_dim:] + other_boxlist.boxes[:, :boxlist.n_dim] + 1) / 2
        center_dist = ((boxlist_center - other_boxlist_center) ** 2).sum(-1)

        return center_dist / corner_dist

    @staticmethod
    def distance_iou(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """
        Compute the Distance IoU (one to one) of two set of boxes of the SAME length.
        https://learnopencv.com/iou-loss-functions-object-detection/
        Note: only used for a loss computation, where there is a one-to-one matching,
              unlike where the IoU matrix is used to compute matches.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :return: distance iou sized [N]
        """
        iou, _, _ = BoxListOps._one_to_one_iou(boxlist, other_boxlist)

        return iou - BoxListOps._distance_iou_penalty(boxlist, other_boxlist)

    @staticmethod
    def complete_iou(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """
        Compute the Complete IoU (one to one) of two set of boxes of the SAME length.
        https://learnopencv.com/iou-loss-functions-object-detection/
        Note: only used for a loss computation, where there is a one-to-one matching,
              unlike where the IoU matrix is used to compute matches.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :return: complete iou sized [N]
        """
        assert boxlist.n_dim >= 2
        iou, _, _ = BoxListOps._one_to_one_iou(boxlist, other_boxlist)
        distance_pen = BoxListOps._distance_iou_penalty(boxlist, other_boxlist)

        # Aspect ratio
        h, w = boxlist.convert(boxlist.Mode.zyxdhw).boxes[:, -2:].unbind(dim=-1)
        other_h, other_w = other_boxlist.convert(other_boxlist.Mode.zyxdhw).boxes[:, -2:].unbind(dim=-1)

        arctan = (torch.arctan(other_w / (other_h + BoxListOps.EPS)) - torch.arctan(w / (h + BoxListOps.EPS))) ** 2

        v = 4 / (torch.pi ** 2) * arctan
        with torch.no_grad():
            alpha = v / (1 - iou + v + BoxListOps.EPS)
        ratio_pen = alpha * v

        return iou - distance_pen - ratio_pen

    @staticmethod
    def voxel_complete_iou(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """
        Compute the Complete IoU (one to one) of two set of boxes of the SAME length.
        https://learnopencv.com/iou-loss-functions-object-detection/
        Note: only used for a loss computation, where there is a one-to-one matching,
              unlike where the IoU matrix is used to compute matches.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :return: complete iou sized [N]
        """
        assert boxlist.n_dim >= 3
        iou, _, _ = BoxListOps._one_to_one_iou(boxlist, other_boxlist)
        distance_pen = BoxListOps._distance_iou_penalty(boxlist, other_boxlist)

        # Aspect ratio
        d, h, w = boxlist.convert(boxlist.Mode.zyxdhw).boxes[:, -3:].unbind(dim=-1)
        other_d, other_h, other_w = other_boxlist.convert(other_boxlist.Mode.zyxdhw).boxes[:, -3:].unbind(dim=-1)

        arctan1 = (torch.arctan(other_w / (other_h + BoxListOps.EPS)) - torch.arctan(w / (h + BoxListOps.EPS))) ** 2
        arctan2 = (torch.arctan(other_h / (other_d + BoxListOps.EPS)) - torch.arctan(h / (d + BoxListOps.EPS))) ** 2
        arctan3 = (torch.arctan(other_d / (other_w + BoxListOps.EPS)) - torch.arctan(d / (w + BoxListOps.EPS))) ** 2

        v = 4 / (torch.pi ** 2) * (arctan1 + arctan2 + arctan3)
        with torch.no_grad():
            alpha = v / (1 - iou + v + BoxListOps.EPS)
        ratio_pen = alpha * v

        return iou - distance_pen - ratio_pen
