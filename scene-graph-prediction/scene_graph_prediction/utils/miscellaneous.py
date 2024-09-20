# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os

import numpy as np
import torch

from scene_graph_prediction.structures import BoxList, BoxListOps
from .comm import is_main_process
from .config import AccessTrackingCfgNode


def mkdir(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise OSError from e


def save_config(cfg: AccessTrackingCfgNode, path: str):
    """Save the config. Will merge with existing config if any."""
    if is_main_process():
        if not os.path.exists(path):
            # There is no config yet, so just write relevant stuff
            with open(path, 'w') as f:
                f.write(cfg.dump_accessed_only())
            return
        # Else load existing config and add relevant stuff and finally save all
        with open(path, "r") as f:
            existing_cfg = AccessTrackingCfgNode.load_cfg(f)
        existing_cfg.set_new_allowed(True)
        existing_cfg.merge_from_other_cfg(cfg.clone_only_accessed())
        with open(path, 'w') as f:
            f.write(existing_cfg.dump())


def intersect_2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Given two arrays [m1, n], [m2, n], returns a [m1, m2] array where each entry is True if those rows match.
    Note: mainly used to compare two sets of relation triplets.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError(f"Input arrays must have same #columns ({x1.shape[1]} vs {x2.shape[1]}).")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # noinspection PyUnresolvedReferences
    return (x1[..., None] == x2.T[None, ...]).all(1)


def argsort_desc(scores: np.ndarray) -> np.ndarray:
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def bboxes_iou_from_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Note: supports ND.
    :param boxes1: (m, 2 * n_dim) np.ndarray of zyxzyx bounding boxes
    :param boxes2: (n, 2 * n_dim) np.ndarray of zyxzyx bounding boxes
    :return: iou (m, n)
    """
    assert np.prod(boxes1.shape) > 0
    n_dim = boxes1.shape[1] // 2
    boxes1 = BoxList(boxes1, (0,) * n_dim, BoxList.Mode.zyxzyx)
    boxes2 = BoxList(boxes2, (0,) * n_dim, BoxList.Mode.zyxzyx)
    return BoxListOps.iou(boxes1, boxes2).numpy()


def torch_bbox_2d(arr: torch.BoolTensor) -> torch.LongTensor:
    """
    Given a binary 2D mask, compute the bounding box of the non-zeo values.
    :param arr: Binary 2D mask.
    :return: 2D bounding box of the non-zeo values.
    """
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = torch.any(arr, dim=1)
    cols = torch.any(arr, dim=0)
    r_min, r_max = torch.where(rows)[0][[0, -1]]
    c_min, c_max = torch.where(cols)[0][[0, -1]]

    return torch.tensor([r_min, c_min, r_max, c_max]).to(dtype=torch.int64, device=arr.device)


def torch_bbox_3d(arr: torch.BoolTensor) -> torch.LongTensor:
    """
    Given a binary 3D mask, compute the bounding box of the non-zeo values.
    :param arr: Binary 3D mask.
    :return: 3D bounding box of the non-zeo values.
    """
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    r = arr.any(2).any(1)
    c = arr.any(2).any(0)
    z = arr.any(1).any(0)

    r_min, r_max = torch.where(r)[0][[0, -1]]
    c_min, c_max = torch.where(c)[0][[0, -1]]
    z_min, z_max = torch.where(z)[0][[0, -1]]

    # Cast np.int32 to int to avoid JSON serialization issues
    return torch.tensor([r_min, c_min, z_min, r_max, c_max, z_max]).to(dtype=torch.int64, device=arr.device)


def segmentation_to_binary_masks(
        boxes: torch.Tensor,
        labels: torch.LongTensor,
        segmentation: torch.LongTensor
) -> torch.LongTensor:
    """
    Given a batch of bounding boxes (zyxzyx) and their corresponding labels,
    return the corresponding binary masks as Nx1xDxHxW based on the semantic segmentation (DxHxW).
    Note: works with ND.
    """
    n_dim = boxes.shape[1] // 2
    masks = torch.zeros((labels.shape[0],) + segmentation.shape, dtype=torch.long, device=segmentation.device)
    boxes = boxes.detach()
    # Round boxes (expanding)
    boxes[:n_dim] = boxes[:n_dim].floor()
    boxes[n_dim:] = boxes[n_dim:].ceil()
    boxes = boxes.int()
    segmentation = segmentation.detach()

    # Iterate over boxes
    for box_idx in range(boxes.shape[0]):
        box = boxes[box_idx]
        # Create slicer for bbox area
        slicer = tuple(slice(0, box[dim + n_dim].item()) for dim in range(n_dim))
        # Copy binary mask over bbox area
        masks[box_idx][slicer] = segmentation[slicer] == labels[box_idx]

    return masks[:, None]  # Add channel dim


def get_pred_masks(prediction: BoxList) -> torch.Tensor:
    """
    Given a BoxList with predictions, returns the binary masks for each bbox.
    These may need to be computed from a predicted segmentation.
    :return: binary masks for each bbox as Nx1xDxHxW.
    """
    if prediction.has_field(BoxList.PredictionField.PRED_MASKS):
        # We just need to slice the masks so that they have the same shape as the unpadded segmentation
        # Note: we expect Bx1xDxHxW predictions.
        slicer = slice(None), slice(None), *tuple(slice(0, s) for s in prediction.size)
        return prediction.PRED_MASKS[slicer]

    # We just need to slice the segmentation so that it has the same shape as the unpadded segmentation
    slicer = tuple(slice(0, s) for s in prediction.size)
    return segmentation_to_binary_masks(prediction.boxes, prediction.PRED_LABELS, prediction.PRED_SEGMENTATION[slicer])


def get_gt_masks(boxes: BoxList) -> torch.Tensor:
    """
    Given a BoxList with predictions, returns the binary masks for each bbox.
    These may need to be computed from a predicted segmentation.
    :return: binary masks for each bbox as Nx1xDxHxW.
    """
    if boxes.has_field(BoxList.AnnotationField.MASKS):
        return boxes.MASKS
    return segmentation_to_binary_masks(boxes.boxes, boxes.LABELS, boxes.SEGMENTATION)
