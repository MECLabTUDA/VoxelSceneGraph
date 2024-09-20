import itertools
from functools import reduce

import numpy as np
import torch

from scene_graph_prediction.structures import BoxList


def get_box_info(
        boxes: torch.Tensor,
        need_norm: bool = True,
        proposal: BoxList | None = None
) -> torch.Tensor:
    """
    Note: may be partially redundant with .motifs.encode_box_info
    :param boxes: [batch_size, "zyxzyx"]
    :param need_norm: whether to rescale based on the image size (contained in proposal)
    :param proposal:
    :returns: [batch_size, (y1,x1,y2,x2,cy,cx,h,w)] (in 2D)
    """
    n_dim = boxes.shape[1] // 2
    lengths = boxes[:, n_dim:] - boxes[:, :n_dim] + 1.0  # w, h
    centers_lengths = torch.cat((boxes[:, :n_dim] + 0.5 * lengths, lengths), 1)  # cx, ccy, w, h
    box_info = torch.cat((boxes, centers_lengths), 1)  # x1, y1, x2, y2, cx, cy, w, h
    if need_norm:
        assert proposal
        box_info /= float(max(*proposal.size))
    return box_info


def get_box_pair_info(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    :param box1: [batch_size, (y1,x1,y2,x2,cy,cx,h,w)] (in 2D, see get_box_info)
    :param box2: [batch_size, (y1,x1,y2,x2,cy,cx,h,w)] (in 2D, see get_box_info)
    :returns: [*box1, *box2, *union_box, *intersection_box] (shape N * 32)
    """
    assert box1.shape == box2.shape
    n_dim = box1.shape[1] // 4
    # Union box
    union_box = torch.empty((box1.shape[0], 2 * n_dim))
    for dim in range(n_dim):
        union_box[:, dim] = torch.min(box1[:, dim], box2[:, dim])
        union_box[:, n_dim + dim] = torch.max(box1[:, n_dim + dim], box2[:, n_dim + dim])
    union_info = get_box_info(union_box, need_norm=False)

    # Intersection box
    intersection_box = torch.empty((box1.shape[0], 2 * n_dim))
    for dim in range(n_dim):
        union_box[:, dim] = torch.max(box1[:, dim], box2[:, dim])
        union_box[:, n_dim + dim] = torch.min(box1[:, n_dim + dim], box2[:, n_dim + dim])
    intersection_info = get_box_info(intersection_box, need_norm=False)

    # noinspection PyTypeChecker
    cases = [
        torch.nonzero(
            intersection_box[:, n_dim + dim].contiguous().view(-1) < intersection_box[:, dim].contiguous().view(-1)
        ).view(-1)
        for dim in range(n_dim)
    ]

    for dim in range(n_dim):
        if cases[dim].numel() > 0:
            intersection_info[cases[dim]] = 0

    return torch.cat((box1, box2, union_info, intersection_info), 1)


def classwise_boxes_iou(boxes: torch.Tensor, n_dim: int) -> torch.Tensor:
    """Compute IoU for NxCLSx2N_DIM tensor."""
    assert boxes.dim() == 3  # Needs to have batch+dim classes
    n = boxes.size(0)
    c = boxes.size(1)

    max_xy = torch.min(boxes[:, None, :, n_dim:].expand(n, n, c, n_dim),
                       boxes[None, :, :, n_dim:].expand(n, n, c, n_dim))
    min_xy = torch.max(boxes[:, None, :, :n_dim].expand(n, n, c, n_dim),
                       boxes[None, :, :, :n_dim].expand(n, n, c, n_dim))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, c
    inters = reduce(lambda a, b: a * b, [inter[..., dim] for dim in range(n_dim)])
    boxes_flat = boxes.view(-1, 2 * n_dim)
    areas_flat = reduce(lambda a, b: a * b, [boxes_flat[:, n_dim + dim] - boxes_flat[:, dim] + 1.
                                             for dim in range(n_dim)])

    areas = areas_flat.view(boxes.size(0), boxes.size(1))  # Areas per box and per channel
    union = -inters + areas[None] + areas[:, None]
    return inters / union


def layer_init(layer: torch.nn.Module, init_para: float = 0.1, normal: bool = False):
    """Init layer with normal distribution (or Xavier if not normal)."""
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
    else:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
    torch.nn.init.constant_(layer.bias, 0)


def obj_prediction_nms(
        boxes_per_cls: torch.LongTensor,
        pred_logits: torch.FloatTensor,
        nms_thresh: float = 0.3
) -> torch.LongTensor:
    """
    :param boxes_per_cls: [num_obj, num_cls, 2 * n_dim]
    :param pred_logits: [num_obj, num_category]
    :param nms_thresh:
    """
    n_dim = boxes_per_cls.shape[-1] // 2
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    # n, n, c
    is_overlap = classwise_boxes_iou(
        boxes_per_cls.view(boxes_per_cls.shape[0], -1, 2 * n_dim), n_dim
    ).cpu().numpy() >= nms_thresh

    prob_sampled = torch.nn.functional.softmax(pred_logits, 1).cpu().numpy()
    prob_sampled[:, 0] = 0  # Set bg to 0

    # noinspection PyTypeChecker
    pred_label: torch.LongTensor = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) <= 0:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0  # This way we won't re-sample

    return pred_label


def block_orthogonal(tensor: torch.Tensor, split_sizes: list[int], gain: float = 1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError(
            "Tensor dimensions must be divisible by their respective split_sizes. "
            f"Found size: {sizes} and split_sizes: {split_sizes}"
        )
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]

    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e, split_size[i] for dimension i).

        # This is a tuple of slices corresponding to: tensor[index: index + step_size, ...].
        # This is required because we could have an arbitrary number of dimensions.
        # The actual slices we need are the start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([
            slice(start_index, start_index + size)
            for start_index, size in zip(block_start_indices, split_sizes)
        ])

        # Let's not initialize empty things to 0s because THAT SOUNDS AWFUL
        n_dim = len(block_slice)
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(*([max(sizes)] * n_dim))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)

        tensor_slice = tuple(slice(0, sizes[dim]) for dim in range(n_dim))
        tensor[block_slice] = tensor_copy[tensor_slice]
