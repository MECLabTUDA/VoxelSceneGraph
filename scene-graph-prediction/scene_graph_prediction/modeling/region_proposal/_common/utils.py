# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Utility functions manipulating the prediction layers."""

import torch

from ...utils import cat


def permute_and_flatten(tensor: torch.Tensor, n: int, a: int, c: int, *zyx_lengths: int) -> torch.Tensor:
    """
    Returns the tensor as flattened to B x total anchors x C.
    Note: supports ND.
    :param tensor: a Tensor
    :param n: batch size
    :param a: number of boxes per location
    :param c: classes for objectness; 2*n_dim for regressions
    :param zyx_lengths: (height, width) in 2D; (depth, height, width) in 3D...
    :return: Tensor permuted (0, 3, 4, 1, 2) and reshaped to (n, -1, c)
    """
    n_dim = len(zyx_lengths)
    n_a_c_lengths = [n, a, c] + list(zyx_lengths)
    permuted = [0] + list(range(3, 3 + n_dim)) + [1, 2]  # n, *l, a, c
    flattened = [n, -1, c]
    # return tensor.view(n, -1, c, h, w).permute(0, 3, 4, 1, 2).reshape(n, -1, c)
    # Note: the first view has to be done in case the channel dim is missing e.g. with objectness tensors
    return tensor.view(*n_a_c_lengths).permute(*permuted).reshape(*flattened)


def concat_box_prediction_layers(
        box_cls: list[torch.Tensor],
        box_regression: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature level, permute the outputs to make them be in the same format as the labels.
    Note that the labels are computed for all feature levels concatenated, so we keep the same representation
    for the objectness and the box_regression
    """
    assert box_cls
    assert box_regression
    n_dim = box_cls[0].dim() - 2

    box_cls_flattened = []
    box_regression_flattened = []
    # Iterate over feature levels (each tensor per level, contains predictions for a batch of images)
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        n, axc, *lengths = box_cls_per_level.shape
        a = box_regression_per_level.shape[1] // (2 * n_dim)
        c = axc // a
        box_cls_per_level = permute_and_flatten(box_cls_per_level, n, a, c, *lengths)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, n, a, 2 * n_dim, *lengths)
        box_regression_flattened.append(box_regression_per_level)

    # Concatenate on the first dimension (representing the feature levels), to take into account
    # the way the labels were generated (with all feature maps being concatenated as well)
    # noinspection PyUnboundLocalVariable
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, c)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 2 * n_dim)
    return box_cls, box_regression
