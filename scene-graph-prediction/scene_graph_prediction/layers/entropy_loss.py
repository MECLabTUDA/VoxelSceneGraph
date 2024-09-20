# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Literal

import torch


def entropy_loss(input_tensor: torch.Tensor,
                 eps: float = 1e-9,
                 reduction: Literal["mean", "sum"] = 'sum') -> torch.Tensor:
    # noinspection PyTypeChecker
    loss = -(input_tensor * torch.log(input_tensor + eps))

    if reduction == 'sum':
        loss = loss.sum(-1)
    elif reduction == 'mean':
        loss = loss.mean(-1)
    else:
        raise RuntimeError(f"Unknown reduction {reduction}")

    return loss.mean()
