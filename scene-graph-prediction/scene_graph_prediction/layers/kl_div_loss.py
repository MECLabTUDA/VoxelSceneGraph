# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Literal

import torch


def kl_div_loss(
        input_tensor: torch.Tensor,
                target: torch.Tensor,
                eps: float = 1e-9,
        reduction: Literal["mean", "sum"] = "sum"
) -> torch.Tensor:
    assert len(input_tensor.shape) == 2
    assert len(target.shape) == 2

    # noinspection PyUnresolvedReferences
    log_target = (target + eps).log()
    # noinspection PyUnresolvedReferences
    log_input = (input_tensor + eps).log()

    loss = target.detach() * (log_target.detach() - log_input)

    if reduction == "sum":
        loss = loss.sum(-1)
    elif reduction == "mean":
        loss = loss.mean(-1)
    else:
        raise RuntimeError(f"Unknown reduction {reduction}")

    return loss.mean()
