import logging
from typing import Iterable

import torch


def clip_grad_norm(
        named_parameters: Iterable[tuple[str, torch.autograd.Variable]],
        max_norm: float,
        logger: logging.Logger,
        clip: bool = False,
        verbose: bool = False
) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    :returns: Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm **= 1. / 2
    clip_coefficient = max_norm / (total_norm + 1e-6)
    if clip_coefficient < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coefficient)

    if verbose:
        logger.info(f'---Total norm {total_norm:.5f} clip coefficient {clip_coefficient:.5f}-----------------')
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            logger.info(f"{name:<50s}: {norm:.5f}, ({param_to_shape[name]})")
        logger.info(f'-------------------------------')

    return total_norm
