# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

import torch
from yacs.config import CfgNode

from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau


def build_optimizer(
        cfg: CfgNode,
        model: torch.nn.Module,
        logger: logging.Logger,
        slow_heads: list[str] | None = None,
        slow_ratio: float = 10.0,
        rl_factor: float = 1.0
) -> torch.optim.Optimizer:
    params = []

    # noinspection PyUnresolvedReferences
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if slow_heads is not None:
            for item in slow_heads:
                if item in key:
                    logger.info(f"SLOW HEADS: {key} is slowed down by ratio of {slow_ratio}.")
                    lr /= slow_ratio
                    break

        params.append({"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay})

    return torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)


# noinspection PyProtectedMember
def build_lr_scheduler(
        cfg: CfgNode,
        optimizer: torch.optim.Optimizer,
        logger: logging.Logger | None = None
) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )

    else:
        raise ValueError("Invalid Schedule Type")
