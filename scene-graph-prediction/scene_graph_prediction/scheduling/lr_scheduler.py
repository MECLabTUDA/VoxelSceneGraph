# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Literal

import torch


# noinspection PyProtectedMember
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Note: Ideally this would be achieved with a CombinedLRScheduler, separating MultiStepLR with WarmupLR,
          but the current LRScheduler design doesn't allow it
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: list[int],
            gamma: float = 0.1,
            warmup_factor: float = 1.0 / 3,
            warmup_iters: int = 500,
            warmup_method: Literal["constant", "linear"] = "linear",
            last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(f"Milestones should be a list of increasing integers. Got {milestones}")
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Only 'constant' or 'linear' warmup_method accepted got {warmup_method}")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    # noinspection DuplicatedCode
    def get_lr(self) -> list[float]:
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise RuntimeError

        factor = warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


class MetricsAwareScheduler(torch.optim.lr_scheduler._LRScheduler, ABC):
    @abstractmethod
    def step(self, metrics: float | None, epoch: int | None = None):
        """
        Special step method that allows to adjust the lr based on a value summarizing the target metrics.
        Currently, we use the negative mean validation loss.
        :param epoch: the current epoch
        :param metrics: a metrics aggregation (float) or None. If None, no update will be performed.
        """
        raise NotImplementedError


class WarmupReduceLROnPlateau(MetricsAwareScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            gamma: float = 0.5,
            warmup_factor: float = 1.0 / 3,
            warmup_iters: int = 500,
            warmup_method: Literal["constant", "linear"] = "linear",
            last_epoch: int = -1,
            patience: int = 2,
            threshold: float = 1e-4,
            cooldown: int = 1,
            logger: logging.Logger | None = None,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Only 'constant' or 'linear' warmup_method accepted got {warmup_method}")
        super().__init__(optimizer, last_epoch)

        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.logger = logger

        self.step(last_epoch)

    def state_dict(self) -> dict:
        """
        Returns the state of the scheduler.
        It contains an entry for every variable in self.__dict__ which is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state."""
        self.__dict__.update(state_dict)

    # noinspection DuplicatedCode
    def get_lr(self) -> list[float]:
        warmup_factor = 1
        # during warming up
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise RuntimeError

        return [base_lr * warmup_factor * self.gamma ** self.stage_count for base_lr in self.base_lrs]

    def step(self, epoch: int, metrics: float | None = None):
        # The following part is modified from ReduceLROnPlateau
        if metrics is None:
            # Validation did not occur yet
            pass
        else:
            if float(metrics) > (self.best + self.threshold):
                self.best = float(metrics)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs >= self.patience:
                if self.logger is not None:
                    self.logger.info(f"Trigger Schedule Decay, RL has been reduced by factor {self.gamma}")
                self.stage_count += 1  # this will automatically decay the learning rate
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
