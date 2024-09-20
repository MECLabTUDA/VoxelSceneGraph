# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict, deque

import numpy as np
import torch

from scene_graph_prediction.utils.config import AccessTrackingCfgNode

try:
    # Optional aim-compatibility for tracking results
    import aim
except ImportError:
    aim = None


class _SmoothedAvg:
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size: int = 20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self) -> float:
        # noinspection PyTypeChecker
        return np.median(self.deque)

    @property
    def avg(self) -> float:
        # noinspection PyTypeChecker
        return np.mean(self.deque)

    @property
    def global_avg(self) -> float:
        return self.total / self.count


class MetricLogger:
    """Metric tracking with aim compatibility (https://github.com/aimhubio/aim)."""

    def __init__(self, cfg: AccessTrackingCfgNode, separator: str = "\t"):
        self.metrics = defaultdict(_SmoothedAvg)
        self.separator = separator

        if aim is not None and cfg.AIM_TRACKING:
            self.run = aim.Run()
            self.run["hparams"] = cfg.convert_to_dict()
        else:
            self.run = None

    def update(self, iteration: int, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.metrics[k].update(v)
            if self.run is not None:
                self.run.track(v, name=k, step=iteration)

    def __getattr__(self, attr):
        if attr in self.metrics:
            return self.metrics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.metrics.items():
            loss_str.append(f"{name}: {meter.median:.4f} ({meter.global_avg:.4f})")
        return self.separator.join(loss_str)
