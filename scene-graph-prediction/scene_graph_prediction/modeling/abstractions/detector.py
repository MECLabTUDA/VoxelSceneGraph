# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Implements the Generalized R-CNN framework."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import ImageList, BoxList
from .backbone import Backbone
from .loss import LossDict
from .region_proposal import RPN
from .roi_heads import CombinedROIHeads


class AbstractDetector(torch.nn.Module, ABC):
    """
    Main class for Generalized R-CNN. Currently, supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    _registered_detectors: dict[str, Callable[[CfgNode], AbstractDetector]] = {}

    def __init__(
            self,
            cfg: CfgNode,
            backbone: Backbone,
            rpn: RPN,
            roi_heads: CombinedROIHeads
    ):
        """
        Common constructor used building each detector type.
        Note: if we ever need individual constructor prototypes, we'll have to replace the dict with a registry.
        """
        super().__init__()
        self.n_dim = cfg.INPUT.N_DIM
        self.cfg = cfg
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers any subclass that has been initialized."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            # noinspection PyTypeChecker
            AbstractDetector._registered_detectors[cls.__name__] = cls

    @abstractmethod
    def forward(
            self,
            images: ImageList | list[torch.Tensor],
            targets: list[BoxList] | None = None,
            loss_during_testing: bool = False
    ) -> tuple[list[BoxList], LossDict]:
        """
        :param images: images to be processed
        :param targets: ground-truth boxes present in the image (optional)
        :param loss_during_testing: whether to compute the loss for relevant modules even when evaluating.

        :returns: The output from the model.
                  During training, it returns a dict[Tensor] which contains the losses.
                  During testing, it returns list[BoxList] contains additional fields
                  like `pred_scores`, `pred_labels` and `pred_masks` (for Mask R-CNN models).
        """
        raise NotImplementedError

    @staticmethod
    def get_registered_detector(name: str) -> Callable[[CfgNode], AbstractDetector]:
        """Return the detector type registered with the provided name or raises a KeyError."""
        return AbstractDetector._registered_detectors[name]

    @staticmethod
    def build(cfg: CfgNode) -> AbstractDetector:
        """Build the detector using the config. Raises a KeyError if the specified detector is not registered."""
        return AbstractDetector.get_registered_detector(cfg.MODEL.META_ARCHITECTURE)(cfg)
