from abc import ABC, abstractmethod
from typing import Callable

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, BoxListOps


class BoxRegressionLoss(torch.nn.Module, ABC):
    _registered_detectors: dict[str, Callable[[CfgNode], "BoxRegressionLoss"]] = {}

    def __init__(self, require_box_coding: bool):
        super().__init__()
        self.require_box_coding = require_box_coding

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers any subclass that has been initialized."""
        super().__init_subclass__(**kwargs)
        # noinspection PyTypeChecker
        BoxRegressionLoss._registered_detectors[cls.__name__] = cls

    @abstractmethod
    def forward(self, pred: torch.FloatTensor | BoxList, target: torch.FloatTensor | BoxList) -> torch.FloatTensor:
        """Reduction should be "none", but only float value per box."""
        raise NotImplementedError

    @staticmethod
    def build(cfg: CfgNode) -> "BoxRegressionLoss":
        """Build the detector using the config. Raises a KeyError if the specified detector is not registered."""
        return BoxRegressionLoss._registered_detectors[cfg.MODEL.ROI_BOX_HEAD.REGRESSION_LOSS](cfg)


class L1Loss(BoxRegressionLoss):
    def __init__(self, cfg: CfgNode):
        super().__init__(True)
        self.bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA

    def forward(self, pred: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        # noinspection PyTypeChecker
        return torch.mean(
            torch.nn.functional.smooth_l1_loss(
                pred,
                target,
                beta=self.bbox_reg_beta,
                reduction="none"
            ),
            dim=1
        )


class GIoULoss(BoxRegressionLoss):
    def __init__(self, _: CfgNode):
        super().__init__(False)

    def forward(self, pred: BoxList, target: BoxList) -> torch.FloatTensor:
        return 1 - BoxListOps.generalized_iou(pred, target)


class DIoULoss(BoxRegressionLoss):
    def __init__(self, _: CfgNode):
        super().__init__(False)

    def forward(self, pred: BoxList, target: BoxList) -> torch.FloatTensor:
        return 1 - BoxListOps.distance_iou(pred, target)


class CIoULoss(BoxRegressionLoss):
    def __init__(self, cfg: CfgNode):
        assert cfg.INPUT.N_DIM in [2, 3]
        super().__init__(False)

    def forward(self, pred: BoxList, target: BoxList) -> torch.FloatTensor:
        if pred.n_dim == 3:
            return 1 - BoxListOps.voxel_complete_iou(pred, target)
        return 1 - BoxListOps.complete_iou(pred, target)
