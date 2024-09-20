from __future__ import annotations

from enum import Enum, IntFlag, auto
from typing import Type

import torch
from pycocotools3d import IouType as CocoIouType
from pycocotools3d.params import EvaluationParams, DefaultParams, Params3D
from typing_extensions import Self
from yacs.config import CfgNode

from scene_graph_prediction.utils.registry import Registry

# Evaluation parameters registry for COCO
COCO_EVALUATION_PARAMETERS: Registry[str, Type[EvaluationParams]] = Registry(
    default=DefaultParams,
    default_3d=Params3D,
)


class EvaluationType(IntFlag):
    """
    Enum describing which parts of the prediction should be evaluated,
    i.e. COCO-style object detection, semantic segmentation (if masks/seg are predicted) or SGG.
    """
    COCO = auto()
    SemanticSegmentation = auto()
    SGG = auto()


class IouType(Enum):
    """Extends IouTypes from pycocotools3d for other applications/metrics."""
    BoundingBox = CocoIouType.BoundingBox.value  # mAP
    Keypoints = CocoIouType.Keypoints.value
    Segmentation = CocoIouType.Segmentation.value
    Attributes = "attributes"
    Relations = "relations"
    RegionProposal = "box_proposal"  # mAR

    def to_coco(self) -> CocoIouType:
        match self:
            case self.BoundingBox | self.RegionProposal:
                return CocoIouType.BoundingBox
            case self.Keypoints:
                return CocoIouType.Keypoints
            case self.Segmentation:
                return CocoIouType.Segmentation
            case _:
                raise ValueError(f"IouType ({self.value}) is not an original COCO IouType.")

    @staticmethod
    def build_iou_types(cfg: CfgNode) -> tuple["IouType", ...]:
        """Build the tuple of IouTypes that should be COCO-evaluated based on the config."""
        # Either only evaluate region proposals (RPN_ONLY) or evaluate the different heads
        if cfg.MODEL.RPN_ONLY:
            return IouType.RegionProposal,
        iou_types = IouType.BoundingBox,
        if cfg.MODEL.MASK_ON:
            iou_types += IouType.Segmentation,
        if cfg.MODEL.KEYPOINT_ON:
            iou_types += IouType.Keypoints,
        if cfg.MODEL.RELATION_ON:
            iou_types += IouType.Relations,
        if cfg.MODEL.ATTRIBUTE_ON:
            iou_types += IouType.Attributes,
        return iou_types


class SGGEvaluationMode(Enum):
    """Evaluation modes for Scene Graph evaluation."""
    PredicateClassification = "predcls"  # Given bbox+label, predict relation
    SceneGraphClassification = "sgcls"  # Given bbox, predict bbox label and relation
    SceneGraphGeneration = "sgdet"  # Predict everything from scratch; match if each box's IoU > threshold
    PhraseDetection = "phrdet"  # predict everything from scratch; match if box union's IoU > 0.5

    @staticmethod
    def build(cfg: CfgNode) -> SGGEvaluationMode:
        """Returns the correct SGGEvaluationMode based on the current config."""
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                return SGGEvaluationMode.PredicateClassification
            return SGGEvaluationMode.SceneGraphClassification
        if cfg.TEST.RELATION.IOU_THRESHOLD != -1:
            return SGGEvaluationMode.SceneGraphGeneration
        return SGGEvaluationMode.PhraseDetection


class ConfMatrix:
    def __init__(self, num_classes: int):
        """:param num_classes:  number of classes including the background"""
        self.num_classes = num_classes
        self.conf_matrix = torch.zeros([num_classes, num_classes]).long()

    def __add__(self, other: Self) -> Self:
        """Add in-place."""
        assert self.num_classes == other.num_classes
        self.conf_matrix += other.conf_matrix
        return self

    def cuda(self):
        self.conf_matrix = self.conf_matrix.cuda()

    def cpu(self):
        self.conf_matrix = self.conf_matrix.cpu()

    def normalize_confusion(self) -> torch.Tensor:
        """Normalize along rows."""
        matrix = self.conf_matrix.clone().float()
        sum_vec = torch.sum(matrix, 1).unsqueeze(1)
        sum_vec[sum_vec == 0] = 1
        return matrix / sum_vec

    def init_perfect(self):
        """Initialize matrix to eye(num_classes)."""
        self.conf_matrix.fill_(0)
        for i in range(self.conf_matrix.size(0)):
            self.conf_matrix[i, i] = 1

    # Reset the Matrix
    def init_zero(self):
        """Initialize to empty matrix."""
        self.conf_matrix.fill_(0)

    def add_prediction(self, mask: torch.Tensor, pred: torch.Tensor) -> Self:
        """Update matrix given prediction and ground truth."""

        self.conf_matrix += torch.bincount(
            (self.num_classes * mask.reshape(-1) + pred.reshape(-1)).to(self.conf_matrix.dtype),
            minlength=self.num_classes ** 2,
        ).view(self.num_classes, self.num_classes)

        return self

    def get_class_dice(self) -> torch.Tensor:
        """
        Return a tensor with Dice per class.
        Note: will contain nan values if the class is not present.
        """
        return 2 * self.conf_matrix.diag().float() / (
                torch.sum(self.conf_matrix, 0).float() +
                torch.sum(self.conf_matrix, 1).float()
        )

    def get_dice(self) -> torch.Tensor:
        """Return mean Dice for classes that are present (including background) in the ground truth."""
        dice = self.get_class_dice()
        return torch.mean(dice[torch.isnan(dice) == 0])

    def get_class_iou(self) -> torch.Tensor:
        """
        Return a tensor with IoU per class.
        Note: will contain nan values if the class is not present.
        """
        return self.conf_matrix.diag().float() / (
                torch.sum(self.conf_matrix, 0).float() +
                torch.sum(self.conf_matrix, 1).float() -
                self.conf_matrix.diag().float()
        )

    def get_iou(self) -> torch.Tensor:
        """Return mean IoU for classes that are present (including background) in the ground truth."""
        iou = self.get_class_iou()
        return torch.mean(iou[torch.isnan(iou) == 0])
