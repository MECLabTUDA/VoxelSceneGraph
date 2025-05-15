from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from pycocotools3d.coco import COCO, COCO3d
from torch.utils.data import Dataset as _Dataset
from typing_extensions import NotRequired
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import Split
from scene_graph_prediction.data.transforms import Compose
from scene_graph_prediction.structures import BoxList
from scene_graph_prediction.utils.miscellaneous import reindex_tensor, remap_id_tensor, remap_segmentation


class ImgInfo(TypedDict):
    file_path: str
    depth: NotRequired[int]
    height: int
    width: int


ObjectClasses = list[str]
AttributeClasses = list[str]
RelationClasses = list[str]


class DatasetStatistics(TypedDict):
    """Used for RelationContext in ROIRelationHeads."""
    fg_matrix: torch.LongTensor  # ((obj1, obj2), rel) co-occurrence matrix of a pair of object class and relation class
    pred_dist: torch.Tensor  # Log probability of fg_matrix
    obj_classes: ObjectClasses
    rel_classes: AttributeClasses
    att_classes: RelationClasses


class Dataset(_Dataset, ABC):
    """Interface for our use of datasets. Exposes a few extra methods."""
    BACKGROUND_CLASS_NAME = "Background"

    n_dim: int = 0
    categories: list[str]  # Mapping from category id to category name
    contiguous_category_id_to_json_id: dict[int, int]  # Mapping from contiguous cat id to original cat id, including bg
    json_category_id_to_contiguous_id: dict[int, int]  # Mapping from original cat id to contiguous cat id, including bg
    contiguous_image_id_to_json_id: dict[int, int]  # Mapping from contiguous img id to original img id

    def __init__(self, cfg: CfgNode, datasets_dir: str, transforms: Compose, split: Split):
        self._cfg = cfg
        self._datasets_dir = datasets_dir  # Root of dataset directory
        self._transforms = transforms  # Transformations to apply to the data, i.e. conv to tensor + image aug
        self._split = split  # Split: train or val or test

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BoxList, int]:
        """
        Note: also sets the IMG_PATH field in the BoxList target.
        :returns: img, target, idx.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_img_info(self, index: int) -> ImgInfo:
        """Returns a dict containing the shape of the image."""
        raise NotImplementedError

    def reindex_groundtruth(self, target: BoxList, mapping: dict[int, int] | None = None) -> BoxList:
        """
        Given a mapping (typically json_category_id_to_contiguous_id), remap ids in the ground truth.
        Note: maps LABELS and SEGMENTATION (if present) fields
        Note: makes a lazy copy of the BoxList
        """
        if mapping is None:
            mapping = self.json_category_id_to_contiguous_id

        target = target.copy_with_all_fields()
        if all(k == v for k, v in mapping.items()):
            # Mapping is no-op
            return target

        target.LABELS = remap_id_tensor(target.LABELS, mapping)
        if target.has_field(target.AnnotationField.SEGMENTATION):
            target.SEGMENTATION = remap_segmentation(target.SEGMENTATION, mapping)

        return target

    def reindex_prediction(self, boxes: BoxList, mapping: dict[int, int] | None = None) -> BoxList:
        """
        Reverse operation of reindex_groundtruth.
        Should be applied to all predictions, such that their ids match the knowledge graph definition.
        Note: maps PRED_LABELS, PRED_LOGITS, BOXES_PER_CLS, PRED_SEGMENTATION, PRED_CLS_SCORES (if present) fields
        Note: no copy required
        """
        if mapping is None:
            mapping = self.contiguous_category_id_to_json_id

        if all(k == v for k, v in mapping.items()):
            # Mapping is no-op
            return boxes

        boxes.PRED_LABELS = remap_id_tensor(boxes.PRED_LABELS, mapping)
        if boxes.has_field(boxes.PredictionField.PRED_LOGITS):
            boxes.PRED_LOGITS = reindex_tensor(boxes.PRED_LOGITS, mapping)
        if boxes.has_field(boxes.PredictionField.BOXES_PER_CLS):
            boxes.BOXES_PER_CLS = reindex_tensor(
                boxes.BOXES_PER_CLS.view(len(boxes), 2 * boxes.n_dim, - 1), mapping, dim=2
            ).view(len(boxes), -1)
        if boxes.has_field(boxes.PredictionField.PRED_CLS_SCORES):
            boxes.PRED_CLS_SCORES = reindex_tensor(boxes.PRED_CLS_SCORES, mapping)
        if boxes.has_field(boxes.PredictionField.PRED_SEGMENTATION):
            boxes.PRED_SEGMENTATION = remap_segmentation(boxes.PRED_SEGMENTATION, mapping)

        return boxes


class COCOEvaluableDataset(Dataset, ABC):
    # Note: since COCO also support attribute and relation annotation, this information should also be made available
    predicates: list[str]  # Mapping from int to relation name
    attributes: list[str]  # Mapping from int to attribute name
    filenames: list[str]  # List image paths, needs to match data indexing

    @abstractmethod
    def get_groundtruth(self, index: int) -> BoxList:
        """
        :param index:
        :returns: a RelationHeadTarget
        """
        raise NotImplementedError

    @property
    def coco(self) -> COCO | COCO3d:
        """
        COCO object: used for COCO evaluation and adjusted for binary classification if RPN_ONLY is True.
        The computation should be done JIT, as the computation needs to be done only once, is slow and memory-expensive.
        We also want to avoid computing this for each dataset when using a ConcatDataset.
        """
        raise NotImplementedError


class SGGEvaluableDataset(COCOEvaluableDataset, ABC):
    """Note: SGGEvaluableDatasets need to be COCO-evaluable as we may want to evaluate improved predictions."""

    def get_statistics(self) -> DatasetStatistics:
        """
        Note: only needs to be overwritten is the dataset is intended to support relation prediction.
        :returns: The statistics needed for RelationContext in ROIRelationHeads.
        """
        raise NotImplementedError
