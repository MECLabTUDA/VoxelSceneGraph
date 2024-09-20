"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import TYPE_CHECKING

import torch

from .box_list_fields import AnnotationField, PredictionField
from .segmentation_mask import BinaryMaskList

if TYPE_CHECKING:
    from .box_list import BoxList


class FieldExtractor:
    """
    Helper class designed to extract specific information from a BoxList.
    E.g. we want to get the labelmap. It can be present,
    but it can also be automatically computed from the binary masks and the corresponding labels.
    Note: Since BoxLists with predictions can also contain AnnotationFields,
          but no ground truth BoxList contains PredictionFields,
          we consider that a BoxList with at least one PredictionField is a prediction.
    """

    # TODO maybe at some point we want to be able to force extracting some fields using the predictions,
    #  if we had to it like that for another field

    @staticmethod
    def labels(boxlist: "BoxList") -> BinaryMaskList:
        """
        Extract the box labels.
        :raises ValeError if the field could not be extracted.
        """
        if boxlist.has_field(PredictionField.PRED_LABELS):
            return boxlist.PRED_LABELS
        if boxlist.has_field(AnnotationField.LABELS):
            return boxlist.LABELS
        raise ValueError("Could not retrieve the labels.")

    @staticmethod
    def masks(boxlist: "BoxList") -> BinaryMaskList:
        """
        Extract binary masks.
        :raises ValeError if the field could not be extracted.
        """
        # Handle prediction case first
        if boxlist.has_field(PredictionField.PRED_MASKS):
            return boxlist.PRED_MASKS

        # Then handle ground truth case
        if boxlist.has_field(AnnotationField.MASKS):
            return boxlist.MASKS

        # Convert from labelmap
        if not boxlist.has_field(AnnotationField.LABELMAP):
            raise ValueError("Could not compute the binary masks given the available fields.")

        return FieldExtractor._labelmap_to_masks(boxlist.LABELMAP, boxlist.size)

    @staticmethod
    def _labelmap_to_masks(labelmap: torch.Tensor, size: tuple[int, ...]) -> BinaryMaskList:
        """
        Convert a labelmap to binary masks in a BinaryMaskList.
        :param labelmap: the labelmap
        :param size: the original image size (pre-padding i.e. the boxlist.size).
        """
        masks = []
        for obj_id in torch.unique(labelmap):
            if obj_id == 0:
                # Skip background
                continue
            mask = torch.zeros_like(labelmap, dtype=torch.uint8, device=labelmap.device)
            mask[labelmap == obj_id] = 1
            masks.append(mask)
        return BinaryMaskList(masks, size)

    @staticmethod
    def labelmap(boxlist: "BoxList") -> torch.LongTensor:
        """
        Extract the labelmap.
        :raises ValeError if the field could not be extracted.
        """
        # Handle prediction case first
        if boxlist.has_field(PredictionField.PRED_MASKS):
            return FieldExtractor._masks_to_labelmap(boxlist.PRED_MASKS)

        # Then handle ground truth case
        if boxlist.has_field(AnnotationField.LABELMAP):
            return boxlist.LABELMAP

        if boxlist.has_fields(AnnotationField.MASKS):
            return FieldExtractor._masks_to_labelmap(boxlist.MASKS)

        raise ValueError("Could not compute the labelmap given the available fields.")

    @staticmethod
    def _masks_to_labelmap(masks: BinaryMaskList) -> torch.LongTensor:
        """
        Convert a labelmap to binary masks in a BinaryMaskList.
        """
        size = masks.masks.shape[1:]
        labelmap = torch.zeros(size, dtype=torch.uint8, device=masks.device)
        for idx, mask in enumerate(masks.masks, start=1):
            labelmap[mask == 1] = idx
        # noinspection PyTypeChecker
        return labelmap

    @staticmethod
    def segmentation(boxlist: "BoxList") -> torch.LongTensor:
        """
        Extract the semantic segmentation.
        :raises ValeError if the field could not be extracted.
        """
        # Handle prediction case first
        if boxlist.has_field(PredictionField.PRED_SEGMENTATION):
            return boxlist.PRED_SEGMENTATION

        # Check if we can compute a labelmap to compute the segmentation from
        if boxlist.has_fields(PredictionField.PRED_MASKS, PredictionField.PRED_LABELS):
            pred_labelmap = FieldExtractor._masks_to_labelmap(boxlist.PRED_MASKS)
            return FieldExtractor._labelmap_and_labels_to_segmentation(pred_labelmap, boxlist.PRED_LABELS)

        # Then handle ground truth case
        if boxlist.has_field(AnnotationField.SEGMENTATION):
            return boxlist.SEGMENTATION

        # Check if we can compute a labelmap to compute the segmentation from
        if boxlist.has_field(AnnotationField.LABELMAP):
            labelmap = boxlist.LABELMAP
        elif boxlist.has_fields(AnnotationField.MASKS, AnnotationField.LABELS):
            labelmap = FieldExtractor._masks_to_labelmap(boxlist.MASKS)
        else:
            raise ValueError("Could not compute the semantic segmentation given the available fields.")

        return FieldExtractor._labelmap_and_labels_to_segmentation(labelmap, boxlist.LABELS)

    @staticmethod
    def _labelmap_and_labels_to_segmentation(labelmap: torch.LongTensor, labels: torch.LongTensor) -> torch.LongTensor:
        """
        Convert a labelmap to binary masks in a BinaryMaskList.
        """
        seg = torch.zeros_like(labelmap)
        for box_id, label in enumerate(labels, start=1):
            # noinspection PyTypeChecker
            seg[labelmap == box_id] = label
        # noinspection PyTypeChecker
        return seg

    @staticmethod
    def relation_matrix(boxlist: "BoxList") -> torch.LongTensor:
        """
        Extract the NxN relation matrix.
        :raises ValeError if the field could not be extracted.
        """
        if boxlist.has_fields(PredictionField.REL_PAIR_IDXS, PredictionField.PRED_REL_LABELS):
            relation_matrix = torch.zeros((len(boxlist), len(boxlist)), dtype=torch.uint8, device=boxlist.boxes.device)
            for (sub_idx, ob_idx), lbl in zip(boxlist.REL_PAIR_IDXS, boxlist.PRED_REL_LABELS):
                relation_matrix[sub_idx, ob_idx] = lbl
            # noinspection PyTypeChecker
            return relation_matrix

        if boxlist.has_field(AnnotationField.RELATIONS):
            return boxlist.RELATIONS

        raise ValueError("Could not compute the relation matrix given the available fields.")
