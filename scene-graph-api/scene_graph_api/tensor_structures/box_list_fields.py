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
from enum import Enum


class AnnotationField(Enum):
    """Fields used to store annotations."""
    ATTRIBUTES = "attributes"
    IMAGE_ATTRIBUTES = "image_attributes"
    LABELS = "labels"  # Tensor with the int label for each box
    KEYPOINTS = "keypoints"
    MASKS = "masks"  # Individual binary masks for each object as a BinaryMaskList
    IMPORTANCE = "importance"  # Weight for each object; allows to focus the training on more relevant objects.
    #                            Usually computed on the fly based on relations as (1 + #rels implicating this object)
    LABELMAP = "labelmap"  # Instance segmentation mask for the whole image
    SEGMENTATION = "segmentation"  # Semantic segmentation mask for the whole image
    RELATIONS = "relations"  # Relations as NxN matrix with relation label for each pair of object
    AFFINE_MATRIX = "affine_matrix"  # Affine matrix for radiology images, removes the need to read the image for this
    IMG_PATH = "path"  # Path to the corresponding image; can be useful when saving results

    def __lt__(self, other):
        if isinstance(other, Enum):
            return self.value < other.value
        return self.value < other

    def __repr__(self):
        return f"'{self.value}'"

    def indexing_power(self) -> int:
        match self:
            case AnnotationField.ATTRIBUTES | AnnotationField.LABELS | \
                 AnnotationField.MASKS | AnnotationField.IMPORTANCE:
                return 1
            case AnnotationField.RELATIONS:
                return 2
            case _:
                return 0


class PredictionField(Enum):
    """Commonly used field names."""
    # Related to predictions
    PRED_SEGMENTATION = "pred_segmentation"  # Predicted semantic segmentation for the whole image (LongTensor)
    PRED_SEGMENTATION_LOGITS = "pred_segmentation_logits"  # Predicted logits for the semantic segmentation
    PRED_LABELS = "pred_labels"  # Predicted object class
    PRED_MASKS = "pred_masks"  # Predicted binary instance-segmentation for each bbox
    PRED_SCORES = "pred_scores"  # Predicted score for PREDICTED OBJECT CLASS only
    PRED_CLS_SCORES = "pred_cls_scores"  # Predicted score for each class (background included); box-head internal
    PRED_LOGITS = "pred_logits"  # Class logits for detected objects;
    #                              Can be refined in the relation head, however not used for predcls
    ATTRIBUTE_LOGITS = "attribute_logits"  # Logits for all attribute classes
    PRED_ATTRIBUTES = "pred_attributes"  # Post-sigmoid logits
    KEYPOINT_LOGITS = "keypoint_logits"
    # Relations
    PRED_REL_CLS_SCORES = "pred_rel_cls_scores"  # Predicted score for each predicate class (including background)
    PRED_REL_LABELS = "pred_rel_labels"  # Predicted predicate class; mostly unused, rather computed as argmax of scores
    REL_PAIR_IDXS = "rel_pair_idxs"  # (object idx, subject idx) pairs for predicted relations
    BOXES_PER_CLS = "boxes_per_cls"  # Added by the box head; bbox prediction for each class (including the background)
    #                                  (required (for sgdet only) because the obj class prediction is not final yet,
    #                                  and we might need to switch to the predicted box for the new class)
    # RPN
    MATCHED_IDXS = "matched_idxs"  # RPN-internal
    REGRESSION_TARGETS = "regression_targets"  # RPN-internal
    VISIBILITY = "visibility"  # Whether an anchor is enabled (can be disabled when partially outside the image); RPN-in
    OBJECTNESS = "objectness"  # Objectness score for regions predicted by the RPN (two-stage methods only)
    ANCHOR_LVL = "anchor_lvl"  # Feature map level for each anchor; used for ATSS matching; no need to be contiguous

    def __lt__(self, other):
        if isinstance(other, Enum):
            return self.value < other.value
        return self.value < other

    def __repr__(self):
        return f"'{self.value}'"

    def indexing_power(self) -> int:
        match self:
            case PredictionField.PRED_SEGMENTATION | PredictionField.KEYPOINT_LOGITS | PredictionField.REL_PAIR_IDXS:
                return 0
            case PredictionField.PRED_REL_CLS_SCORES | PredictionField.PRED_REL_LABELS | \
                 PredictionField.REL_PAIR_IDXS:
                # Has an indexing power of 0, but require some processing during indexing
                return 0
            case _:
                return 1
