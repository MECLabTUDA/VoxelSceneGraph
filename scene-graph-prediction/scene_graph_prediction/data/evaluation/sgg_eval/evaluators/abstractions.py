from dataclasses import dataclass

import numpy as np

from ...utils import SGGEvaluationMode


@dataclass
class MetaContext:
    """All meta information needed for metric computation."""
    # Evaluation mode: used for debugging as all evaluators already know which mode is used
    mode: SGGEvaluationMode
    # Number of relation classes
    n_rel_classes: int
    # Whether to compute upper bound of metrics given box predictions (is False, we do regular metric computation)
    compute_upper_bound: bool
    # Allow reclassification of objects by the
    obj_class_agnostic_upper_bound: bool
    # IoU threshold to use for object detection (if not phrdet)
    iou_thr: float
    # Top-K values to use for evaluation e.g. [20, 50, 100]
    ks: list[int]
    # For no graph constraint, up to how many predictions are considered per (sub, ob) pair
    no_gc_top_n_pred: int


@dataclass
class ImageContext:
    """
    All information about ground truth and predicted objects/relations for a single image.
    Note: mostly a holder class for all commonly used BoxList fields.
    """
    # About groundtruth:
    gt_boxes: np.ndarray
    gt_labels: np.ndarray
    gt_rels: np.ndarray  # (idx sub, idx ob, lbl rel)
    # About predictions:
    pred_boxes: np.ndarray
    pred_labels: np.ndarray
    pred_obj_scores: np.ndarray
    pred_rel_idxs: np.ndarray | None  # Relation with bbox ids if we're not computing upper bounds
    pred_rel_scores: np.ndarray | None  # If we're not computing upper bounds
    pred_rel_labels: np.ndarray | None  # If we're not computing upper bounds


# Some type hints for results structure:
# {metric_name: {K (int): [per image value(s)]}}
PerImageResultsAtK = list
ResultsAtK = float
SGGMetricResults = dict[int, ResultsAtK | PerImageResultsAtK]
SGGResults = dict[str, SGGMetricResults]
