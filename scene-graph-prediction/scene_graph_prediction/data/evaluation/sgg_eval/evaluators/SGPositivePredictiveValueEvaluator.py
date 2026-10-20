from functools import reduce

import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import MetaContext, ImageContext, SGGResults
from ...utils import SGGEvaluationMode


class SGPositivePredictiveValueEvaluator(BaseEvaluator):
    """
    New metric: given all predicted relations, compute the predictive value.
    If class_idx is None: then it is the classic Positive Predictive Value.
    If class_idx is > 0: then it is the classic Positive Predictive Value for the specified class.
    """

    @property
    def metric_name(self) -> str:
        if self._class_idx is None:
            return "PositivePredictiveValue"
        return f"PositivePredictiveValue{self._class_idx}"

    @property
    def short_metric_name(self) -> str:
        if self._class_idx is None:
            return "PPV"
        return f"PPV{self._class_idx}"

    def __init__(self, mode: SGGEvaluationMode, meta: MetaContext, class_idx: int | None = None):
        super().__init__(mode, meta)
        self._class_idx = class_idx

    def evaluate(self, image_context: ImageContext, results: SGGResults):
        """
        Evaluate prediction.
        :param image_context: context about the image to evaluate.
        :param results: object that will store the results and will be used to summarize the results.
        """
        # Recall- and pair-acc-based metrics do not need to run on images with no relations
        if len(image_context.gt_rels) == 0:
            return

        # Format predictions to triplet
        pred_rels = np.column_stack((image_context.pred_rel_idxs, image_context.pred_rel_scores.argmax(1)))
        pred_scores = image_context.pred_rel_scores.max(1)

        # Filter out relations that are predicted as background
        fg_rel_mask = image_context.pred_rel_labels > 0
        pred_rels = pred_rels[fg_rel_mask]
        pred_scores = pred_scores[fg_rel_mask]

        # ==============================================================================================================
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = self._format_triplet(
            pred_rels,
            image_context.pred_labels,
            image_context.pred_boxes,
            pred_scores,
            image_context.pred_obj_scores
        )

        # Format gt to triplet
        gt_triplets, gt_triplet_boxes, _ = self._format_triplet(
            image_context.gt_rels, image_context.gt_labels, image_context.gt_boxes
        )

        # PPV on filtered set
        # For each predicted triplet find all matches in the ground truth
        pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes
        )

        # Compute PPV
        res = results[self.metric_name]
        # We also want to have this metric when considering all predictions made by the model
        for k in self._meta.ks + [999]:
            if len(pred_to_gt) == 0:
                res[k].append(0.)
                continue

            pred_to_gt_k = pred_to_gt[:k]
            if self._class_idx is not None:
                # We only want to consider predicted relations for the specified class within the top K,
                # without messing with the ordering
                fg_rel_mask = image_context.pred_rel_labels == self._class_idx
                pred_to_gt_k = [v for v, b in zip(pred_to_gt_k, fg_rel_mask[:k]) if b]
            if len(pred_to_gt_k) == 0:
                # FIXME Since we rely on key ordering for the final print, we need to add this key if we're skipping
                if k not in res.keys():
                    res[k] = []
                continue

            match = reduce(np.union1d, pred_to_gt_k)  # Computes the set of all matches to gt
            ppv = len(match) / len(pred_to_gt_k)
            res[k].append(ppv)
