from functools import reduce

import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import MetaContext, ImageContext, SGGResults
from ...utils import SGGEvaluationMode

class SGRecallEvaluator(BaseEvaluator):
    """
    Traditional Recall, implement based on: https://github.com/rowanz/neural-motifs.
    If class_idx is None: then it is the classic Recall.
    If class_idx is > 0: then it is the classic Recall for the specified class.
    """

    @property
    def metric_name(self) -> str:
        if self._class_idx is None:
            return "Recall"
        return f"Recall{self._class_idx}"

    @property
    def short_metric_name(self) -> str:
        if self._class_idx is None:
            return "R"
        return f"R{self._class_idx}"

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
        if self._class_idx is None:
            if len(image_context.gt_rels) == 0:
                return
        else:
            # No relation for specified class
            if not np.any(image_context.gt_rels[:, -1] == self._class_idx):
                return

        # Format predictions to triplet
        pred_rels = np.column_stack((image_context.pred_rel_idxs, 1 + image_context.pred_rel_scores[:, 1:].argmax(1)))
        pred_scores = image_context.pred_rel_scores[:, 1:].max(1)

        # Filter out relations that are predicted as background
        fg_rel_mask = image_context.pred_rel_labels > 0
        pred_rels = pred_rels[fg_rel_mask]
        pred_scores = pred_scores[fg_rel_mask]

        pred_triplets, pred_triplet_boxes, _ = self._format_triplet(
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

        # For each predicted triplet find all matches in the groundtruth
        pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes
        )

        # Compute recall
        res = results[self.metric_name]
        for k in self._meta.ks:
            if len(pred_to_gt) == 0:
                res[k].append(0.)
                continue

            pred_to_gt_k = pred_to_gt[:k]
            if self._class_idx is not None:
                # We only want to consider predicted relations for the specified class within the top K,
                # without messing with the ordering
                fg_rel_mask = pred_rels[:, -1] == self._class_idx
                pred_to_gt_k = [v for v, b in zip(pred_to_gt_k, fg_rel_mask[:k]) if b]
            if len(pred_to_gt_k) == 0:
                # FIXME Since we rely on key ordering for the final print, we need to add this key if we're skipping
                if k not in res.keys():
                    res[k] = []
                continue

            match = reduce(np.union1d, pred_to_gt_k)  # Computes the set of all matches to gt
            if self._class_idx is None:
                num_get_rels = image_context.gt_rels.shape[0]
            else:
                num_get_rels = np.sum(image_context.gt_rels[:, -1] == self._class_idx)
            rec = len(match) / num_get_rels
            res[k].append(rec)
