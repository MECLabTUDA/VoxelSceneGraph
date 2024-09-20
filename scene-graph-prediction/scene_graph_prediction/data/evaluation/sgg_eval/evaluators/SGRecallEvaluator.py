from functools import reduce

import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGRecallEvaluator(BaseEvaluator):
    """Traditional Recall, implement based on: https://github.com/rowanz/neural-motifs."""

    metric_name = "Recall"
    short_metric_name = "R"

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

            # The following code is copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])  # Computes the set of all matches to gt
            rec = len(match) / image_context.gt_rels.shape[0]
            res[k].append(rec)
