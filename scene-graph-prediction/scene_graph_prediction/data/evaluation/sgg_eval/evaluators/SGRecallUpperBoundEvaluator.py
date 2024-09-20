from functools import reduce

import numpy as np

from .BaseEvaluator import BaseUpperBoundEvaluator
from .abstractions import SGGResults, ImageContext


class SGRecallUpperBoundEvaluator(BaseUpperBoundEvaluator):
    """Best Recall achievable, implement based on: https://github.com/rowanz/neural-motifs."""

    metric_name = "RecallUpperBound"
    short_metric_name = "R UP"

    def evaluate(self, image_context: ImageContext, results: SGGResults):
        """
        Evaluate prediction.
        :param image_context: context about the image to evaluate.
        :param results: object that will store the results and will be used to summarize the results.
        """
        # Recall- and pair-acc-based metrics do not need to run on images with no relations
        if len(image_context.gt_rels) == 0:
            return

        # Prepare all possible combinations (except reflexive relations) and prepare boxes
        n_pred_obj = image_context.pred_boxes.shape[0]
        if n_pred_obj == 0:
            return

        sub_id, ob_id = np.meshgrid(list(range(n_pred_obj)), list(range(n_pred_obj)))
        non_self_rel_mask = np.ones((n_pred_obj, n_pred_obj), dtype=bool) ^ np.eye(n_pred_obj, dtype=bool)
        sub_id, ob_id = sub_id[non_self_rel_mask], ob_id[non_self_rel_mask]
        pred_triplet_boxes = np.column_stack((
            image_context.pred_boxes[sub_id],
            image_context.pred_boxes[ob_id]
        ))

        # Format gt to triplet
        gt_triplets, gt_triplet_boxes, _ = self._format_triplet(
            image_context.gt_rels, image_context.gt_labels, image_context.gt_boxes
        )

        # Prepare representation for GT and predictions
        if self._meta.obj_class_agnostic_upper_bound:
            # We need to fake the relation representations such that we get a match for all pairs
            pred_repr = np.ones((sub_id.shape[0], 2), dtype=int)
            gt_repr = np.ones((gt_triplets.shape[0], 2), dtype=int)
        else:
            # Since we do not allow object reclassification, we will compute matches based on (sub, ob) label pairs
            sub_lbl = image_context.pred_labels[sub_id]
            ob_lbl = image_context.pred_labels[ob_id]
            pred_repr = np.column_stack([sub_lbl, ob_lbl])
            gt_repr = gt_triplets[:, ::2]  # Just skip the predicate class in the middle

        # For each predicted triplet find all matching groundtruth triplets
        # I.e. first match on labels-triplet then check boxes
        pred_to_gt = self._compute_pred_matches(
            gt_repr,
            pred_repr,
            gt_triplet_boxes,
            pred_triplet_boxes
        )

        # We greedily reorder prediction boxes to maximize the number of matches (since the top-K pred will be selected)
        pred_to_gt = sorted(pred_to_gt, key=lambda matches: len(matches), reverse=True)

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
