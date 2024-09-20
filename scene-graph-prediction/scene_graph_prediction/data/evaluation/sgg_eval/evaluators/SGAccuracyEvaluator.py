import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGPairAccuracyEvaluator(BaseEvaluator):
    """New metric: given all predicted relations, compute the classification accuracy."""

    metric_name = "Accuracy"
    short_metric_name = "A"
    K = 999  # Dummy K value that is big enough that all GT pairs are considered (but only 3 digits for logging)

    def evaluate(self, image_context: ImageContext, results: SGGResults):
        """
        Evaluate prediction.
        :param image_context: context about the image to evaluate.
        :param results: object that will store the results and will be used to summarize the results.
        """
        # Here, we need to consider bg relations being predicted as fg or accurately as bg.
        if len(image_context.pred_rel_labels) == 0:
            return

        # Format predictions to triplet
        pred_rels = np.column_stack((image_context.pred_rel_idxs, image_context.pred_rel_scores.argmax(1)))
        pred_scores = image_context.pred_rel_scores.max(1)

        # ==============================================================================================================
        # First pass, we check whether fg relations are correctly predicted
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

        # For each predicted triplet find all matches in the groundtruth
        pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes
        )
        has_match = np.array([len(e) > 0 for e in pred_to_gt])

        # Predicted as fg and has at least one match
        fg_correct = np.logical_and(pred_triplets[1] > 0, has_match)

        # ==============================================================================================================
        # Second pass, we check whether bg relations are correctly predicted
        # This is done in the same way than for class agnostic metric upper bound computation (skip
        pred_to_gt = self._compute_pred_matches(
            gt_triplets[:, ::2],
            pred_triplets[:, ::2],
            gt_triplet_boxes,
            pred_triplet_boxes
        )
        has_match = np.array([len(e) > 0 for e in pred_to_gt])

        # Predicted as bg and object pair has no match
        bg_correct = np.logical_and(pred_triplets[1] == 0, ~has_match)

        # ==============================================================================================================
        # Final result
        results[self.metric_name][self.K].append((np.sum(np.logical_or(bg_correct, fg_correct)), has_match.shape[0]))

    # FIXME need to factorize the aggregation method
    def aggregate(self, results: SGGResults):
        res = results[self.metric_name]
        array = np.array(res[self.K])
        if len(array) == 0:
            res[self.K] = 0.
            return
        hits = array[:, 0]
        counts = array[:, 1]
        res[self.K] = np.mean(hits) / np.mean(counts)  # Note: more stable than np.sum(hits) / np.sum(counts)
