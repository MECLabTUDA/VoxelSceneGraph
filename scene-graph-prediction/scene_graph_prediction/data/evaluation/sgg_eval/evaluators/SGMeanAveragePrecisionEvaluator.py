import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGMeanAveragePrecisionEvaluator(BaseEvaluator):
    """Mean Average Precision evaluator."""

    metric_name = "MeanAveragePrecision"
    short_metric_name = "mAP"

    def evaluate(self, image_context: ImageContext, results: SGGResults):
        """
        Evaluate prediction.
        :param image_context: context about the image to evaluate.
        :param results: object that will store the results and will be used to summarize the results.
        """
        # Format predictions to triplet
        pred_rels = np.column_stack((image_context.pred_rel_idxs, 1 + image_context.pred_rel_scores[:, 1:].argmax(1)))
        pred_scores = image_context.pred_rel_scores[:, 1:].max(1)

        # Filter out relations that are predicted as background
        # Not really needed for mAP computation but slightly faster for pre-processing
        fg_rel_mask = image_context.pred_rel_labels > 0
        pred_rels = pred_rels[fg_rel_mask]
        pred_scores = pred_scores[fg_rel_mask]

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

        # Convert the gt triplet idx match to the corresponding relation label
        pred_to_gt_lbl = [[gt_triplets[gt_id][1] for gt_id in matches] for matches in pred_to_gt]

        res = results[self.metric_name]
        for k in self._meta.ks:
            # Safeguard against no predictions causing the defaultdict to never instantiate a list for k
            if len(pred_scores) == 0:
                continue

            for pred_score, pred_lbl, lbl_matches in zip(pred_scores[:k], pred_rels[:, -1][:k], pred_to_gt_lbl[:k]):
                if pred_lbl in lbl_matches:
                    # We have at least one match for this relation
                    tp = 1
                    fp = 0
                else:
                    tp = 0
                    fp = 1
                # Save all of this information, because during the aggregation we need to:
                # 1. Filter per relation class
                # 2. Sort by score
                # 3. Compute the cumulative sum of tp and fp to get the AP
                res[k].append([pred_score, pred_lbl, tp, fp])

    def aggregate(self, results: SGGResults):
        res = results[self.metric_name]
        for k in self._meta.ks:
            if k not in res:
                res[k] = 0.
                continue

            # Unpack
            array = np.array(res[k])
            scores = array[:, 0]
            lbl = array[:, 1]
            tp = array[:, 2]
            fp = array[:, 3]

            ap_sum = 0.
            n_class = 0
            # Compute the AP for each class and sum them
            for gt_class in range(1, self._meta.n_rel_classes):
                # Filter entries for this class
                is_right_class = lbl == gt_class
                if not np.any(is_right_class):
                    # Skip class as there is no match
                    continue
                indices = np.argsort(-scores[is_right_class], kind="mergesort")
                # Compute array of recalls and precisions
                tp_sum = np.cumsum(tp[is_right_class][indices])
                fp_sum = np.cumsum(fp[is_right_class][indices])
                n_pos = np.sum(is_right_class)
                recalls = tp_sum / n_pos
                precisions = tp_sum / (tp_sum + fp_sum)
                # Add sentinel values (the indices computed by np.where do not work otherwise...)
                recalls = np.concatenate(([0], recalls, [1]))
                precisions = np.concatenate(([0], precisions, [0]))
                # Compute the precision envelope
                for idx in range(n_pos - 1, 0, -1):
                    if precisions[idx] > precisions[idx - 1]:
                        precisions[idx - 1] = precisions[idx]
                # To calculate area under PR curve, look for points where X axis (recall) changes value...
                indices = np.where(recalls[1:] != recalls[:-1])[0]
                # and sum (\Delta recall) * prec
                ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
                ap_sum += ap
                n_class += 1  # Safeguard against relation class not being present in the val/test split
            res[k] = ap_sum / n_class
