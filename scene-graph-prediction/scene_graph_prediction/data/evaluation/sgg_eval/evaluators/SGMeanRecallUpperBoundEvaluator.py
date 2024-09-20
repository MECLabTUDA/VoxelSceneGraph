from functools import reduce

import numpy as np

from .BaseEvaluator import BaseUpperBoundEvaluator
from .abstractions import SGGResults, ImageContext


class SGMeanRecallUpperBoundEvaluator(BaseUpperBoundEvaluator):
    """
    Best Mean Recall achievable: Proposed in: https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
    Note: Recall is averaged across labels.
    """

    metric_name = "MeanRecallUpperBound"
    short_metric_name = "mR UP"

    def evaluate(self, image_context: ImageContext, results: SGGResults):
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
        gt_rels = image_context.gt_rels
        res = results[self.metric_name]
        for k in self._meta.ks:
            if len(pred_to_gt) == 0:
                res[k].append(0.)
                continue

            # The following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])

            # Compute Mean Recall for each category independently
            # Compute hits
            recall_hit = [0] * self._meta.n_rel_classes
            for idx in range(len(match)):
                match_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(match_label)] += 1
                recall_hit[0] += 1

            # Compute count in GT annotation
            recall_count = [0] * self._meta.n_rel_classes
            for idx in range(gt_rels.shape[0]):
                match_label = gt_rels[idx, 2]
                recall_count[int(match_label)] += 1
                recall_count[0] += 1

            # Note: we need to have a -1 placeholder, otherwise the order of the classes is not respected,
            # and we cannot average
            # Note: count for class 0 is actually the normal recall
            res[k].append([
                recall_hit[n] / recall_count[n] if recall_count[n] > 0 else -1
                for n in range(self._meta.n_rel_classes)
            ])

    def aggregate(self, results: SGGResults):
        res = results[self.metric_name]
        for k, v in res.items():
            sum_recall = 0
            num_rel = 0  # Safeguard against some relation classes not being present in the val/test set
            for n in range(1, self._meta.n_rel_classes):
                recalls = [r[n] for r in res[k] if r[n] != -1]
                if recalls:
                    sum_recall += np.mean(recalls)
                    num_rel += 1
            res[k] = sum_recall / num_rel
