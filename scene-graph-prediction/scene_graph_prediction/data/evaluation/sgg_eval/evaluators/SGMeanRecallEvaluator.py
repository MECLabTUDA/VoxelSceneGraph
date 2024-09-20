from functools import reduce

import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGMeanRecallEvaluator(BaseEvaluator):
    """
    Mean Recall: Proposed in: https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
    Note: Recall is averaged across labels.
    """

    metric_name = "MeanRecall"
    short_metric_name = "mR"

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

        gt_rels = image_context.gt_rels
        res = results[self.metric_name]
        for k in self._meta.ks:
            # Compute count in GT annotation
            recall_count = [0] * self._meta.n_rel_classes
            for idx in range(gt_rels.shape[0]):
                match_label = gt_rels[idx, 2]
                recall_count[int(match_label)] += 1
                recall_count[0] += 1

            # Abort if no prediction (but we still need to know which GT classes are present)
            if len(pred_to_gt) == 0:
                res[k].append([
                    0. if recall_count[n] > 0 else -1
                    for n in range(self._meta.n_rel_classes)
                ])
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

            # Per class recall for each image
            # Note: we need to have a -1 placeholder, otherwise the order of the classes is not respected,
            # and we cannot average
            # Note: count for class 0 is actually the normal (overall) recall
            res[k].append([
                recall_hit[n] / recall_count[n] if recall_count[n] > 0 else -1
                for n in range(self._meta.n_rel_classes)
            ])

    def aggregate(self, results: SGGResults):
        res = results[self.metric_name]
        # Fpr each K, average per-class (over all images) then average across classes
        for k, v in res.items():
            sum_recall = 0
            num_rel = 0  # Safeguard against some relation classes not being present in the val/test set
            for n in range(1, self._meta.n_rel_classes):
                recalls = [r[n] for r in res[k] if r[n] != -1]
                if recalls:
                    # Average over all images for the given relation class
                    sum_recall += np.mean(recalls)
                    num_rel += 1
            res[k] = sum_recall / num_rel
