from functools import reduce

import numpy as np

from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGPairAccuracyEvaluator(BaseEvaluator):
    """
    Gives Ground Truth Object-Subject Pairs. Calculate Recall for SG-Cls and Pred-Cls.
    Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls.
    Note: it's accuracy and not recall because there is no object score since GT detections used.
    Note: only works when using GT boxes.
    """

    metric_name = "Pair Accuracy"
    short_metric_name = "PA"
    K = 999  # Dummy K value that is big enough that all GT pairs are considered (but only 3 digits for logging)

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

        res = results[self.metric_name]
        if len(pred_to_gt) == 0:
            res[self.K].append((0., image_context.gt_rels.shape[0]))
            return

        # To calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" for sgcls and predcls
        gt_pair = self._prepare_gt_pair(image_context)
        gt_pair_pred_to_gt = []
        # noinspection PyTypeChecker
        for p, is_gt_pair in zip(pred_to_gt, gt_pair):
            if is_gt_pair:
                gt_pair_pred_to_gt.append(p)
        if len(gt_pair_pred_to_gt) > 0:
            gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt)
        else:
            gt_pair_match = []
        # Number of matches, number in gt
        res[self.K].append((len(gt_pair_match), image_context.gt_rels.shape[0]))

    def aggregate(self, results: SGGResults):
        res = results[self.metric_name]
        array = np.array(res[self.K])
        if len(array) == 0:
            res[self.K] = 0.
            return
        hits = array[:, 0]
        counts = array[:, 1]
        res[self.K] = np.mean(hits) / np.mean(counts)  # Note: more stable than np.sum(hits) / np.sum(counts)

    def _prepare_gt_pair(self, image_context: ImageContext) -> np.ndarray:
        # We are using GT boxes, so the box ordering is the same in predictions as in the ground truth
        n_rel_classes = self._meta.n_rel_classes
        pred_pair_idx = image_context.pred_rel_idxs[:, 0] * n_rel_classes + image_context.pred_rel_idxs[:, 1]
        gt_pair_idx = image_context.gt_rels[:, 0] * n_rel_classes + image_context.gt_rels[:, 1]
        # noinspection PyUnresolvedReferences
        return (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0
