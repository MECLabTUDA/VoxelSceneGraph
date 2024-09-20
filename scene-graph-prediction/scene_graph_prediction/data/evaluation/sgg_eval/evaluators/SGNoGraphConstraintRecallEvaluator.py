from functools import reduce

import numpy as np

from scene_graph_prediction.utils.miscellaneous import argsort_desc
from .BaseEvaluator import BaseEvaluator
from .abstractions import SGGResults, ImageContext


class SGNoGraphConstraintRecallEvaluator(BaseEvaluator):
    """
    No Graph Constraint Recall, implement based on: https://github.com/rowanz/neural-motifs.
    I.e. relation recall where we only care about the subject and object (and not the predicate).
    """

    metric_name = "RecallNoGC"
    short_metric_name = "ng-R"

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
        # Note: compared to SGRecallEvaluator, we need to consider all plausible relations for a given pair of object
        #       (and not only the top 1).
        obj_scores_per_rel = image_context.pred_obj_scores[image_context.pred_rel_idxs].prod(1)
        no_gc_overall_scores = obj_scores_per_rel[:, None] * image_context.pred_rel_scores

        # Why do we do the following indexing operations?
        # 1. We want to limit the number of predictions per (sub, ob) pair,
        # so we first have to figure out the top-N predictions
        # 2. If the model confidently predicts a background relation, it should be accounted as such,
        # i.e. one of the N-spots need to be filled by this background relation
        top_n_classes_by_pair = np.argsort(-no_gc_overall_scores)[:, :self._meta.no_gc_top_n_pred]
        pred_arange = np.tile(np.arange(no_gc_overall_scores.shape[0]), (self._meta.no_gc_top_n_pred, 1)).T
        no_gc_top_n_scores_idxs = np.hstack((pred_arange.reshape(-1, 1), top_n_classes_by_pair.reshape(-1, 1)))
        # 3. However, if we keep the background relations we're screwed,
        # because they are all very confidently predicted and rise to the top.
        # So, we have to remove them again.
        no_gc_top_n_scores_idxs = no_gc_top_n_scores_idxs[no_gc_top_n_scores_idxs[:, 1] > 0]
        # 4. Now we can index the scores and run the argsort to sort predictions across pairs
        filtered_scores = no_gc_overall_scores[no_gc_top_n_scores_idxs[:, 0], no_gc_top_n_scores_idxs[:, 1]]
        no_gc_score_idxs = argsort_desc(filtered_scores)[:max(self._meta.ks), 0]
        # 5. Finally, we just have to tie everything together by building the triplet and indexing the scores
        no_gc_pred_rels = np.column_stack([
            image_context.pred_rel_idxs[no_gc_top_n_scores_idxs[no_gc_score_idxs, 0]],
            no_gc_top_n_scores_idxs[no_gc_score_idxs, 1]
        ])
        no_gc_pred_scores = filtered_scores[no_gc_score_idxs]

        no_gc_pred_triplets, no_gc_pred_triplet_boxes, _ = self._format_triplet(
            no_gc_pred_rels,
            image_context.pred_labels,
            image_context.pred_boxes,
            no_gc_pred_scores,
            image_context.pred_obj_scores
        )

        # Format gt to triplet
        gt_triplets, gt_triplet_boxes, _ = self._format_triplet(
            image_context.gt_rels, image_context.gt_labels, image_context.gt_boxes
        )

        # For each predicted triplet find all matches in the groundtruth
        no_gc_pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            no_gc_pred_triplets,
            gt_triplet_boxes,
            no_gc_pred_triplet_boxes
        )

        res = results[self.metric_name]
        for k in self._meta.ks:
            match = reduce(np.union1d, no_gc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(image_context.gt_rels.shape[0])
            res[k].append(rec_i)
