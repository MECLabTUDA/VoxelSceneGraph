from abc import ABC, abstractmethod

import numpy as np

from scene_graph_prediction.utils.miscellaneous import intersect_2d, bboxes_iou_from_np
from .abstractions import MetaContext, ImageContext, SGGResults
from ...utils import SGGEvaluationMode


class BaseEvaluator(ABC):
    """Base class for all evaluators. Each one will compute a different metric."""

    def __init__(self, mode: SGGEvaluationMode, meta: MetaContext):
        self._mode = mode
        self._meta = meta

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric that is being evaluated."""
        raise NotImplementedError

    @property
    @abstractmethod
    def short_metric_name(self) -> str:
        """Short name (up to 6 characters) of the metric that is being evaluated."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, image_context: ImageContext, results: SGGResults):
        """
        Evaluate prediction. The content written to the ResultsAtK (a single aggregate value) or a
        PerImageResultsAtK (list of whatever values for each image).
        Note: in the latter case, the aggregate method needs to be implemented to convert
        the PerImageResultsAtK to a ResultsAtK.
        :param image_context: context about the image to evaluate.
        :param results: object that will store the results and will be used to summarize the results.
        """
        raise NotImplementedError

    def aggregate(self, results: SGGResults):
        """
        Optional step: collect results e.g. compute mean recall from recall.
        I.e. convert the PerImageResultsAtK in the results dict to a ResultsAtK.
        Note: in this default implementation we compute the mean of a list of float (up to one value per image).
        :param results: object that will store the results and will be used to summarize the results.
        """
        res = results[self.metric_name]
        for k, v in res.items():
            res[k] = np.mean(v)

    def summarize(self, results: SGGResults) -> str:
        """Generate string summarizing the results."""
        result_str = "SGG eval: "
        # Safeguard against no prediction (and no keys being set)
        # Also a safeguard for when metrics use their own set of K (PairAcc for instance)
        ks = results[self.metric_name].keys() if results[self.metric_name].keys() else self._meta.ks
        for k in ks:
            result_str += f" {self.short_metric_name:6s} @ {k:3d}: {results[self.metric_name].get(k, 0):.3f}; "
        result_str += f"for mode={self._mode.value}, type={self.metric_name}.\n"
        return result_str

    @staticmethod
    def _format_triplet(
            relations: np.ndarray,
            classes: np.ndarray,
            boxes: np.ndarray,
            predicate_scores: np.ndarray | None = None,
            class_scores: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Helper function to:
        - format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
        - format object boxes for each relation
        - format scores for objects and relation
        :param relations: (#rel, 3) of (sub_id, ob_id, pred_label)
        :param classes: #objs class labels of objects
        :param boxes: (#objs, 2 * n_dim)
        :param predicate_scores: #rel scores for each predicate (optional)
        :param class_scores: #objs scores for each object (optional)
        :returns:
            triplets: (#rel, 3) of (sub_label, pred_label, ob_label)
            triplets_boxes: (#rel, 2 * 2 * n_dim) array of boxes for the parts
            triplets_scores: (#rel, 3) of (sub_score, pred_score, ob_score)
        """
        sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
        triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
        triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

        if predicate_scores is not None and class_scores is not None:
            triplet_scores = np.column_stack((class_scores[sub_id], predicate_scores, class_scores[ob_id]))
        else:
            triplet_scores = None

        return triplets, triplet_boxes, triplet_scores

    def _compute_pred_matches(
            self,
            gt_rel_repr: np.ndarray,
            pred_rel_repr: np.ndarray,
            gt_boxes: np.ndarray,
            pred_boxes: np.ndarray
    ) -> list[list[int]]:
        """
        Given a set of predicted triplets, returns the list of matching GT for each of the given predictions.
        :param gt_rel_repr: (#gt_rel, N) : any representation of gt relations e.g. (sub_label, pred_label, ob_label)
        :param pred_rel_repr: (#pred_rel, N) any representation of pred relations (must match gt rels)
        :param gt_boxes: (#gt_rel, 2 * 2 * n_dim) : (sub_label, pred_label, ob_label)
        :param pred_boxes: (#pred_rel, 2 * 2 * n_dim) array of boxes for the parts
        :returns: a list for each predicted triplet containing the index of all matched gt triplets.
        """
        is_phrdet = self._mode == SGGEvaluationMode.PhraseDetection
        iou_threshold = self._meta.iou_thr

        # Note: this is why the actual relation representation does not matter
        #       this function can match basically any 2 things having the same format
        keeps = intersect_2d(gt_rel_repr, pred_rel_repr)
        gt_has_match = keeps.any(1)

        n_dim = pred_boxes.shape[1] // 4

        pred_to_gt: list[list[int]] = [[] for _ in range(pred_boxes.shape[0])]
        # For each gt triplet that has at least a match
        for gt_ind, gt_box, keep_indexes in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match]):
            boxes = pred_boxes[keep_indexes]
            if is_phrdet:
                # Evaluate where the union box >= 0.5
                gt_box_union = gt_box.reshape((2, 2 * n_dim))
                gt_box_union = np.concatenate((gt_box_union.min(0)[:n_dim], gt_box_union.max(0)[n_dim:]), 0)

                pred_boxes_union = boxes.reshape((-1, 2, 2 * n_dim))
                pred_boxes_union = np.concatenate((pred_boxes_union.min(1)[:, :n_dim],
                                                   pred_boxes_union.max(1)[:, n_dim:]), 1)

                overlap_mask = bboxes_iou_from_np(gt_box_union[None], pred_boxes_union)[0] >= .5
            else:
                # Individual box IoU thresholding
                sub_iou = bboxes_iou_from_np(gt_box[None, :2 * n_dim], boxes[:, :2 * n_dim])[0]
                obj_iou = bboxes_iou_from_np(gt_box[None, 2 * n_dim:], boxes[:, 2 * n_dim:])[0]

                overlap_mask = (sub_iou >= iou_threshold) & (obj_iou >= iou_threshold)

            # For each predicted triplet that matched with this gt triplet, mark the gt triplet as a match
            for i in np.where(keep_indexes)[0][overlap_mask]:
                pred_to_gt[i].append(int(gt_ind))
        return pred_to_gt


class BaseUpperBoundEvaluator(BaseEvaluator, ABC):
    """
    Base class for all upper bound evaluators.
    If an object detector is trained for relation detection, then it can be important to know the best possible
    performance that can be achieved, given a "perfect" relation detector
    Indeed, if a box is not detected, no related relations can be detected
    Note: this computation assumes that the relation detector can perfectly reclassify boxes and reorder predictions
    Note: this code requires the groundtruth annotation to contain the RELATIONS field.
    Note: used to contain a legacy method.
    """

    # def _compute_box_matches(
    #         self,
    #         gt_boxes: np.ndarray,
    #         pred_boxes: np.ndarray
    # ) -> list[list[int]]:
    #     """
    #     Matching like _compute_pred_matches, but completely ignoring labels.
    #     This is done to evaluate the upper bound of metrics (assuming that the relation predictor is perfect).
    #     I.e. what can we achieve with the given object detector?
    #     Note: we have to assume that all relevant predicted triplets have been supplied. Fix me later?
    #     :param gt_boxes: (#gt_rel, 2 * 2 * n_dim) : (sub_label, pred_label, ob_label)
    #     :param pred_boxes: (#pred_rel, 2 * 2 * n_dim) array of boxes for the parts
    #     :returns: a list for each predicted triplet containing the index of all matched gt triplets.
    #     """
    #     is_phrdet = self._mode == SGGEvaluationMode.PhraseDetection
    #     iou_threshold = self._meta.iou_thr
    #
    #     n_dim = pred_boxes.shape[1] // 4
    #     pred_to_gt: list[list[int]] = [[] for _ in range(pred_boxes.shape[0])]
    #
    #     if is_phrdet:
    #         # Evaluate where the union box >= 0.5
    #         # Compute union box for each gt/pred triplet
    #         pred_boxes = pred_boxes.reshape((-1, 2, 2 * n_dim))
    #         pred_boxes_union = np.concatenate((pred_boxes.min(1)[:, :n_dim], pred_boxes.max(1)[:, n_dim:]), 1)
    #         gt_boxes_union = gt_boxes.reshape((-1, 2, 2 * n_dim))
    #         gt_boxes_union = np.concatenate((gt_boxes_union.min(0)[:n_dim], gt_boxes_union.max(0)[n_dim:]), 0)
    #         overlap_mask = bboxes_iou_from_np(pred_boxes_union, gt_boxes_union) >= .5
    #     else:
    #         # For each gt triplet that has at least a match
    #         sub_iou = bboxes_iou_from_np(gt_boxes[:, :2 * n_dim], pred_boxes[:, :2 * n_dim])
    #         obj_iou = bboxes_iou_from_np(gt_boxes[:, 2 * n_dim:], pred_boxes[:, 2 * n_dim:])
    #         overlap_mask = (sub_iou >= iou_threshold) & (obj_iou >= iou_threshold)
    #
    #     # Return matches where True
    #     # noinspection PyArgumentList
    #     for gt_idx, pred_idx in zip(*np.where(overlap_mask)):
    #         pred_to_gt[pred_idx].append(gt_idx)
    #     return pred_to_gt
