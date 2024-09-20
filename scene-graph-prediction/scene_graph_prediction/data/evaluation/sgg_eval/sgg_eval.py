import logging
from collections import defaultdict
from typing import TypedDict

import numpy as np
from scene_graph_api.utils.tensor import relation_matrix_to_triplets
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .evaluators import *
from ..utils import SGGEvaluationMode, IouType
from ...datasets import COCOEvaluableDataset, SGGEvaluableDataset

SGG_EVAL_RESULTS_FILE = "eval_results.pth"
SGG_EVAL_RESULTS_DICT_FILE = "result_dict.pth"


class SGGEvalResults(TypedDict):
    """Content of what is stored in SGG_EVAL_RESULTS_FILE."""
    groundtruths: list[BoxList]
    predictions: list[BoxList]


class RelationsModeEvaluator:
    """Class used to instantiate the appropriate evaluators and wrap them for convenience."""

    def __init__(self, mode: SGGEvaluationMode, meta: MetaContext):
        # Select the appropriate metrics
        if meta.compute_upper_bound:
            self.evaluators = [SGRecallUpperBoundEvaluator(mode, meta)]
            if meta.n_rel_classes - 1 > 1:  # Exclude de background
                self.evaluators.append(SGMeanRecallUpperBoundEvaluator(mode, meta))
            return

        # Usual SGG metrics
        self.evaluators = [
            SGRecallEvaluator(mode, meta),
            # SGNoGraphConstraintRecallEvaluator(mode, meta),
            SGMeanAveragePrecisionEvaluator(mode, meta)
        ]

        if meta.n_rel_classes - 1 > 1:  # Exclude de background
            self.evaluators.append(SGMeanRecallEvaluator(mode, meta))
            # self.evaluators.append(SGNoGraphMeanRecallEvaluator(mode, meta))

        if mode in [SGGEvaluationMode.PredicateClassification, SGGEvaluationMode.SceneGraphClassification]:
            self.evaluators.append(SGPairAccuracyEvaluator(mode, meta))
            if meta.n_rel_classes - 1 > 1:  # Exclude de background
                self.evaluators.append(SGPairMeanAccuracyEvaluator(mode, meta))

    def evaluate(self, image_context: ImageContext, results: SGGResults):
        for evaluator in self.evaluators:
            evaluator.evaluate(image_context, results)

    def aggregate(self, results: SGGResults):
        for evaluator in self.evaluators:
            evaluator.aggregate(results)

    def results_to_str(self, results: SGGResults) -> str:
        result_str = ""
        for evaluator in self.evaluators:
            result_str += evaluator.summarize(results)
        return result_str


def sgg_evaluation(
        cfg: CfgNode,
        dataset: COCOEvaluableDataset | SGGEvaluableDataset,
        predictions: dict[int, BoxList],
        output_folder: str,
        logger: logging.Logger
) -> dict[IouType, dict[str, dict[str, float]]]:
    # Extract evaluation settings from cfg
    mode = SGGEvaluationMode.build(cfg)

    groundtruths = {image_id: dataset.get_groundtruth(image_id) for image_id in range(len(predictions))}

    result_str = "\n" + "=" * 100 + "\n"

    # Evaluate relations
    meta = MetaContext(
        mode=mode,
        n_rel_classes=len(dataset.predicates),
        compute_upper_bound=cfg.TEST.RELATION.COMPUTE_RELATION_UPPER_BOUND,
        obj_class_agnostic_upper_bound=cfg.TEST.RELATION.UPPER_BOUND_ALLOW_RECLASSIFICATION,
        iou_thr=cfg.TEST.RELATION.IOU_THRESHOLD,
        ks=cfg.TEST.RELATION.RECALL_AT_K,
        no_gc_top_n_pred=cfg.TEST.RELATION.NO_GRAPH_CONSTRAINT_TOP_N_RELATION
    )

    evaluator = RelationsModeEvaluator(mode, meta)
    result_dict: SGGResults = defaultdict(lambda: defaultdict(list))
    for image_id in groundtruths:
        # Note: we assume that the relation predictions are already sorted
        _evaluate_relation_of_one_image(
            groundtruths[image_id],
            predictions[image_id],
            result_dict,
            evaluator,
            mode,
            meta.compute_upper_bound
        )
    # Now that all images have been evaluated, we can compute aggregates
    # E.g. once we have the recall for each image, we want to store the mean across images
    evaluator.aggregate(result_dict)

    # Print result
    result_str += evaluator.results_to_str(result_dict)
    logger.info(result_str)

    return {
        IouType.Relations: {
            "Overall": {
                f"{metric_name}@{k}": float(np.mean(per_img_res))  # Linter going haywire without the float cast
                for metric_name, res in result_dict.items()
                for k, per_img_res in res.items()
            }
        }
    }


def _evaluate_relation_of_one_image(
        groundtruth: BoxList,
        prediction: BoxList,
        results: dict,
        evaluator: RelationsModeEvaluator,
        mode: SGGEvaluationMode,  # Only used for sanity checks
        compute_upper_bound: bool
):
    gt_rel_matrix = groundtruth.RELATIONS.long().detach().cpu()
    gt_rels = relation_matrix_to_triplets(gt_rel_matrix).numpy()

    # Note: we cannot skip an image that has no relations for mAP computation
    # if len(gt_rels) == 0:
    #     return

    if compute_upper_bound:
        # Since we're computing upper bounds, we only have predicted boxes (no relations)
        pred_rel_idxs = None
        pred_rel_scores = None
        pred_rel_labels = None
    else:
        pred_rel_idxs = prediction.REL_PAIR_IDXS.long().detach().cpu().numpy()
        # Note: PRED_REL_LABELS is not used as it is usually computed on the fly as argmax of the scores...
        # pred_rel_labels = prediction.get_field(BoxList.PredictionField.PRED_REL_LABELS).long().detach().cpu().numpy()
        if pred_rel_idxs.shape[0] == 0:
            return
        pred_rel_scores = prediction.PRED_REL_CLS_SCORES.detach().cpu().numpy()
        pred_rel_labels = prediction.PRED_REL_LABELS.detach().cpu().numpy()

    image_context = ImageContext(
        gt_boxes=groundtruth.convert(BoxList.Mode.zyxzyx).boxes.detach().cpu().numpy(),
        gt_labels=groundtruth.LABELS.long().detach().cpu().numpy(),
        gt_rels=gt_rels,
        pred_boxes=prediction.convert(BoxList.Mode.zyxzyx).boxes.detach().cpu().numpy(),
        pred_labels=prediction.PRED_LABELS.long().detach().cpu().numpy(),
        pred_obj_scores=prediction.PRED_SCORES.detach().cpu().numpy(),
        pred_rel_idxs=pred_rel_idxs,
        pred_rel_scores=pred_rel_scores,
        pred_rel_labels=pred_rel_labels
    )

    # Some sanity check
    if not compute_upper_bound:
        if mode == SGGEvaluationMode.PredicateClassification:
            assert image_context.gt_boxes.shape == image_context.pred_boxes.shape
            # Check that we have the same boxes+labels and all obj class-wise scores are 0 or 1
            assert image_context.gt_boxes.shape == image_context.pred_boxes.shape
            assert np.alltrue(image_context.gt_labels == image_context.pred_labels)
            assert np.allclose(image_context.pred_obj_scores, 1)
        elif mode == SGGEvaluationMode.SceneGraphClassification:
            # Check that we have the same boxes
            assert image_context.gt_boxes.shape == image_context.pred_boxes.shape

    evaluator.evaluate(image_context, results)
