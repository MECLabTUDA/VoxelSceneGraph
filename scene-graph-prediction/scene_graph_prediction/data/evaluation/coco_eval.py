import logging
from collections import defaultdict
from typing import TypedDict

import numpy as np
import pycocotools3d.mask as mask_util
import pycocotools3d.mask3d as mask_util3d
from pycocotools3d.coco import COCO, COCO3d
from pycocotools3d.coco.abstractions.object_detection import MinimalPrediction
from pycocotools3d.cocoeval import COCOeval
from pycocotools3d.params import EvaluationParams
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, AbstractMaskList, BoxListOps
from scene_graph_prediction.utils.miscellaneous import get_pred_masks
from .utils import IouType, COCO_EVALUATION_PARAMETERS
from ..datasets import COCOEvaluableDataset


class _COCOBoundingBoxPred(TypedDict):
    image_id: int
    category_id: int
    bbox: list
    score: float


class _COCOSegmentationPred(TypedDict):
    image_id: int
    category_id: int
    segmentation: mask_util.CompressedRLE  # RLE encoded numpy array
    score: float


class _COCOKeypointPred(TypedDict):
    image_id: int
    category_id: int
    keypoints: list
    score: float


def coco_evaluation(
        cfg: CfgNode,
        dataset: COCOEvaluableDataset,
        predictions: dict[int, BoxList],
        output_folder: str | None,
        logger: logging.Logger
) -> tuple[dict[IouType, dict[str, dict[str, float]]], dict]:
    iou_types = IouType.build_iou_types(cfg)

    if IouType.RegionProposal in iou_types:
        assert len(iou_types) == 1, "If RegionProposal in an IouType that we need to evaluate, then RPN_ONLY is True." \
                                    " And any other type of evaluation is incompatible."

    logger.debug("Preparing results for COCO format")
    # Format predictions for each type of desired evaluation
    coco_results = {}
    if IouType.BoundingBox in iou_types:
        logger.debug("Preparing bbox results")
        coco_results[IouType.BoundingBox] = _prepare_for_coco_detection(predictions, dataset)
    if IouType.Segmentation in iou_types:
        logger.debug("Preparing segm results")
        coco_results[IouType.Segmentation] = _prepare_for_coco_segmentation(predictions, dataset)
    if IouType.Keypoints in iou_types:
        logger.debug("Preparing keypoints results")
        coco_results[IouType.Keypoints] = _prepare_for_coco_keypoint(predictions, dataset)
    if IouType.RegionProposal in iou_types:
        # Note: in this case the ground truth supplied by dataset.coco needs to be adapted for binary classification
        logger.debug("Preparing region proposals")
        coco_results[IouType.RegionProposal] = _prepare_for_coco_region_proposal(predictions, dataset)

    results: dict[IouType, dict[str, dict[str, float]]] = defaultdict(dict)
    logger.debug("Evaluating predictions")
    # For each type of evaluation
    for iou_type in iou_types:
        if iou_type == IouType.Relations:
            continue

        # Evaluate
        # noinspection PyTypeChecker
        coco_eval = _evaluate_predictions_on_coco(
            cfg,
            dataset.coco,
            coco_results[iou_type],
            iou_type,
            logger
        )

        logger.info("Metrics overall categories:")
        coco_eval.summarize(omit_missing=True)

        # Save metrics overall
        results[iou_type]["Overall"] = dict(coco_eval.stats)

        # Results per category are not currently saved
        if cfg.TEST.COCO.PER_CATEGORY_METRICS:
            if not coco_eval.params.useCats:
                logger.warning("cfg.TEST.COCO.PER_CATEGORY_METRICS is True, but EvaluationParams.useCats is False: "
                               "per category metrics will be ignored")
            else:
                for cat_id in coco_eval.params.catIds:
                    logger.info(f"Metrics for category {cat_id}:")
                    coco_eval.summarize(cat_id, omit_missing=True)
                    # Save metrics for each class
                    class_name = coco_eval.cocoGt.cats[cat_id]["name"]
                    results[iou_type][class_name] = dict(coco_eval.stats)

    return results, coco_results


def _evaluate_predictions_on_coco(
        cfg: CfgNode,
        coco_gt: COCO | COCO3d,
        coco_results: list[MinimalPrediction],
        iou_type: IouType,
        logger: logging.Logger
) -> COCOeval:
    coco_dt = coco_gt.loadRes(coco_results) if coco_results else type(coco_gt)()

    parameters: EvaluationParams = COCO_EVALUATION_PARAMETERS[cfg.DATASETS.EVALUATION_PARAMETERS](iou_type.to_coco())
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type.to_coco(), params=parameters, logger=logger)
    coco_eval.evaluate()
    coco_eval.accumulate()

    return coco_eval


def _prepare_for_coco_detection(
        predictions: dict[int, BoxList],
        dataset: COCOEvaluableDataset
) -> list[_COCOBoundingBoxPred]:
    coco_results = []

    for image_id, prediction in predictions.items():
        # noinspection DuplicatedCode
        original_id = dataset.contiguous_image_id_to_json_id[image_id]
        if len(prediction) == 0:
            continue

        prediction = prediction.convert(BoxList.Mode.zyxdhw)
        boxes: list = prediction.boxes.tolist()
        scores: list = prediction.PRED_SCORES.tolist()
        labels: list = prediction.PRED_LABELS.tolist()

        # Compute the IoU between predictions and groundtruth to more easily identify false positives in the pred file
        target = dataset.get_groundtruth(image_id)
        iou = BoxListOps.iou(prediction, target)
        # Mask out label mismatches
        iou[prediction.pred_labels[:, None] != target.labels[None]] = 0
        best_iou = iou.max(1)[0]

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        coco_results.extend([
            {
                "image_id": original_id,
                "category_id": mapped_labels[k],
                "bbox": box,
                "score": scores[k],
                "iou": best_iou[k].item()
            }
            for k, box in enumerate(boxes)
        ])

    # noinspection PyTypeChecker
    return coco_results


def _prepare_for_coco_segmentation(
        predictions: dict[int, BoxList],
        dataset: COCOEvaluableDataset
) -> list[_COCOSegmentationPred]:
    assert predictions
    n_dim = predictions[0].n_dim

    coco_results = []

    for image_id, prediction in predictions.items():  # type: int, BoxList
        # noinspection DuplicatedCode
        original_id = dataset.contiguous_image_id_to_json_id[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        if n_dim == 3:
            # noinspection PyTypedDict
            image_depth = img_info["depth"]
            lengths = image_depth, image_height, image_width
        else:
            lengths = image_height, image_width
        masks: AbstractMaskList = get_pred_masks(prediction)

        # Masker is necessary only if masks haven't been already resized.
        assert len(lengths) == prediction.n_dim

        scores = prediction.PRED_SCORES.tolist()
        labels = prediction.PRED_LABELS.tolist()

        if n_dim == 2:
            rle_list = [mask_util.encode(np.array(mask[0], order="F")) for mask in masks]
        else:
            rle_list = [mask_util3d.encode(np.array(mask[0], order="F")) for mask in masks]

        for rle in rle_list:
            # noinspection PyTypeChecker,PyTypedDict
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rle_list)
            ]
        )
    return coco_results


def _prepare_for_coco_keypoint(
        predictions: dict[int, BoxList],
        dataset: COCOEvaluableDataset
) -> list[_COCOKeypointPred]:
    assert dataset.n_dim == 2

    coco_results = []
    for image_id, prediction in predictions.items():
        original_id = dataset.contiguous_image_id_to_json_id[image_id]
        if len(prediction.boxes) == 0:
            continue

        prediction = prediction.convert(BoxList.Mode.zyxdhw)

        scores = prediction.PRED_SCORES.tolist()
        labels = prediction.LABELS.tolist()
        keypoints = prediction.KEYPOINTS
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "keypoints": keypoint,
                    "score": scores[k]
                }
                for k, keypoint in enumerate(keypoints)
            ]
        )

    return coco_results


def _prepare_for_coco_region_proposal(
        predictions: dict[int, BoxList],
        dataset: COCOEvaluableDataset
) -> list[_COCOBoundingBoxPred]:
    """Like for object detection but with binary classification (bg vs fg)."""
    coco_results = []
    # FIXME we could actually just use "useCats=0" in pycocotools
    for image_id, prediction in predictions.items():
        # noinspection DuplicatedCode
        original_id = dataset.contiguous_image_id_to_json_id[image_id]
        if len(prediction) == 0:
            continue

        prediction = prediction.convert(BoxList.Mode.zyxdhw)
        boxes: list = prediction.boxes.tolist()
        scores: list = prediction.OBJECTNESS.tolist()

        coco_results.extend(
            [{
                "image_id": original_id,
                "category_id": 1,
                "bbox": box,
                "score": scores[k],
            } for k, box in enumerate(boxes)]
        )

    return coco_results
