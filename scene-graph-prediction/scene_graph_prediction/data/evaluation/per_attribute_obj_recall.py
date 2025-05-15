import logging

import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, BoxListOps
from .utils import IouType, COCO_EVALUATION_PARAMETERS
from ..datasets.Dataset import COCOEvaluableDataset
from pycocotools3d import IouType as CocoIouType


def per_attribute_obj_recall(
        cfg: CfgNode,
        dataset: COCOEvaluableDataset,
        predictions: dict[int, BoxList],
        logger: logging.Logger
) -> dict[IouType, dict[str, dict[str, float]]]:
    n_attr = cfg.INPUT.N_ATT_CLASSES
    if n_attr == 0:
        return {IouType.BoundingBox: {"Overall": {}}}

    iou_thr = cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD
    top_k = COCO_EVALUATION_PARAMETERS[cfg.DATASETS.EVALUATION_PARAMETERS](CocoIouType.BoundingBox).maxDets[-1]

    tgt_attr_cnt = torch.zeros(5, dtype=torch.int64)
    matched_attr_cnt = torch.zeros(5, dtype=torch.int64)

    for idx in range(len(dataset)):
        gt = dataset.get_groundtruth(idx)

        tgt_attr_cnt += gt.ATTRIBUTES.sum(0)
        pred = predictions[idx][:top_k]

        if len(pred) == 0:
            continue

        # Do class-wise offset to only compute class-wise matches
        offset = max(gt.size)
        BoxListOps.offset_classwise(gt, offset, BoxList.AnnotationField.LABELS)
        BoxListOps.offset_classwise(pred, offset, BoxList.PredictionField.PRED_LABELS)

        # Only keep GT with matches
        ious = BoxListOps.iou(gt, pred)
        matched = ious.max(1)[0] > iou_thr
        matched_attr_cnt += gt[matched].ATTRIBUTES.sum(0)

        # We need to offset back, because we're changing the boxes
        BoxListOps.offset_classwise(gt, -offset, BoxList.AnnotationField.LABELS)
        BoxListOps.offset_classwise(pred, -offset, BoxList.PredictionField.PRED_LABELS)

    dataset_detection_rates = matched_attr_cnt / (tgt_attr_cnt + 1e-9)

    logger.info(f"Per attribute recall: {dataset_detection_rates.tolist()}")

    # noinspection PyTypeChecker
    return {
        IouType.BoundingBox: {
            "Overall": {
                f"R{top_k}_per_attr": dataset_detection_rates.tolist()
            }
        }
    }
