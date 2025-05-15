import logging

import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .utils import IouType
from ..datasets.Dataset import COCOEvaluableDataset


def non_unique_stats_eval(
        cfg: CfgNode,
        dataset: COCOEvaluableDataset,
        predictions: dict[int, BoxList],
        logger: logging.Logger
) -> dict[IouType, dict[str, dict[str, float]]]:
    # Weird circular import...
    from scene_graph_prediction.modeling.utils.label_assignment import assign_label_to_proposals_always_match_special

    num_unique_fg_classes = cfg.INPUT.N_UNIQUE_OBJ_CLASSES
    num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - num_unique_fg_classes - 1

    # We're in the prediction part now, so we cannot assume that unique classes are last...
    # So we have to force remap ids
    proposals = []
    targets = []
    for idx, proposal in predictions.items():
        target = dataset.get_groundtruth(idx)
        proposal = dataset.reindex_prediction(
            proposal.copy_with_fields([BoxList.PredictionField.PRED_LABELS]),
            dataset.json_category_id_to_contiguous_id
        )
        target = dataset.reindex_groundtruth(target.copy_with_fields([BoxList.AnnotationField.LABELS]))
        proposals.append(proposal[proposal.PRED_LABELS <= num_normal_fg_classes])
        targets.append(target[target.LABELS <= num_normal_fg_classes])

    assign_label_to_proposals_always_match_special(
        proposals, targets, cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, num_normal_fg_classes
    )

    # noinspection PyTypeChecker
    n_fp = [torch.sum(pred.LABELS == 0).cpu().item() for pred in proposals]
    # noinspection PyTypeChecker
    ratio_fp = [torch.sum(pred.LABELS == 0).cpu().item() / (len(pred) + 1e-5) for pred in proposals]
    n_tp = [torch.sum(pred.LABELS > 0).cpu().item() for pred in proposals]
    ratio_tp = [torch.sum(pred.LABELS > 0).cpu().item() / (len(pred) + 1e-5) for pred in proposals]
    n_gt = [len(tgt) for tgt in targets]
    recall = [torch.unique(pred.MATCHED_IDXS[pred.MATCHED_IDXS >= 0]).numel() / len(tgt) for pred, tgt in
              zip(proposals, targets)]
    precision = [
        torch.unique(pred.MATCHED_IDXS[pred.MATCHED_IDXS >= 0]).numel() / (len(pred) + 1e-5)
        for pred in proposals
    ]
    f1_score = [2 / (1 / (prec + 1e-5) + 1 / (rec + 1e-5)) for prec, rec in zip(precision, precision)]

    logger.info(
        f"Box detection metrics:\n"
        f"#FP: {np.mean(n_fp):.3f}+-{np.std(n_fp):.3f}  "
        f"%FP: {np.mean(ratio_fp):.3f}+-{np.std(ratio_fp):.3f}  "
        f"#TP: {np.mean(n_tp):.3f}+-{np.std(n_tp):.3f}  "
        f"%TP: {np.mean(ratio_tp):.3f}+-{np.std(ratio_tp):.3f}  "
        f"#GT: {np.mean(n_gt):.3f}+-{np.std(n_gt):.3f}  "
        f"  R: {np.mean(recall):.3f}+-{np.std(recall):.3f}  "
        f" F1: {np.mean(f1_score):.3f}+-{np.std(f1_score):.3f}  "
    )

    # noinspection PyTypeChecker
    return {
        IouType.BoundingBox: {
            "NonUnique": {
                "#FP": np.mean(n_fp),
                "%FP": np.mean(ratio_fp),
                "#TP": np.mean(n_tp),
                "%TP": np.mean(ratio_tp),
                "#GT": np.mean(n_gt),
                "  R": np.mean(recall),
                " F1": np.mean(f1_score),
            }
        }
    }
