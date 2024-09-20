import json
import logging
from pathlib import Path

from scene_graph_api.utils.pathing import remove_suffixes
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList
from .coco_eval import coco_evaluation
from .segmentation_eval import segmentation_eval
from .sgg_eval import sgg_evaluation
from .utils import IouType, SGGEvaluationMode, EvaluationType
from ..datasets import Dataset


def evaluate(
        cfg: CfgNode,
        dataset: Dataset,
        dataset_name: str,
        predictions: dict[int, BoxList],
        evaluation_type: EvaluationType,
        output_folder: str | None,
        logger: logging.Logger
) -> dict[IouType, dict[str, dict[str, float]]]:
    """
    Evaluate dataset using different methods based on dataset type.
    :param cfg:
    :param dataset: Dataset object
    :param dataset_name: Name of the dataset
    :param predictions: dataset idx -> prediction
    :param evaluation_type: flags indicating which type of evaluation needs to be performed.
    :param output_folder: output folder, to save evaluation files or results.
    :param logger: the logger.
    :return: evaluation results as dict[IouType.value, dict["overall" or class name, dict[metric, value]].
    """

    # Start by saving predictions...
    if output_folder and cfg.TEST.SAVE_BOXLISTS:
        folder = Path(output_folder)
        logger.debug(f"Saving predictions to {folder.absolute()}")
        for image_id, prediction in predictions.items():
            img_path = dataset.get_img_info(image_id)["file_path"]
            target = folder / Path(img_path).name
            # Replace the original file extension with ".pth"
            prediction.save(target.with_name(remove_suffixes(target) + ".pth"))

    metrics = {}

    if evaluation_type & EvaluationType.COCO:
        # noinspection PyTypeChecker
        detec_metrics, _ = coco_evaluation(
            cfg=cfg,
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )
        metrics.update(detec_metrics)

    if evaluation_type & EvaluationType.SemanticSegmentation:
        # noinspection PyTypeChecker
        seg_metrics = segmentation_eval(
            cfg=cfg,
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )
        # Note: the key for IouType segmentation might have already been set by COCO, so be careful when updating
        if IouType.Segmentation in metrics:
            metrics[IouType.Segmentation].update(seg_metrics[IouType.Segmentation])
        else:
            metrics.update(seg_metrics)

    if evaluation_type & EvaluationType.SGG:
        # noinspection PyTypeChecker
        sgg_metrics = sgg_evaluation(
            cfg=cfg,
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )
        metrics.update(sgg_metrics)

    # Save metrics as JSON
    if output_folder:
        folder = Path(output_folder)
        logger.debug(f"Saving metrics to {folder.absolute()}")
        with open(folder / f"{dataset_name}.json", "w") as f:
            json.dump({k.value: v for k, v in metrics.items()}, f)

    return metrics


__all__ = ["evaluate", "EvaluationType", "IouType", "SGGEvaluationMode"]
