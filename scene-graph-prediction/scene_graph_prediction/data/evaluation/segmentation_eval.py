import json
import logging
from pathlib import Path

import numpy as np
import torch
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.utils.pathing import remove_suffixes
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, FieldExtractor
from .utils import ConfMatrix, IouType
from ..datasets import COCOEvaluableDataset


def segmentation_eval(
        cfg: CfgNode,
        dataset: COCOEvaluableDataset,
        predictions: dict[int, BoxList],
        output_folder: str | None,
        logger: logging.Logger
) -> dict[IouType, dict[str, dict[str, float]]]:
    """Evaluate semantic segmentation."""

    if output_folder is not None:
        output_folder = Path(output_folder) / "seg"
        output_folder.mkdir(exist_ok=True)

    logger.debug("Preparing results for COCO format")
    overall_matrix = ConfMatrix(cfg.INPUT.N_OBJ_CLASSES)
    patient_matrix = ConfMatrix(cfg.INPUT.N_OBJ_CLASSES)

    per_patient_dices = []
    per_patient_metrics = {}

    for image_id in range(len(dataset)):
        prediction = predictions[image_id]
        groundtruth = dataset.get_groundtruth(image_id)

        # Get formatted masks
        pred_seg = FieldExtractor.segmentation(prediction)
        gt_seg = _prepare_groundtruth(groundtruth, pred_seg.shape)

        # Compute Dice for this image
        patient_matrix.init_zero()
        patient_matrix.add_prediction(gt_seg, pred_seg)

        patient_class_dices = patient_matrix.get_class_dice()
        per_patient_dices.append(patient_class_dices)

        # Add it to the overall cf
        overall_matrix += patient_matrix

        path = dataset.get_img_info(image_id)["file_path"]
        filename = remove_suffixes(Path(path))

        # Save some metrics
        # noinspection PyUnresolvedReferences
        per_patient_metrics[filename] = {idx: dice.item() for idx, dice in enumerate(patient_class_dices)}

        if output_folder is not None and cfg.TEST.SAVE_SEGMENTATIONS:
            # Save segmentation and associated groundtruth
            if groundtruth.has_field(BoxList.AnnotationField.AFFINE_MATRIX):
                affine = groundtruth.AFFINE_MATRIX.cpu().numpy()
            else:
                affine = np.eye(4)
            NiftiImageWrapper.from_array(pred_seg.numpy(), affine, None, True).save(
                output_folder / f"{filename}_pred.nii.gz")
            NiftiImageWrapper.from_array(gt_seg.numpy(), affine, None, True).save(
                output_folder / f"{filename}_gt.nii.gz")

    # Log and compute metrics over all voxels
    class_dice = overall_matrix.get_class_dice()

    logger.info(f"Segmentation metrics overall:")
    per_class_string = ", ".join(f"{e.cpu().item():.3f}" for e in class_dice)
    logger.info(f"Semantic segmentation Dice / class: {per_class_string}")
    logger.info(f"Semantic segmentation Dice FG avg:  {torch.mean(class_dice[1:]).cpu().item():.3f}")

    # Build results dict
    segmentation_overall = {}
    for i, e in enumerate(class_dice):
        # noinspection PyUnresolvedReferences
        segmentation_overall[f"meanDice_class_{i}"] = e.cpu().item()
    segmentation_overall[f"meanDice_fg"] = torch.mean(class_dice[1:]).cpu().item()

    # Log and compute metrics for each patient and average
    per_patient_dices = torch.stack(per_patient_dices)
    mean_dices = torch.mean(per_patient_dices, dim=0)
    std_dices = torch.std(per_patient_dices, dim=0)

    logger.info(f"Segmentation metrics per patient:")
    per_class_string = ", ".join(f"{e.cpu().item():.3f}±{f.cpu().item():.3f}" for e, f in zip(mean_dices, std_dices))
    logger.info(f"Semantic segmentation mean Dice / class: {per_class_string}")
    logger.info(f"Semantic segmentation mean Dice FG avg:  "
                f"{torch.mean(mean_dices[1:]).cpu().item():.3f}±{torch.mean(std_dices[1:]).cpu().item():.3f}")

    # Build results dict
    segmentation_per_pat = {}
    for i, e in enumerate(mean_dices):
        # noinspection PyUnresolvedReferences
        segmentation_per_pat[f"meanDice_class_{i}"] = e.cpu().item()
    segmentation_per_pat[f"meanDice_fg"] = torch.mean(mean_dices[1:]).cpu().item()

    # Store some metrics as JSON
    if output_folder is not None:
        metrics = {
            "overall": {
                "per_class_voxel_mean_dice": [e.cpu().item() for e in class_dice],
                "voxel_mean_fg_iou": torch.mean(class_dice[1:]).cpu().item(),
                "per_class_patient_mean_dice": [e.cpu().item() for e in mean_dices],
                "per_class_patient_std_dice": [e.cpu().item() for e in std_dices],
                "patient_mean_fg_dice": torch.mean(mean_dices[1:]).cpu().item(),
                "patient_std_fg_dice": torch.mean(std_dices[1:]).cpu().item(),
            },
            "per_patient": per_patient_metrics
        }
        with open(output_folder / f"metrics.json", "w") as f:
            json.dump(metrics, f)

    return {
        IouType.Segmentation: {
            "Segmentation_overall": segmentation_overall,
            "Segmentation_per_pat": segmentation_per_pat
        }
    }


def _prepare_groundtruth(groundtruth: BoxList, target_shape: tuple[int, ...]) -> torch.LongTensor:
    seg = FieldExtractor.segmentation(groundtruth)
    # Need to zero-pad
    pad_img = torch.zeros(target_shape).to(seg)
    slicer = tuple(slice(0, seg.shape[i]) for i in range(len(seg.shape)))
    pad_img[slicer].copy_(seg)

    return pad_img
