# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from scene_graph_prediction.data import Dataset
from scene_graph_prediction.modeling.abstractions.detector import AbstractDetector
from scene_graph_prediction.structures import BoxList, ImageList
from scene_graph_prediction.utils.logger import setup_logger
from .bbox_aug import im_detect_bbox_aug
from ..modeling.abstractions.loss import LossDict
from ..utils.comm import all_gather, is_main_process, get_world_size, synchronize, reduce_dict
from ..utils.timer import Timer


def inference(
        cfg: CfgNode,
        model: AbstractDetector,
        data_loader: DataLoader,
        dataset_name: str,
        compute_loss: bool = False,
        device: torch.device = "cuda",
        logger: logging.Logger | None = None
) -> tuple[dict[int, BoxList], LossDict] | None:
    """Perform inference for a full dataset. Predictions are ordered by their id in the dataset."""
    num_devices = get_world_size()

    if logger is None:
        # Output directory won't be set up, but at least we have some logging
        logger = setup_logger("scene_graph_prediction.inference", save_dir="", distributed_rank=1)

    dataset: Dataset = data_loader.dataset
    # noinspection PyTypeChecker
    logger.info(f"Start evaluation on {dataset_name} dataset({len(dataset)} images).\n")
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    preds_per_gpu, dataset_losses = _compute_on_dataset(
        cfg, model, data_loader, compute_loss, device, cfg.TEST.RELATION.SYNC_GATHER, inference_timer
    )
    predictions = _reduce_predictions_from_multiple_gpus(preds_per_gpu, cfg.TEST.RELATION.SYNC_GATHER)

    # Wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = Timer.timedelta_str(total_time)
    device_str = "device" if num_devices == 1 else "devices"
    logger.info(
        f"Total run time: {total_time_str} ({total_time * num_devices / len(dataset)} s / img per device, "
        f"on {num_devices} {device_str})"
    )
    total_infer_time = Timer.timedelta_str(inference_timer.total_time)
    logger.info(
        f"Model inference time: {total_infer_time} "
        f"({inference_timer.total_time * num_devices / len(dataset)} s / img per device, "
        f"on {num_devices} {device_str})\n"
    )

    if is_main_process():
        return predictions, dataset_losses


def _compute_on_dataset(
        cfg: CfgNode,
        model: AbstractDetector,
        data_loader: DataLoader,
        compute_loss: bool,
        device: torch.device,
        synchronize_gather: bool = True,
        timer: Timer | None = None
) -> tuple[dict[int, BoxList], LossDict]:
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    all_losses = defaultdict(list)
    for batch in tqdm(data_loader):
        with torch.no_grad():
            images, targets, image_ids = batch  # type: ImageList, tuple[BoxList, ...], tuple[int, ...]
            images = ImageList.to_image_list(images, cfg.INPUT.N_DIM, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)
            targets: list[BoxList] = [target.to(device) for target in targets]
            if timer:
                timer.tic()

            if cfg.TEST.BBOX_AUG.ENABLED:
                assert not cfg.MODEL.RELATION_ON
                # Note: if cfg.TEST.BBOX_AUG.ENABLED then we're doing object detection
                #       in this case, targets are not required but are passed for easier debugging
                output, loss_dict = im_detect_bbox_aug(model, images, compute_loss, device, targets)
            else:
                # Note: this branch can both handle object detection and relation prediction
                #       only the latter requires targets
                output, loss_dict = model(images.to(device), targets, compute_loss)

            if timer:
                timer.toc()
            output = [o.to(cpu_device) for o in output]
            if cfg.MODEL.DEVICE != 'cpu':
                torch.cuda.synchronize()

        # Gather outputs
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})
        # Aggregate losses
        loss_dict_reduced = {
            k: v
            for k, v in reduce_dict(loss_dict, average=True).items()
            if not k.startswith("_") and v.numel() > 0
        }
        for k in loss_dict_reduced:
            all_losses[k].append(loss_dict_reduced[k].cpu().item())
        losses_reduced = sum(loss.cpu().item() for loss in loss_dict_reduced.values())
        all_losses["loss"].append(losses_reduced)

    torch.cuda.empty_cache()
    # Losses are averaged over batches
    return results_dict, {k: np.mean(all_losses[k]) for k in all_losses}


def _reduce_predictions_from_multiple_gpus(
        predictions_per_gpu: dict[int, BoxList] | list[dict[int, BoxList]],
        synchronize_gather: bool = True
) -> dict[int, BoxList] | None:
    """
    Note: this function does the sync gather at the end of the prediction for the whole dataset
          (instead of after each batch), if synchronize_gather is False.
    """
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # Merge the list of dicts
        predictions = {}
        for p in all_gather(predictions_per_gpu):
            predictions.update(p)

    # Convert a dict where the key is the index in a list
    if len(predictions) > 0 and len(predictions) - 1 != max(predictions.keys()):
        logger = setup_logger("scene_graph_prediction.inference", "", distributed_rank=1)
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation."
        )

    # Convert to a list
    return predictions
