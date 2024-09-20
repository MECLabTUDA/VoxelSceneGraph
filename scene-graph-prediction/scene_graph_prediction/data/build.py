# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
from pathlib import Path
from typing import Sequence, Generator

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from yacs.config import CfgNode

from scene_graph_prediction.utils.comm import get_world_size
from .collate_batch import BatchCollator
from .datasets import DatasetStatistics, Dataset, Split, SGGEvaluableDataset, ConcatDataset
from .paths_catalog import DatasetCatalog
from .samplers import DistributedSampler, GroupedBatchSampler, IterationBasedBatchSampler, \
    KnowledgeGuidedObjectSampler, KnowledgeGuidedRelationSampler
from .transforms import build_transforms, Compose, ToTensor
from .utils import save_labels


def get_dataset_statistics(cfg: CfgNode) -> DatasetStatistics:
    """
    Get dataset statistics (e.g., frequency bias) from training data.
    Will be called to help construct FrequencyBias module.
    """
    logger = logging.getLogger(__name__)
    logger.info("-" * 100)
    logger.info("Get dataset statistics...")

    save_dir = Path(DatasetCatalog.CACHE_DIR)
    save_dir.mkdir(exist_ok=True)

    # Load statistics from the specified file
    if cfg.DATASETS.STATISTICS_OVERRIDE != "":
        save_file = save_dir / f"{cfg.DATASETS.STATISTICS_OVERRIDE}.cache"
        if not save_file.is_file():
            raise RuntimeError(f"Dataset statistics file ({save_file.as_posix()}) does not exist."
                               f"Either compute it or change cfg.DATASETS.STATISTICS_OVERRIDE.")
        logger.info("Loading data statistics from: " + save_file.as_posix())
        logger.info("-" * 100)
        return torch.load(save_file, map_location=torch.device("cpu"), weights_only=True)

    dataset_names = cfg.DATASETS.TRAIN
    assert dataset_names

    # Otherwise, we're using the training data

    # Note: we might have a ConcatDataset
    data_statistics_name = "".join(dataset_names) + "_statistics"
    save_file = save_dir / f"{data_statistics_name}.cache"

    if save_file.is_file():
        logger.info("Loading data statistics from: " + save_file.as_posix())
        logger.info("-" * 100)
        return torch.load(save_file, map_location=torch.device("cpu"), weights_only=True)

    logger.info("Unable to load data statistics from: " + save_file.as_posix())
    # Note: this can handle ConcatDataset cases
    datasets = list(_build_dataset(cfg, dataset_names, None, Split.TRAIN))
    assert len(datasets) == 1 and isinstance(datasets[0], SGGEvaluableDataset)
    # noinspection PyUnresolvedReferences
    statistics = datasets[0].get_statistics()

    logger.info("Save data statistics to: " + save_file.as_posix())
    logger.info("-" * 100)
    torch.save(statistics, save_file)

    return statistics


def build_train_data_loader(
        cfg: CfgNode,
        is_distributed: bool = False,
        start_iter: int = 0
) -> DataLoader:
    """Return a single dataloader for the training data."""
    # For training, concatenate all datasets into a single one
    # Always return a list of Dataset to have a consistent return type
    dataloaders = list(_build_data_loader(cfg, Split.TRAIN, is_distributed, start_iter))
    assert len(dataloaders) == 1
    # Save category_id to label name mapping
    save_labels([dataloaders[0].dataset], cfg.OUTPUT_DIR)
    return dataloaders[0]


def build_val_data_loaders(
        cfg: CfgNode,
        is_distributed: bool = False,
        start_iter: int = 0
) -> list[DataLoader]:
    """Return a list of dataloaders for the validation data."""
    return list(_build_data_loader(cfg, Split.VAL, is_distributed, start_iter))


def build_test_data_loaders(
        cfg: CfgNode,
        is_distributed: bool = False,
        start_iter: int = 0
) -> Generator[DataLoader, None, None]:
    """
    Return a generator of dataloaders for the test data.
    Compared to validation, we usually have significantly more data and we only need to iterate once.
    """
    yield from _build_data_loader(cfg, Split.TEST, is_distributed, start_iter)


def _build_data_loader(
        cfg: CfgNode,
        split: Split = Split.TRAIN,
        is_distributed: bool = False,
        start_iter: int = 0
) -> Generator[DataLoader, None, None]:
    # This variable enables running a test on any data split,
    # even on the training dataset without actually flagging it for training....
    num_gpus = get_world_size()
    is_train = split == Split.TRAIN
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            f"SOLVER.IMS_PER_BATCH ({images_per_batch}) must be divisible by the number of GPUs ({num_gpus}) used."
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            f"TEST.IMS_PER_BATCH ({images_per_batch}) must be divisible by the number of GPUs ({num_gpus}) used."
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False
        num_iters = None
        start_iter = 0

    # if images_per_gpu > 1:
    #     logger = logging.getLogger(__name__)
    #     logger.warning(
    #         "When using more than one image per GPU you may encounter "
    #         "an out-of-memory (OOM) error if your GPU does not have "
    #         "sufficient memory. If this happens, you can reduce "
    #         "SOLVER.IMS_PER_BATCH (for training) or "
    #         "TEST.IMS_PER_BATCH (for inference). For training, you must "
    #         "also adjust the learning rate and schedule length according "
    #         "to the linear scaling rule. See for example: "
    #         "https://github.com/facebookresearch/Detectron/blob/master/"
    #         "configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
    #     )

    if split == Split.TRAIN:
        dataset_list = cfg.DATASETS.TRAIN
    elif split == Split.VAL:
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None.
    # The default will just transform the image to tensor.
    # Other transformations will be applied later (e.g. engine/bbox_aug/_im_detect_bbox or _im_detect_bbox_h_flip)
    transforms = build_transforms(cfg, is_train)

    for dataset in _build_dataset(cfg, dataset_list, transforms, split):
        batch_sampler = _build_batch_data_sampler(
            cfg=cfg,
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            aspect_grouping=[1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else [],
            images_per_batch=images_per_gpu,
            num_iters=num_iters,
            start_iter=start_iter
        )
        # FIXME need a proper interface
        if hasattr(batch_sampler, "batch_collation"):
            collator = batch_sampler.batch_collation
        else:
            collator = BatchCollator(dataset.n_dim, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)
        yield DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )


def _build_dataset(
        cfg: CfgNode,
        dataset_list: Sequence[str],
        transforms: Compose | None = None,
        split: Split = Split.TRAIN
) -> Generator[Dataset, None, None]:
    """
    :param cfg:
    :param dataset_list: Contains the names of the datasets, i.e., coco_2014_train, coco_2014_val, etc.
    :param transforms: transforms to apply to each (image, target) sample
    :returns: During training, returns a list with only one dataset (ConcatDataset created if needed).
              During testing, returns a list with all test datasets.
    """
    if not isinstance(dataset_list, Sequence):
        raise RuntimeError(f"dataset_list should be a list of strings, got {dataset_list}")

    # If no transformation is supplied, we at least need to convert to tensor
    if transforms is None:
        transforms = Compose([ToTensor()])

    if split == Split.TRAIN and len(dataset_list) > 1:
        # Create a concat dataset instead
        # noinspection PyTypeChecker
        yield ConcatDataset([DatasetCatalog.get(dataset_name, cfg, transforms, split) for dataset_name in dataset_list])
    else:
        for dataset_name in dataset_list:
            yield DatasetCatalog.get(dataset_name, cfg, transforms, split)


def _quantize(x: list[float], bins: Sequence) -> list[int]:
    # From looking at the rest of the code, bins is either [] or [1]
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset: Dataset) -> list[float]:
    assert dataset.n_dim == 2, f"Not implemented for {dataset.n_dim}D."
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _build_batch_data_sampler(
        cfg: CfgNode,
        dataset: Dataset,
        shuffle: bool,
        distributed: bool,
        aspect_grouping: Sequence[int],
        images_per_batch: int,
        num_iters: int | None = None,
        start_iter: int = 0
) -> torch.utils.data.sampler.BatchSampler:
    # Check for incompatible combination
    n_sampling_options = sum([
        cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        cfg.DATALOADER.KNOWLEDGE_GUIDED_BOX_GROUPING,
        cfg.DATALOADER.KNOWLEDGE_GUIDED_RELATION_GROUPING,
    ])
    if n_sampling_options > 1:
        raise ValueError(f"At most one sampling method can be selected at a time. "
                         f"{n_sampling_options} are currently selected")

    if (distributed and
            (cfg.DATALOADER.KNOWLEDGE_GUIDED_BOX_GROUPING or
             cfg.DATALOADER.KNOWLEDGE_GUIDED_RELATION_GROUPING)):
        raise ValueError("Knowledge-guided samplers cannot be used with distributed training.")

    # Use the correct combination of samplers
    if n_sampling_options == 0 or cfg.DATALOADER.ASPECT_RATIO_GROUPING or not shuffle:
        # Either no grouping, or ASPECT_RATIO_GROUPING, or we're actually testing (i.e. no shuffle)
        # First prepare the sampler
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        # From looking at the rest of the code, aspect_grouping is either [] or [1]
        if aspect_grouping:
            # Only implemented for 2D
            aspect_ratios = _compute_aspect_ratios(dataset)
            group_ids = _quantize(aspect_ratios, aspect_grouping)
            batch_sampler = GroupedBatchSampler(sampler, group_ids, images_per_batch, drop_incomplete_batches=False)
        else:
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_batch, drop_last=False)
        if num_iters is not None:
            return IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        return batch_sampler

    # Knowledge-guided samplers do most of this already
    if cfg.DATALOADER.KNOWLEDGE_GUIDED_BOX_GROUPING:
        return KnowledgeGuidedObjectSampler(
            cfg,
            images_per_batch,
            cfg.DATALOADER.ITER_PER_GROUP,
            num_iters,
            start_iter,
            cfg.DATALOADER.STRICT_SAMPLING
        )
    return KnowledgeGuidedRelationSampler(
        cfg,
        images_per_batch,
        cfg.DATALOADER.ITER_PER_GROUP,
        num_iters,
        start_iter,
        cfg.DATALOADER.STRICT_SAMPLING
    )
