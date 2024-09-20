import itertools
import logging
from pathlib import Path
from typing import Iterator, Generator
from abc import ABC, abstractmethod
import torch
from torch.utils.data.sampler import BatchSampler, Sampler, RandomSampler
from yacs.config import CfgNode

from scene_graph_api.tensor_structures import BoxList
from scene_graph_prediction.data.paths_catalog import DatasetCatalog
from scene_graph_prediction.data.datasets import COCOEvaluableDataset, Split
import json

from scene_graph_prediction.structures import ImageList


class KnowledgeGuidedAbstractSampler(BatchSampler, ABC):
    """
    Similar to a GroupedBatchSampler fused with a RandomSampler with replacement and with an IterationBasedBatchSampler,
    but we sequentially select a group for a fixed number of iterations and directly sample images
    with objects containing objects in this group.
    If strict_sampling, only the objects (or relations) selected for the group will be used for training.

    Note: these samplers also take care of the batch collation.
    Note: since we need to first compute the set of suitable images for each group, we cache this intermediate result.
    """

    def __init__(
            self,
            cfg: CfgNode,
            batch_size: int,
            n_groups: int,
            iterations_per_group: int,
            start_batch: int,
            num_batches: int,
            strict_sampling: bool = False,
            group_to_ids: dict[int, list[int]] | None = None  # Mostly for testing purposes
    ):
        super().__init__([], batch_size, False)
        self.cfg = cfg
        self.n_groups = n_groups
        self.iterations_per_group = iterations_per_group
        self.start_batch = start_batch
        self.num_batches = num_batches
        self.strict_sampling = strict_sampling

        # Keep a state of what has been sampled
        self.generator = None
        self.current_group = 0

        # Don't try to load anything if the mapping is supplied
        if group_to_ids is not None:
            self.group_to_ids = group_to_ids
            return

        # Check if the group to image mapping is cached, otherwise compute it
        logger = logging.getLogger(__name__)
        logger.info("-" * 100)
        logger.info("Get dataset groups...")

        save_dir = Path(DatasetCatalog.CACHE_DIR)
        save_dir.mkdir(exist_ok=True)

        dataset_names = cfg.DATASETS.TRAIN
        assert dataset_names

        # Note: we might have a ConcatDataset
        data_statistics_name = type(self).__name__ + "_" + "".join(dataset_names) + "_groups"
        save_file = save_dir / f"{data_statistics_name}.cache"

        if save_file.is_file():
            logger.info("Loading data groups from: " + save_file.as_posix())
            logger.info("-" * 100)
            with save_file.open("r") as f:
                # Also need to remap int keys from str. Thx JSON!
                self.group_to_ids = {int(k): v for k, v in json.load(f).items()}
                return

        logger.info("Unable to load data groups from: " + save_file.as_posix())
        # Note: this can handle ConcatDataset cases
        from scene_graph_prediction.data.build import _build_dataset
        datasets = list(_build_dataset(cfg, dataset_names, None, Split.TRAIN))
        assert len(datasets) == 1 and isinstance(datasets[0], COCOEvaluableDataset)
        # noinspection PyTypeChecker
        self.group_to_ids = self._compute_group_ids_mapping(datasets[0])

        logger.info("Save data groups to: " + save_file.as_posix())
        logger.info("-" * 100)

        with save_file.open("w") as f:
            json.dump(self.group_to_ids, f)

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        group_randperm_generators = {
            group_idx: self._infinite_randperm(len(self.group_to_ids[group_idx]), generator)
            for group_idx in self.group_to_ids
        }

        yield [40]
        yield [40]
        yield [40]
        yield [40]
        yield [40]

        batch = self.start_batch
        while True:
            # Check whether we need to update the currently selected group
            for group_idx in range(self.n_groups):
                self.current_group = group_idx
                numel_group = len(self.group_to_ids[group_idx])
                for _ in range(self.iterations_per_group):
                    if batch >= self.num_batches:
                        return

                    # Check whether there is any image for current group
                    if numel_group == 0:
                        yield []
                        batch += 1
                        continue

                    # Sample a random batch with replacement
                    yield [
                        self.group_to_ids[group_idx][next(group_randperm_generators[group_idx])]
                        for _ in range(self.batch_size)
                    ]
                    batch += 1

    @abstractmethod
    def _compute_group_ids_mapping(self, dataset: COCOEvaluableDataset) -> dict[int, list[int]]:
        """Given a dataset, compute the mapping of group ids to relevant image ids."""
        raise NotImplementedError

    @staticmethod
    def _infinite_randperm(n: int, generator: torch.Generator) -> Generator[int, None, None]:
        while True:
            yield from torch.randperm(n, generator=generator).tolist()

    def batch_collation(self, batch: list) -> tuple[ImageList, tuple[BoxList, ...], tuple[int, ...]]:
        # tuple[tuple[torch.Tensor, ...], tuple[BoxList, ...], tuple[int, ...]]
        images, targets, img_ids = list(zip(*batch))
        images = ImageList.to_image_list(
            images,
            self.cfg.INPUT.N_DIM,
            size_divisible=self.cfg.DATALOADER.SIZE_DIVISIBILITY
        )
        return images, targets, img_ids

    def __len__(self) -> int:
        return self.num_batches
