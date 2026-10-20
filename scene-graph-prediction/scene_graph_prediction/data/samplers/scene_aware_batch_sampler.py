from typing import Iterator, Generator

import torch
from torch.utils.data.sampler import BatchSampler, Sampler
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, ImageList


class SceneAwareBatchSampler(BatchSampler):
    """
    A batch sampler typically samples a fixed number of images.
    For relation prediction, the relevant metric can rather be the total number of objects in the image batch.
    This batch sampler rather samples images until the total number of objects does not exceed the batch size.
    Also includes an IterationBasedSampler.
    """

    def __init__(
            self,
            cfg: CfgNode,
            sampler: Sampler,
            batch_size: int,
            drop_last: bool,
            start_batch: int,
            num_batches: int,
            # Whether to exclude unique objects in the total count
            ignore_unique_objects: bool = False
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.cfg = cfg
        self.start_batch = start_batch
        self.num_batches = num_batches

        # noinspection PyUnresolvedReferences
        dataset = sampler.data_source

        # Compute an idx to length mapping
        mapping = {}
        num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1
        for img_idx in range(len(dataset)):
            # Reindex required to for unique classes to be last in order
            # noinspection PyUnresolvedReferences
            target = dataset.reindex_groundtruth(dataset.get_groundtruth(img_idx))
            cnt = len(target)
            if ignore_unique_objects:
                labels = target.LABELS
                cnt -= torch.sum(labels > num_normal_fg_classes).item()
            mapping[img_idx] = cnt
        self.idx_to_obj_cnt = mapping

    def __iter__(self) -> Iterator[list[int]]:
        batch_idx = self.start_batch
        generator = self._infinite_sampler()

        cur_cnt = 0
        batch = []
        while batch_idx < self.num_batches:
            while True:
                next_idx = next(generator)
                next_cnt = self.idx_to_obj_cnt[next_idx]
                if self.batch_size - cur_cnt < next_cnt:
                    # Batch is too big
                    break
                # Add next_idx to the batch
                batch.append(next_idx)
                cur_cnt += next_cnt
            yield batch

            # Initialize the new batch to contain the last sampled idx
            batch = [next_idx]
            cur_cnt = next_cnt

            batch_idx += 1

    def batch_collation(self, batch: list) -> tuple[ImageList, tuple[BoxList, ...], tuple[int, ...]]:
        images, precomputations, targets, img_ids = list(zip(*batch))
        images = ImageList.to_image_list(
            images,
            self.cfg.INPUT.N_DIM,
            size_divisible=self.cfg.DATALOADER.SIZE_DIVISIBILITY
        )
        return images, precomputations, targets, img_ids

    def _infinite_sampler(self) -> Generator[int, None, None]:
        while True:
            yield from self.sampler

    def __len__(self) -> int:
        return self.num_batches
