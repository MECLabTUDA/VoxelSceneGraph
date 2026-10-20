# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Iterator

from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled."""

    def __init__(
            self,
            batch_sampler: BatchSampler,
            num_iterations: int,
            start_iter: int = 0
    ):
        super().__init__(batch_sampler.sampler, batch_sampler.batch_size, batch_sampler.drop_last)
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self) -> Iterator[list[int]]:
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # If the underlying sampler has a set_epoch method, like DistributedSampler,
            # used for making each process see a different split of the dataset, then set it
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(iteration)
            for batch in super().__iter__():
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self) -> int:
        return self.num_iterations
