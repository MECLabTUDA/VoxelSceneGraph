# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools
from typing import Iterator

import torch
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follow an ordering,
    which is as close as possible to the ordering from the original sampler.
    """

    def __init__(
            self,
            sampler: Sampler,
            group_ids: list[int],
            batch_size: int,
            drop_incomplete_batches: bool = False
    ):
        """
        :param sampler: Base sampler.
        :param group_ids: list or tensor of labels
        :param batch_size: Size of mini-batch.
        :param drop_incomplete_batches: If True, the sampler will drop the batches whose size is less than batch_size.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(f"Sampler should be an instance of torch.utils.data.Sampler, but got {sampler}")
        super().__init__(sampler, batch_size, False)

        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_incomplete_batches = drop_incomplete_batches

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False
        self._batches: list | None = None

    def _prepare_batches(self) -> list[list]:
        dataset_size = len(self.group_ids)
        # Get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # Potentially, not all elements of the dataset were sampled by the sampler (e.g., DistributedSampler).
        # Construct a tensor which contains -1 if the element was not sampled,
        # and a non-negative number indicating the order where the element was sampled.
        # For example, if sampled_ids = [3, 1] and dataset_size = 5, the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # Get a mask with the elements that were sampled
        mask = order >= 0

        # Find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # Get relative order of the elements inside each cluster that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # With the relative order, find the absolute order in the sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # Permute each cluster so that they follow the order from the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # Splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # Now each batch internally has the right order, but they are grouped by clusters.
        # Find the permutation between different batches that brings them as close as possible to
        # the order that we have in the sampler.
        # For that, we will consider the ordering as coming from the first element of each batch, and sort accordingly
        first_element_of_batch = [t[0].item() for t in merged]

        # Get an inverse mapping from sampled indices and the position where they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # From the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor([inv_sampled_ids_map[s] for s in first_element_of_batch])

        # Permute the batches so that they approximately follow the order from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()

        # Finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if not self.drop_incomplete_batches:
            return batches

        # Drop incomplete batches
        kept = []
        for batch in batches:
            if len(batch) == self.batch_size:
                kept.append(batch)
        return kept

    def __iter__(self) -> Iterator[list]:
        if self._can_reuse_batches:
            self._can_reuse_batches = False
        else:
            self._batches = self._prepare_batches()
        return iter(self._batches)

    def __len__(self) -> int:
        # Weird optimization such that if you ever call __len__ before __iter__,
        # then the batches that get computed are reused during the next __iter__ call
        if self._batches is None:
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
