# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """Samples positive and negative cases."""
    IGNORE = -1

    def __init__(self, batch_size_per_image: int, positive_fraction: float):
        """
        :param batch_size_per_image: Number of elements to be selected per image
        :param positive_fraction: Percentage of positive elements per batch.
                                  Note: Not enforced if there are not enough positive samples in a batch.
        """
        super().__init__()
        assert batch_size_per_image > 0
        assert 0 <= positive_fraction <= 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.min_neg = 1

    @abstractmethod
    def __call__(
            self,
            labels: list[torch.Tensor],
            fg_probs: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        :param labels: List of tensors containing labels with 0 as bg or positive values for fg classes.
                       Each tensor corresponds to a specific image.
                       IGNORE (-1) values are ignored.
        :param fg_probs: Maximum predicted foreground probability.
        :return: Two lists of binary masks for each image.
                 The first list contains the positive elements that were selected,
                 and the second list the negative elements.
        """
        raise NotImplementedError

    @staticmethod
    def _random_select(tensor: torch.Tensor, max_numel: int):
        """
        Compute a random permutation of the labels and selects up to max_numel.
        Has a safeguard against having a no-empty tensor and max_numel == 0.
        """
        if max_numel == 0:
            return torch.tensor([], dtype=torch.long, device=tensor.device)
        perm = torch.randperm(tensor.numel(), device=tensor.device)[:max_numel]
        return tensor[perm]
