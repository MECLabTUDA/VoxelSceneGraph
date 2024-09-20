# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from ...abstractions.sampler import Sampler


class HardNegativeSampler(Sampler):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives/negatives.
    Note: negatives are false positives if possible.
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float, pool_size: float = 20):
        """
        :param batch_size_per_image: Number of elements to be selected per image
        :param positive_fraction: Percentage of positive elements per batch.
                                  Note: Not enforced if there are not enough positive samples in a batch.
        :param pool_size: ratio (>= 1) of false positives to sample (compared to number of negatives to sample).
        """
        super().__init__(batch_size_per_image, positive_fraction)
        self.pool_size = pool_size

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

        # Split fg probs
        anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in labels]
        fg_probs_split: list[torch.Tensor] = fg_probs.split(anchors_per_image, 0)

        pos_masks = []
        neg_masks = []
        for matched_label_per_image, fg_probs_per_image in zip(labels, fg_probs_split):
            # noinspection PyTypeChecker
            positive = torch.nonzero(matched_label_per_image >= 1).squeeze(1)
            # noinspection PyTypeChecker
            negative = torch.nonzero(matched_label_per_image == 0).squeeze(1)

            if self.positive_fraction < 1e-5:
                # Positive fraction too low to have any
                num_pos = 0
                num_neg = min(negative.numel(), self.batch_size_per_image)
            else:
                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # Protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)

                # Ensure balancing when not enough positive cases
                num_neg = round(num_pos / self.positive_fraction - num_pos)
                # Protect against not enough negative examples (should never happen)
                if negative.numel() < num_neg:  # Note: if positive_fraction == 1
                    # Cannot have as many negatives as desired, so we need to dial back on the number of positives
                    num_neg = negative.numel()
                    num_pos = round(num_neg / (1 / self.positive_fraction - 1))

                # But at least one negative
                num_neg = max(num_neg, 1)

            # Randomly select positive examples
            pos_idx_per_image = self._random_select(positive, num_pos)

            neg_idx_per_image = self._select_negative(negative, num_neg, fg_probs_per_image)

            # Create binary mask from indices
            pos_idx_per_image_mask = torch.zeros(matched_label_per_image.shape[0], dtype=torch.bool)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            pos_masks.append(pos_idx_per_image_mask)

            neg_idx_per_image_mask = torch.zeros(matched_label_per_image.shape[0], dtype=torch.bool)
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            neg_masks.append(neg_idx_per_image_mask)

        return pos_masks, neg_masks

    def _select_negative(
            self,
            negative: torch.Tensor,
            num_neg: int,
            fg_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        :param negative: Indexes for negative labels.
        :param num_neg: Number of negative examples to sample.
        :param fg_probs: Maximum predicted foreground probability.
        :returns: sampled indexes.
        """
        pool = min(int(num_neg * self.pool_size), negative.numel())
        # Select false negatives
        _, negative_idx_pool = fg_probs[negative].topk(pool, sorted=False)
        negative = negative[negative_idx_pool]

        # Random select from that
        return self._random_select(negative, num_neg)
