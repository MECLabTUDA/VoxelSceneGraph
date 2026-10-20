# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from ...abstractions.sampler import Sampler


class BalancedSampler(Sampler):
    """This class samples batches, ensuring that they contain a fixed proportion of positives/negatives."""

    def __call__(
            self,
            labels: list[torch.Tensor],
            _: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        :param labels: List of tensors containing labels with 0 as bg or positive values for fg classes.
                       Each tensor corresponds to a specific image.
                       IGNORE (-1) values are ignored.
        :return: Two lists of binary masks for each image.
                 The first list contains the positive elements that were selected,
                 and the second list the negative elements.
        """
        pos_masks = []
        neg_masks = []
        for matched_label_per_image in labels:
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
                # This can actually happen if the RPN is very good
                # # Protect against not enough negative examples (should never happen)
                # if negative.numel() < num_neg:  # Note: if positive_fraction == 1
                #     # Cannot have as many negatives as desired, so we need to dial back on the number of positives
                #     num_neg = negative.numel()
                #     num_pos = round(num_neg / (1 / self.positive_fraction - 1))

                # But at least one negative
                num_neg = max(num_neg, 1)

            # Randomly select positive and negative examples
            pos_idx_per_image = self._random_select(positive, num_pos)
            neg_idx_per_image = self._random_select(negative, num_neg)

            # Create binary mask from indices
            pos_idx_per_image_mask = torch.zeros(matched_label_per_image.shape[0], dtype=torch.bool)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            pos_masks.append(pos_idx_per_image_mask)

            neg_idx_per_image_mask = torch.zeros(matched_label_per_image.shape[0], dtype=torch.bool)
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            neg_masks.append(neg_idx_per_image_mask)

        return pos_masks, neg_masks
