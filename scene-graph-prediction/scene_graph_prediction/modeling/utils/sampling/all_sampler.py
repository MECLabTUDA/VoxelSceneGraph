# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from ...abstractions.sampler import Sampler


class AllSampler(Sampler):
    """
    Always sample all (even ignored).
    Note: ignores the batch size and positive fraction.
    """

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
            pos_masks.append(matched_label_per_image != 0)  # Also includes ignored objects
            neg_masks.append(matched_label_per_image == 0)

        # noinspection PyTypeChecker
        return pos_masks, neg_masks
