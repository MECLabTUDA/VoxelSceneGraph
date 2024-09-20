# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from scene_graph_prediction.structures import ImageList, BoxList


class BatchCollator:
    """
    From a list of samples from the dataset, returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, n_dim: int, size_divisible: tuple[int, ...] = 0):
        self.size_divisible = size_divisible
        self.n_dim = n_dim

    def __call__(self, batch: list) -> tuple[ImageList, tuple[BoxList, ...], tuple[int, ...]]:
        # tuple[tuple[torch.Tensor, ...], tuple[BoxList, ...], tuple[int, ...]]
        images, targets, img_ids = list(zip(*batch))
        images = ImageList.to_image_list(images, self.n_dim, size_divisible=self.size_divisible)
        return images, targets, img_ids
