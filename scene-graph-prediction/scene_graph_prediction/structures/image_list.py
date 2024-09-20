# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import annotations

import math
from typing import Sequence, Iterable

import torch
from typing_extensions import Self

_SIZE_T = tuple[int, ...]


class ImageList:
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size, and storing in a field the original sizes of each image.
    Note: padding is done by creating a new tensors of the new size and copying the old tensor in the top-left corner.
    Note: supports ND.
    """

    def __init__(self, tensors: torch.Tensor, image_sizes: Sequence[_SIZE_T]):
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.n_dim = len(image_sizes[0])

    def to(self, *args, **kwargs) -> Self:
        return ImageList(self.tensors.to(*args, **kwargs), self.image_sizes)

    def ith_image_as_image_list(self, i: int) -> Self:
        """Method to index a single image and return it as an ImageList of length 1."""
        if i >= len(self):
            raise ValueError(f"Trying to index image {i} in an ImageList of length {len(self)}")
        return ImageList(self.tensors[i][None], [self.image_sizes[i]])

    @staticmethod
    def to_image_list(
            tensors: ImageList | torch.Tensor | Iterable[torch.Tensor],
            n_dim: int,
            size_divisible: tuple[int, ...] | None = None
    ) -> ImageList:
        """
        When tensors is an iterable of tensors, it pads the tensors with zeros so that they have the same shape.
        :param tensors:
        :param n_dim: number of dimensions (typically 2 or 3).
        :param size_divisible: if > 0, the tensor size will be a multiple of size_divisible.
        """
        if size_divisible is None:
            size_divisible = (0,) * n_dim
        assert len(size_divisible) == n_dim

        if isinstance(tensors, ImageList):
            # Note: We should check that the size_divisible for tensors is the same as the one passed as arguments
            #       In practice, it's always the case...
            return tensors

        if isinstance(tensors, torch.Tensor):
            if not any(size_divisible):  # All 0
                # Single tensor shape can be inferred
                if tensors.dim() == n_dim + 1:  # n_dim + channel
                    tensors = tensors[None]

                assert tensors.dim() == n_dim + 2
                image_sizes = tuple(tensor.shape[-n_dim:] for tensor in tensors)
                return ImageList(tensors, image_sizes)

            # Else let the downstream code do the computations
            if tensors.dim() == n_dim + 1:
                # Single tensor: we need to create list with the tensor inside
                # We don't want to do this if we have a batch of tensors as argument (even if it's stupid)
                tensors = [tensors]

        # Sequence of tensors or batch of tensors
        if isinstance(tensors, Sequence | torch.Tensor):
            max_size = [max(s) for s in zip(*[img.shape for img in tensors])]
            for dim in range(n_dim):
                stride = size_divisible[dim]
                if stride > 0:
                    # Ignore channels
                    max_size[1 + dim] = int(math.ceil(max_size[1 + dim] / stride) * stride)

            batch_shape = [len(tensors)] + max_size
            batched_images = tensors[0].new(*batch_shape).zero_()
            for img, pad_img in zip(tensors, batched_images):
                slicer = tuple(slice(0, img.shape[i]) for i in range(len(img.shape)))
                pad_img[slicer].copy_(img)

            image_sizes = tuple(im.shape[-n_dim:] for im in tensors)

            return ImageList(batched_images, image_sizes)

        raise TypeError(f"Unsupported type for to_image_list: {type(tensors)}")

    def __len__(self):
        return len(self.image_sizes)

    def __iter__(self) -> ImageList:
        for idx in range(len(self)):
            yield self.ith_image_as_image_list(idx)
