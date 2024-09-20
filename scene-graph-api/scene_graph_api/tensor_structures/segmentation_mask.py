"""
ABSTRACT
Segmentations come in either:
1) Binary masks
2) Polygons

Binary masks can be represented in a contiguous array and operations can be carried out more efficiently,
therefore BinaryMaskList handles them together.

Polygons are handled separately for each instance, by PolygonInstance and instances are handled by PolygonList.

Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from functools import reduce
from typing import Iterator, Sequence

import cv2
import numpy as np
import pycocotools3d.mask as mask_utils
import pycocotools3d.mask3d as mask_utils3d
import torch
from typing_extensions import Self

from ..utils.indexing import FlipDim
from ..utils.indexing import sanitize_cropping_box

_SIZE_T = tuple[int, ...]
_GET_ITEM_T = int | list[int] | list[bool] | torch.Tensor


class AbstractMask(ABC):
    """Abstraction for segmentation masks."""

    FlipDIm = FlipDim

    @abstractmethod
    def flip(self, dim: FlipDim) -> Self:
        """Mask sizes are assumed to be zyx ordered."""
        raise NotImplementedError

    @abstractmethod
    def crop(self, box: Sequence) -> Self:
        raise NotImplementedError

    @abstractmethod
    def resize(self, size: _SIZE_T) -> Self:
        raise NotImplementedError

    @abstractmethod
    def to(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @abstractmethod
    def convert_to_binary_mask(self) -> torch.Tensor:
        """To raw tensor labelmap."""
        raise NotImplementedError


class AbstractMaskList(ABC):
    """
    Abstraction for lists of segmentation masks.
    All of these classes use __getitem__ and __iter__ to update the indexing
    while returning an object of the same class.
    Use get_mask_tensor to access the tensors once the sampling is done.
    """

    size: _SIZE_T
    n_dim: int
    FlipDim = FlipDim

    @abstractmethod
    def flip(self, dim: FlipDim) -> Self:
        raise NotImplementedError

    @abstractmethod
    def crop(self, box: Sequence[float]) -> Self:
        """Note: needs to match BoxList.crop"""
        raise NotImplementedError

    @abstractmethod
    def resize(self, size: _SIZE_T) -> Self:
        raise NotImplementedError

    @abstractmethod
    def to(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get_mask_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: _GET_ITEM_T) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Self]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def device(self):
        raise NotImplementedError


class BinaryMaskList(AbstractMaskList):
    """
    This class handles binary masks for all objects in the image.
    Note: supports ND (except RLE which is 2D/3D only).
    """

    def __init__(
            self,
            masks: torch.Tensor | list[torch.Tensor] | list[dict] | Self,
            size: _SIZE_T
    ):
        """
        After initialization, a hard copy will be made, to leave the initializing source data intact.
        Mask should be zyx indexed.
        Note: param description are for 2D, but ND is supported.
        :param masks: Either torch.Tensor of [num_instances, H, W]
                      or list of tensors of [H, W] with num_instances elements,
                      or RLE (Run Length Encoding) - interpreted as list of dicts,
                      or BinaryMaskList.
        :param size: absolute image size, zyx ordered
        """
        self.masks = torch.empty(0)
        self.n_dim = len(size)

        assert isinstance(size, tuple)

        if isinstance(masks, torch.Tensor):
            # The raw data representation is passed as argument
            masks = masks
        elif isinstance(masks, Sequence):
            if len(masks) == 0:
                masks = torch.empty((0,) + size)  # num_instances = 0!
            elif isinstance(masks[0], torch.Tensor):
                # noinspection PyTypeChecker
                masks = torch.stack(masks, dim=0)
            elif isinstance(masks[0], dict) and "counts" in masks[0]:
                # RLE interpretation
                assert self.n_dim in [2, 3]
                rle_sizes = [tuple(inst["size"]) for inst in masks]

                if self.n_dim == 2:
                    masks = mask_utils.decode(masks)  # [h, w, n]
                else:
                    masks = mask_utils3d.decode(masks)  # [d, h, w, n]
                masks = torch.tensor(masks).permute(self.n_dim, *list(range(self.n_dim)))

                assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), \
                    f"All the sizes must be the same size: {rle_sizes}"

                # In RLE, height comes first in "size"
                for i in range(self.n_dim):
                    assert masks.shape[1 + i] == rle_sizes[0][i]

                if any(s != rle_s for s, rle_s in zip(size, rle_sizes[0])):
                    masks = torch.nn.functional.interpolate(
                        input=masks[None].float(),
                        size=tuple(size),
                        mode="bilinear",
                        align_corners=False
                    )[0].type_as(masks)
            else:
                RuntimeError(f"Type of `masks[0]` could not be interpreted: {type(masks)}")
        elif isinstance(masks, BinaryMaskList):
            masks = torch.clone(masks.masks)
        else:
            RuntimeError(f"Type of `masks` argument could not be interpreted: {type(masks)}")

        # Add batch dim if necessary
        if len(masks.shape) == self.n_dim:
            # noinspection PyTypeChecker
            masks = masks[None]

        assert len(masks.shape) == self.n_dim + 1
        for i in range(self.n_dim):
            assert masks.shape[i + 1] == size[i], f"{masks.shape[i + 1]} != {size[i]}"

        self.masks: torch.Tensor = masks
        self.size = tuple(size)

    def flip(self, dim: FlipDim) -> Self:
        assert self.n_dim == 3 or dim != FlipDim.DEPTH
        # BxDxHxW
        return BinaryMaskList(self.masks.flip(1 + self.n_dim + dim.to_idx_from_last()), self.size)

    def crop(self, box: Sequence[int]) -> Self:
        """Box is assumed to be xyxy."""
        assert isinstance(box, Sequence), str(type(box))
        box = sanitize_cropping_box(box, self.size)
        slicer = (slice(None),) + tuple(slice(box[dim], box[dim + self.n_dim] + 1) for dim in range(self.n_dim))
        cropped_masks = self.masks[slicer]
        cropped_size = tuple(box[dim + self.n_dim] - box[dim] + 1 for dim in range(self.n_dim))
        return BinaryMaskList(cropped_masks, cropped_size)

    def resize(self, size: _SIZE_T) -> Self:
        assert len(size) == self.n_dim
        for s in size:
            assert s > 0, size

        resized_masks = torch.nn.functional.interpolate(
            input=self.masks[None].float(),  # As NxCxDxHxW
            size=size,
            mode="bilinear" if self.n_dim == 2 else "trilinear",
            align_corners=False
        )[0].type_as(self.masks)
        return BinaryMaskList(resized_masks, size)

    def to(self, *args, **kwargs) -> Self:
        return BinaryMaskList(self.masks.to(*args, **kwargs), self.size)

    def get_mask_tensor(self) -> torch.Tensor:
        return self.masks.squeeze(0)

    def convert_to_polygon_list(self) -> Self:
        if self.masks.numel() == 0:
            return PolygonList([], self.size)
        assert self.n_dim == 2, "Implemented only for 2D."
        return PolygonList(self._find_contours(), self.size)

    def _find_contours(self) -> list[list[list[int]]]:
        """Masks need to have the dtype np.uint8."""
        assert self.n_dim == 2, "Implemented only for 2D."
        contours = []
        # Add the type conversion as a security as float arrays are not supported
        masks = self.masks.detach().numpy().astype(np.uint8)
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

            reshaped_contour: list[list[int]] = []
            for entity in contour:
                entity = entity.get()
                assert len(entity.shape) == 3
                assert entity.shape[1] == 1, "Hierarchical contours are not allowed"

                xyxyxy: list[int] = entity.reshape(-1).tolist()
                yxyxyx: list[int] = reduce(lambda a, b: a + b, [[y, x] for x, y in zip(xyxyxy[::2], xyxyxy[1::2])])
                reshaped_contour.append(yxyxyx)

            # Need to convert coordinates from xyxyxy... to yxyxyx...
            contours.append(reshaped_contour)
        return contours

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, index: _GET_ITEM_T) -> Self:
        if self.masks.numel() == 0:
            raise RuntimeError("Indexing empty BinaryMaskList")
        return BinaryMaskList(self.masks[index], self.size)

    def __iter__(self) -> Iterator[Self]:
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)}, image_size={self.size})"

    @property
    def device(self) -> torch.device:
        return self.masks.device


class PolygonInstance(AbstractMask):
    """
    This class holds a set of polygons that represents a single instance of an object mask.
    The object can be represented as a set of polygons.
    """

    def __init__(self, polygons: list[list[float]] | Self, size: _SIZE_T):
        """
        :param polygons: The first level refers to all the polygons that compose the object,
                         and the second level to the polygon coordinates (zyx ordered).
        """
        self.polygons: list[float] = []
        self.n_dim = len(size)

        if isinstance(polygons, Sequence):
            valid_polygons = []
            for p in polygons:
                p = torch.as_tensor(p, dtype=torch.float32)
                if len(p) >= 3 * self.n_dim:  # At least 3 coordinates
                    valid_polygons.append(p)
            polygons = valid_polygons

        elif isinstance(polygons, PolygonInstance):
            polygons = copy.copy(polygons.polygons)
        else:
            RuntimeError(f"Type of argument `polygons` is not allowed: {type(polygons)}")

        self.polygons = polygons
        self.size = tuple(size)

    def flip(self, dim: FlipDim) -> Self:
        assert self.n_dim == 3 or dim != FlipDim.DEPTH
        int_dim = self.n_dim + dim.to_idx_from_last()
        dim_size = self.size[int_dim]

        flipped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[int_dim::self.n_dim] = dim_size - poly[int_dim::self.n_dim] - 1
            flipped_polygons.append(p)

        return PolygonInstance(flipped_polygons, size=self.size)

    def crop(self, box: Sequence[int]) -> Self:
        """Box is assumed to be xyxy."""
        assert isinstance(box, Sequence), str(type(box))
        box = sanitize_cropping_box(box, self.size)
        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            for dim in range(self.n_dim):
                p[dim::self.n_dim] = p[dim::self.n_dim] - box[dim]
            cropped_polygons.append(p)
        cropped_size = tuple(box[dim + self.n_dim] - box[dim] + 1 for dim in range(self.n_dim))

        return PolygonInstance(cropped_polygons, size=cropped_size)

    def resize(self, size: _SIZE_T) -> Self:
        assert len(size) == self.n_dim
        ratios = tuple(s / s_orig for s, s_orig in zip(size, self.size))

        if ratios == (ratios[0],) * self.n_dim:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return PolygonInstance(scaled_polys, size)

        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            for dim in range(self.n_dim):
                p[dim::self.n_dim] *= ratios[dim]
            scaled_polygons.append(p)

        return PolygonInstance(scaled_polygons, size=size)

    def to(self, *args, **kwargs) -> Self:
        return PolygonInstance(self.polygons, self.size)

    def convert_to_binary_mask(self) -> torch.Tensor:
        """To raw tensor labelmap."""
        assert self.n_dim == 2, "Implemented only for 2D."
        height, width = self.size
        # formatting for COCO PythonAPI
        polygons = [p.numpy() for p in self.polygons]
        all_rle = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(all_rle)
        mask = mask_utils.decode(rle)
        mask = torch.from_numpy(mask)
        return mask

    def __len__(self) -> int:
        return len(self.polygons)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_groups={len(self.polygons)}, image_size={self.size})"


class PolygonList(AbstractMaskList):
    """This class handles PolygonInstances for all objects in the image."""

    def __init__(
            self,
            polygons: list[list[list]] | list[PolygonInstance] | Self,
            size: _SIZE_T
    ):
        self.polygons: list[PolygonInstance] = []
        self.n_dim = len(size)

        if isinstance(polygons, Sequence):
            if len(polygons) == 0:
                polygons = [[[]]]
            if isinstance(polygons[0], Sequence):
                assert isinstance(polygons[0][0], Sequence), str(type(polygons[0][0]))
            else:
                assert isinstance(polygons[0], PolygonInstance), str(type(polygons[0]))

        elif isinstance(polygons, PolygonList):
            polygons = polygons.polygons

        else:
            RuntimeError(f"Type of argument `polygons` is not allowed: {type(polygons)}")

        assert isinstance(size, Sequence), str(type(size))

        self.polygons = []
        for p in polygons:
            p = PolygonInstance(p, size)
            if len(p) > 0:
                self.polygons.append(p)

        self.size = tuple(size)

    def flip(self, dim: FlipDim) -> Self:
        assert self.n_dim == 3 or dim != FlipDim.DEPTH
        return PolygonList([polygon.flip(dim) for polygon in self.polygons], size=self.size)

    def crop(self, box: Sequence[int]) -> Self:
        cropped_polygons = []
        for polygon in self.polygons:
            cropped_polygons.append(polygon.crop(box))
        size = tuple(box[self.n_dim + dim] - box[dim] + 1 for dim in range(self.n_dim))
        return PolygonList(cropped_polygons, size)

    def resize(self, size: _SIZE_T) -> Self:
        assert len(size) == self.n_dim
        return PolygonList([polygon.resize(size) for polygon in self.polygons], size)

    def to(self, *args, **kwargs) -> Self:
        return PolygonList(self, self.size)

    def get_mask_tensor(self) -> torch.Tensor:
        return self.convert_to_binary_mask_list().get_mask_tensor()

    def convert_to_binary_mask_list(self) -> BinaryMaskList:
        if len(self) > 0:
            masks = torch.stack([p.convert_to_binary_mask() for p in self.polygons])
        else:
            masks = torch.empty((0,) + self.size, dtype=torch.uint8)
        return BinaryMaskList(masks, self.size)

    def __len__(self) -> int:
        return len(self.polygons)

    def __getitem__(self, item: _GET_ITEM_T) -> Self:
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self) -> Iterator[Self]:
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self.polygons)}, image_size={self.size})"

    def device(self):
        return torch.device("cpu")
