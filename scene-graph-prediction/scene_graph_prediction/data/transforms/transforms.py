from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from scene_graph_api.utils.tensor import affine_transformation_grid
from torchvision.transforms import functional as func
from yacs.config import CfgNode as _CfgNode

from scene_graph_prediction.structures import BoxList, FieldExtractor, BoxListOps


class AbstractTransform(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, image: Any, target: BoxList) -> tuple[Any, BoxList]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(cfg: _CfgNode, is_train: bool) -> AbstractTransform:
        raise NotImplementedError


class Compose(torch.nn.Module):
    """
    Basically torch.nn.Sequential that:
    - accepts a target as argument and that over the pipeline.
    - converts the input Image/np.ndarray (whatever) to Tensor.
    """

    def __init__(self, transforms: list[AbstractTransform]):
        super().__init__()
        self.transforms = transforms

    def forward(self, image: Any, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n\t" + "\n\t".join(repr(t) for t in self.transforms) + "\n)"


class ResizeImage2D(AbstractTransform):
    """
    Take a 2D image in landscape or portrait format and resize so that
    its largest side is in [min_size, max_size] px.
    Note: tensors need to be CxHxW.
    """

    def __init__(self, min_size: int, max_size: int):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, image: Image | torch.Tensor, target: BoxList) -> tuple[Image | torch.Tensor, BoxList]:
        if isinstance(image, Image):
            w, h = image.size
            h, w, do_op = self._get_size((h, w))
        elif isinstance(image, torch.Tensor):
            # Ignore channel dim
            h, w, do_op = self._get_size(image.size()[1:])
        else:
            raise RuntimeError

        if not do_op:
            return image, target

        # PIL Images are also supported
        image = func.resize(image, [h, w])
        target = BoxListOps.resize(target, (h, w))
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> ResizeImage2D:
        if is_train:
            return ResizeImage2D(min_size=cfg.INPUT.MIN_SIZE_TRAIN, max_size=cfg.INPUT.MAX_SIZE_TRAIN)
        return ResizeImage2D(min_size=cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST)

    # Modified from torchvision to add support for max size
    def _get_size(self, image_size: tuple[int, int]) -> tuple[int, int, bool]:
        h, w = image_size

        min_original_size = min((w, h))
        max_original_size = max((w, h))
        new_max = np.clip(max_original_size, self.min_size, self.max_size)

        if new_max == max_original_size:
            # No-op
            return h, w, False

        scale = new_max / max_original_size
        max_original_size = round(max_original_size * scale)
        min_original_size = round(min_original_size * scale)

        if h < w:
            return min_original_size, max_original_size, True
        return max_original_size, min_original_size, True


class ResizeTensor(AbstractTransform):
    """Resize a [1, C, ..., Y, X] tensor."""

    def __init__(self, size: tuple[int, ...]):
        """:param size: with zyx ordering."""
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        image = torch.nn.functional.interpolate(
            image,
            size=self.size,
            mode="bilinear" if len(self.size) == 2 else "trilinear",
            align_corners=False
        )
        target = BoxListOps.resize(target, self.size)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, _: bool) -> ResizeTensor:
        return ResizeTensor(cfg.INPUT.RESIZE)


class RandomHorizontalFlip(AbstractTransform):
    """
    Performs a horizontal flip with probability prob.
    Note: supports 2D and 3D with or without channel and batch dimensions.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, image: Image | torch.Tensor, target: BoxList) -> tuple[Image | torch.Tensor, BoxList]:
        if random.random() < self.prob:
            # PIL Images are also supported
            image = func.hflip(image)  # Equivalent to torch.flip(image, dims=(image.dim() - 1,))
            target = BoxListOps.flip(target, BoxList.FlipDim.WIDTH)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> RandomHorizontalFlip:
        return RandomHorizontalFlip(cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN if is_train else 0)


class RandomVerticalFlip(AbstractTransform):
    """
    Performs a vertical flip with probability prob.
    Note: supports 2D and 3D with or without channel and batch dimensions.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, image: Image | torch.Tensor, target: BoxList) -> tuple[Image | torch.Tensor, BoxList]:
        if random.random() < self.prob:
            # PIL Images are also supported
            image = func.vflip(image)  # Equivalent to torch.flip(image, dims=(image.dim() - 2,))
            target = BoxListOps.flip(target, BoxList.FlipDim.HEIGHT)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> RandomVerticalFlip:
        return RandomVerticalFlip(cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN if is_train else 0)


class RandomDepthFlip(AbstractTransform):
    """
    Performs a depth flip with probability prob.
    Note: supports 2D and 3D with or without channel and batch dimensions.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        if random.random() < self.prob:
            image = torch.flip(image, dims=(image.dim() - 3,))
            target = BoxListOps.flip(target, BoxList.FlipDim.DEPTH)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> RandomDepthFlip:
        return RandomDepthFlip(cfg.INPUT.DEPTH_FLIP_PROB_TRAIN if is_train else 0)


class ColorJitter(torchvision.transforms.ColorJitter, AbstractTransform):
    """
    Wrapper for torchvision.transforms.ColorJitter to also accept a target argument.
    Only supports 2D.
    """

    # noinspection PyMethodOverriding
    def forward(self, image: Image | torch.Tensor, target: BoxList) -> tuple[Image | torch.Tensor, BoxList]:
        return super()(image), target

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> ColorJitter:
        if is_train:
            return ColorJitter(
                brightness=cfg.INPUT.BRIGHTNESS,
                contrast=cfg.INPUT.CONTRAST,
                saturation=cfg.INPUT.SATURATION,
                hue=cfg.INPUT.HUE
            )
        return ColorJitter(
            brightness=0.,
            contrast=0.,
            saturation=0.,
            hue=0.
        )


class ToTensor(AbstractTransform):
    """Cast whatever to Tensor."""

    def forward(self, image: Image | torch.Tensor | np.ndarray, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        if isinstance(image, torch.Tensor):
            return image, target  # no op
        return func.to_tensor(image), target

    @staticmethod
    def build(_: _CfgNode, __: bool) -> ToTensor:
        return ToTensor()


class Normalize(AbstractTransform):
    """
    Normalize a tensor with a channel dim but no batch dim.
    Note: support 2D and 3D contrary to the torchvision implementation.
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]):
        """
        :param mean: should have as many elements as images have channels.
        :param std: should have as many elements as images have channels.
        """
        assert len(mean) == len(std), f"Mean and std should have the same length ({len(mean)} vs {len(std)})"
        assert all(s > 0 for s in std), "The standard deviation needs to be strictly positive for all dims."
        super().__init__()
        self.mean = mean
        self.std = std

    # noinspection PyMethodOverriding
    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        assert image.shape[0] == len(self.mean)
        image = copy.deepcopy(image)
        for c, (m, s) in enumerate(zip(self.mean, self.std)):
            image[c] -= m
            image[c] /= s

        return image, target

    @staticmethod
    def build(cfg: _CfgNode, _: bool) -> Normalize:
        return Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)


class ClipAndRescale(AbstractTransform):
    """Clips values to [min_val, max_val] and then rescales to [0, 1]."""

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        assert min_val < max_val, (min_val, max_val)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        # noinspection PyTypeChecker
        return torch.clip(image, self.min_val, self.max_val) / (self.max_val - self.min_val), target

    @staticmethod
    def build(cfg: _CfgNode, _: bool) -> ClipAndRescale:
        return ClipAndRescale(min_val=cfg.INPUT.CLIP_MIN, max_val=cfg.INPUT.CLIP_MAX)


class AddChannelDim(AbstractTransform):
    """Add a dummy channel dim for grayscale images e.g. reshape head CT from (d, h, w) to (1, d, h, w)."""

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        return image[None], target

    @staticmethod
    def build(_: _CfgNode, __: bool) -> AddChannelDim:
        return AddChannelDim()


class BoundingBoxPerturbation(AbstractTransform):
    """
    Randomly shifts bounding boxes corners by a random number of pixels (upto max_shift).
    Note: when clipping the boxes, empty ones are removed.
    Note: boxes are not deformed and segmentation masks are not shifted.
    Note: should always be called after transforms that compute the boxes from the segmentation e.g. RandomAffine.
    """

    def __init__(self, max_shift: int | tuple[int, ...], n_dim: int):
        """
        :param max_shift: either a different max shift for each dim or a single shift that will be used for all.
        """
        super().__init__()
        if isinstance(max_shift, int):
            max_shift = (max_shift,) * n_dim
        assert all(shift >= 0 for shift in max_shift), max_shift
        self.max_shift = max_shift
        self.n_dim = n_dim

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        assert target.n_dim == self.n_dim
        target = target.copy_with_all_fields()
        # Compute separate shifts for each dim for each bb
        shifts = [torch.randint(low=-shift, high=shift + 1, size=(len(target), 1)) for shift in self.max_shift]
        stacked_shifts = torch.hstack(shifts * 2)  # Shift each end bb by as much as each start
        target.boxes += stacked_shifts
        target = BoxListOps.clip_to_image(target, remove_empty=True)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, train: bool) -> BoundingBoxPerturbation:
        return BoundingBoxPerturbation(max_shift=cfg.INPUT.MAX_BB_SHIFT if train else 0, n_dim=cfg.INPUT.N_DIM)


class RandomAffine(AbstractTransform):
    """
    Random affine transform. See more at scene_graph_predict.utils.tensor.affine_transformation_grid.
    Also prepare the segmentation mask for training and can replace the PrepareMasks transform.
    """

    def __init__(
            self,
            n_dim: int,
            max_translate: tuple[int, ...],
            scale_range: tuple[tuple[float, float], ...],
            max_rotate: tuple[float, ...],
            is_train: bool = True,
            output_segmentation: bool = False,
            output_masks: bool = False
    ):
        super().__init__()
        assert n_dim in [2, 3]
        assert len(max_translate) == n_dim
        assert len(scale_range) == n_dim and all(r[0] <= r[1] for r in scale_range)
        assert (n_dim == 2 and len(max_rotate) == 1) or (n_dim == 3 and len(max_rotate) == 3)

        self.n_dim = n_dim
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.max_rotate = max_rotate
        self.is_train = is_train
        self.output_segmentation = output_segmentation
        self.output_masks = output_masks

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        """
        :param image: An image tensor with a channel dim and no batch dim.
        :param target: a BoxList containing the annotation.
        :return: The transformed image+target.
        """
        if not self.is_train:
            # Optimize to (almost) no-op
            # We need to make a shallow copy to protect against fields removal
            # Also we need to make sure that only the correct mask fields are present
            not_transformed_target = target.copy_with_all_fields()
            if self.output_segmentation:
                not_transformed_target.SEGMENTATION = FieldExtractor.segmentation(target)
            else:
                not_transformed_target.del_field(BoxList.AnnotationField.SEGMENTATION)
            if self.output_masks:
                not_transformed_target.MASKS = FieldExtractor.masks(target)
            else:
                not_transformed_target.del_field(BoxList.AnnotationField.MASKS)
            not_transformed_target.del_field(BoxList.AnnotationField.LABELMAP)  # Not needed anymore
            return image, not_transformed_target

        assert image.dim() == self.n_dim + 1

        # Sample transform parameters
        translate = tuple(0 if max_t == 0 else np.random.randint(-max_t, max_t + 1) for max_t in self.max_translate)
        scale = tuple(np.random.uniform(min_s, max_s) for min_s, max_s in self.scale_range)
        rotate = tuple(np.random.uniform(-max_r, max_r) for max_r in self.max_rotate)

        # Compute sampling grid (1xDxHxWxn_dim)
        grid = affine_transformation_grid(tensor_size=image.shape[1:], translate=translate, scale=scale, rotate=rotate)
        grid = grid.to(device=image.device, dtype=image.dtype)
        # Adjust to the number of channels
        grid = grid.tile((image.shape[0],) + (1,) * (self.n_dim + 1))

        # Create batch dim for image and remove it after the transform
        transformed_image = torch.nn.functional.grid_sample(image[None], grid, align_corners=False)[0]

        return transformed_image, BoxListOps.affine_transformation(
            target,
            translate=translate,
            scale=scale,
            rotate=rotate,
            output_labelmap=False,  # Not needed anymore
            output_segmentation=self.output_segmentation,
            output_masks=self.output_masks
        )

    @staticmethod
    def build(cfg: _CfgNode, is_train: bool) -> RandomAffine:
        n_dim = cfg.INPUT.N_DIM
        if is_train:
            max_translate = cfg.INPUT.AFFINE_MAX_TRANSLATE
            scale_range = cfg.INPUT.AFFINE_SCALE_RANGE
            max_rotate = cfg.INPUT.AFFINE_MAX_ROTATE
        else:
            max_translate = (0,)
            scale_range = ((1., 1.),)
            max_rotate = (0.,) if n_dim == 2 else (0.,) * 3

        if len(max_translate) == 1:
            max_translate *= n_dim

        if len(scale_range) == 1:
            scale_range *= n_dim

        return RandomAffine(
            n_dim,
            max_translate,
            scale_range,
            max_rotate,
            is_train,
            output_segmentation=cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION,
            output_masks=cfg.MODEL.MASK_ON,
        )


class PrepareMasks(AbstractTransform):
    """Prepare the segmentation mask for training."""

    def __init__(self, output_segmentation: bool = False, output_masks: bool = False):
        super().__init__()
        self.output_segmentation = output_segmentation
        self.output_masks = output_masks

    def forward(self, image: torch.Tensor, target: BoxList) -> tuple[torch.Tensor, BoxList]:
        if self.output_masks:
            if not target.has_field(BoxList.AnnotationField.MASKS):
                target.MASKS = FieldExtractor.masks(target)
        else:
            target.del_field(BoxList.AnnotationField.MASKS)

        if self.output_segmentation:
            if not target.has_field(BoxList.AnnotationField.SEGMENTATION):
                target.SEGMENTATION = FieldExtractor.segmentation(target)
        else:
            target.del_field(target.AnnotationField.SEGMENTATION)

        target.del_field(BoxList.AnnotationField.LABELMAP)
        return image, target

    @staticmethod
    def build(cfg: _CfgNode, _: bool) -> PrepareMasks:
        return PrepareMasks(
            output_segmentation=cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION,
            output_masks=cfg.MODEL.MASK_ON,
        )
