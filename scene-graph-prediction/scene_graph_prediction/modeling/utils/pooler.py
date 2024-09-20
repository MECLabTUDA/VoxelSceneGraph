# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import ROIAlign, ROIAlign3D
from scene_graph_prediction.structures import BoxList, BoxListOps
from .build_layers import build_conv3x3
from .misc import cat, ROIHeadName
from ..abstractions.backbone import FeatureMaps, AnchorStrides


class _LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to, based on the heuristic in the FPN paper."""

    def __init__(
            self,
            k_min: int,
            k_max: int,
            eps: float = 1e-6
    ):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.eps = eps

    def __call__(self, boxlists: list[BoxList]) -> torch.LongTensor:
        # Area needs to be normalized
        areas: torch.Tensor = torch.sqrt(cat([BoxListOps.normalized_area(boxlist) for boxlist in boxlists]))

        # Get the levels in the feature map by leveraging the fact that
        # the network always down-samples by a factor of 2 at each level.
        # Eqn.(1) in FPN paper
        # noinspection PyTypeChecker,PyUnresolvedReferences
        target_levels = (4 + torch.log2(areas + self.eps)).round().long() + 2  # +2 because P0 and P1 are not discarded
        target_levels = torch.clamp(target_levels, min=self.k_min, max=self.k_max)
        # Fix for using P5: https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/models/ufrcnn.py#L404
        target_levels[areas > 0.65] = 5

        return target_levels


class Pooler(torch.nn.Module, ABC):
    """
    Pooler for Detection with or without FPN.
    Note: this class has to be a Module because if cat_all_levels is True, then we have a Conv layer...
          (only used for relation prediction).
    """

    # Note: cat_all_levels is added for relationship detection.
    # We want to concatenate all levels, since detector is fixed in relation detection.
    # Without concatenation if there is any difference among levels, it can not be fine-tuned anymore.
    def __init__(
            self,
            output_size: tuple[int, ...],  # (d,) h, w
            scales: Sequence[Sequence[float]],
            sampling_ratio: int,
            n_dim: int,
            in_channels: int = 512,  # Only used if cat_all_levels
            cat_all_levels: bool = False,
            level_mapper=_LevelMapper(0, 6)
    ):
        """
        :param output_size: Output size for the pooled region.
        :param scales: Scales per dim for each Pooler.
        :param sampling_ratio: Sampling ratio for ROIAlign.
        :param cat_all_levels: whether the features should be extracted at all levels,
                               before being summarized by a convolutional layer.
        """
        super().__init__()
        self.n_dim = n_dim
        self.output_size = output_size
        self.cat_all_levels = cat_all_levels
        self.sampling_ratio = sampling_ratio
        self.roi_align_samplers = [self._build_roi_layer(dim_scales) for dim_scales in scales]

        self.map_levels = level_mapper
        # Reduce the channels
        if self.cat_all_levels:
            self.reduce_channel = build_conv3x3(
                n_dim,
                in_channels * len(self.roi_align_samplers),
                in_channels,
                dilation=1,
                stride=1,
                use_relu=True
            )

    @staticmethod
    def _convert_to_roi_format(boxes: list[BoxList]) -> torch.Tensor:
        """
        Concatenates proposals and add the batch index (that is required by the ROIAlign func) to the box tensor.
        :param boxes: zyxzyx mode
        :returns: ROIs tensor, ready for ROIAlign.
        """
        assert boxes and boxes[0].mode == BoxList.Mode.zyxzyx
        concat_boxes = cat([b.boxes for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def __call__(self, x: FeatureMaps, boxes: list[BoxList]) -> torch.Tensor:
        """
        :param x: feature maps for each level
        :param boxes: boxes to be used to perform the pooling operation.
        """
        rois = self._convert_to_roi_format(boxes)
        assert rois.size(0) > 0

        num_levels = len(self.roi_align_samplers)
        if num_levels == 1:
            return self.roi_align_samplers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        dtype, device = x[0].dtype, x[0].device
        final_channels = num_channels * num_levels if self.cat_all_levels else num_channels
        result = torch.zeros((num_rois, final_channels, *self.output_size), dtype=dtype, device=device)

        for level, (per_level_feature, sampler) in enumerate(zip(x, self.roi_align_samplers)):
            if self.cat_all_levels:
                # Store the pooler result only for all boxes for this level
                result[:, level * num_channels: (level + 1) * num_channels] = sampler(per_level_feature, rois).to(dtype)
            else:
                # Store the pooler result only for the boxes selected for this level
                # noinspection PyTypeChecker
                idx_in_level: torch.Tensor = levels == level
                if idx_in_level.sum() == 0:
                    continue
                result[idx_in_level] = sampler(per_level_feature, rois[idx_in_level]).to(dtype)

        if self.cat_all_levels:
            return self.reduce_channel(result)
        return result

    @abstractmethod
    def _build_roi_layer(self, scales: Sequence[float]) -> ROIAlign | ROIAlign3D:
        raise NotImplementedError


class Pooler2D(Pooler):
    def _build_roi_layer(self, scales: Sequence[float]) -> ROIAlign:
        return ROIAlign(
            self.output_size,
            spatial_scale=scales[0],  # Assume that in 2D scales[0] == scales[1]
            sampling_ratio=self.sampling_ratio
        )


class Pooler3D(Pooler):
    def _build_roi_layer(self, scales: Sequence[float]) -> ROIAlign3D:
        return ROIAlign3D(
            self.output_size,
            spatial_scale=scales[1],  # Assume that in 3D scales[1] == scales[2]
            spatial_scale_depth=scales[0],
            sampling_ratio=self.sampling_ratio,
        )


def build_pooler(cfg: CfgNode, head_name: str | ROIHeadName, anchor_strides: AnchorStrides) -> Pooler:
    """
    :param cfg: the config node.
    :param head_name: the head name needs to be one of ROIHeadName directly or a str.
    :param anchor_strides: Used to compute the scale for each level (scale = 1 / stride)
    """
    n_dim = cfg.INPUT.N_DIM
    assert n_dim in [2, 3]
    if isinstance(head_name, ROIHeadName):
        head_name = head_name.value

    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    resolution_depth = cfg.MODEL[head_name].POOLER_RESOLUTION_DEPTH
    scales = [[1 / stride[dim] for dim in range(n_dim)] for stride in anchor_strides]
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO

    if cfg.INPUT.N_DIM == 2:
        return Pooler2D(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            n_dim=n_dim
        )
    return Pooler3D(
        output_size=(resolution_depth, resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
        n_dim=cfg.INPUT.N_DIM
    )


def build_pooler_extra_args(
        cfg: CfgNode,
        head_name: str | ROIHeadName,
        anchor_strides: AnchorStrides,
        in_channels: int,
        cat_all_levels: bool
) -> Pooler:
    n_dim = cfg.INPUT.N_DIM
    assert n_dim in [2, 3]

    if isinstance(head_name, ROIHeadName):
        head_name = head_name.value

    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    resolution_depth = cfg.MODEL[head_name].POOLER_RESOLUTION_DEPTH
    scales = [[1 / stride[dim] for dim in range(n_dim)] for stride in anchor_strides]
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO

    if cfg.INPUT.N_DIM == 2:
        return Pooler2D(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
            n_dim=cfg.INPUT.N_DIM
        )
    return Pooler3D(
        output_size=(resolution_depth, resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
        in_channels=in_channels,
        cat_all_levels=cat_all_levels,
        n_dim=n_dim
    )
