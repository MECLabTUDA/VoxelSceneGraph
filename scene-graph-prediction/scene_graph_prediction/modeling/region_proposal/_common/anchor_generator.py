# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from functools import reduce
from itertools import chain
from typing import Sequence

import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, ImageList, BufferList
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.region_proposal import AnchorGenerator as _AbstractAnchorGenerator, ImageAnchors, RawAnchorGrid

_ANCHOR_T = np.ndarray  # 2 * n_dim and in zyxzyx format
_ANCHORS_T = np.ndarray  # n x 2 * n_dim and in zyxzyx format


# Note: An anchor represents a window around (0, 0, 0) e.g. (-180., -180., -180.,  180.,  180.,  180.).
#       It will then be shifted to all position in the image to try all locations.


class AnchorGenerator(_AbstractAnchorGenerator):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    Note: support 2D and 3D (see function _ratio_enum).
    Note: There is currently no support for only defining custom anchors and no anchor size.
    Note: needs to be a Module because of the BufferList...
    """

    def __init__(
            self,
            n_dim: int,
            anchor_strides: AnchorStrides,
            sizes: tuple[float, ...] | tuple[tuple[float, ...], ...] = (128, 256, 512),
            aspect_ratios: tuple[float, ...] = (1., .5, 2.),
            depths: tuple[float, ...] | tuple[tuple[float, ...], ...] = (2, 4, 8),
            custom_anchors: tuple[tuple[float, ...], ...] | tuple[tuple[tuple[float, ...], ...], ...] = tuple(),
            straddle_thresh: int = 0,  # Tolerance (in px) for BoundingBox visibility when partially out of frame
    ):
        """
        Either provide one anchor_strides and then all sizes will be used for this stride.
        Or if multiple anchor_strides are provided, then:
          - the length of anchor_strides should be equal to the length of sizes
          - sizes should contain a size for each anchor_stride
        """

        super().__init__()
        assert n_dim in [2, 3]
        self.n_dim = n_dim

        # The next section is ugly:
        #  We're basically interpreting whether we have:
        #  1) RPN or a FPN
        #  2) Then a size or list of sizes per-level
        #  3) Optionally, the same thing but with pre-formed anchors
        if len(anchor_strides) == 1:
            # RPN mode
            if n_dim == 2:
                # 2D
                ratio_anchors = [_generate_anchors2d(anchor_strides[0][0], sizes, aspect_ratios).float()]
            else:
                # 3D
                ratio_anchors = [
                    _generate_anchors3d(
                        anchor_strides[0][1], anchor_strides[0][0], sizes, depths, aspect_ratios
                    ).float()
                ]
            if custom_anchors:
                if n_dim == 2:
                    # 2D
                    centered_custom_anchors = [
                        _stride_center_manual_anchors2d(
                            anchor_strides[0][0],
                            custom_anchors if isinstance(custom_anchors[0], Sequence) else (custom_anchors,)
                        ).float()
                    ]
                else:
                    # 3D
                    centered_custom_anchors = [
                        _stride_center_manual_anchors3d(
                            anchor_strides[0][1], anchor_strides[0][0],
                            custom_anchors if isinstance(custom_anchors[0], Sequence) else (custom_anchors,)
                        ).float()
                    ]
            else:
                centered_custom_anchors = []
        else:
            # FPN mode
            if len(anchor_strides) != len(sizes):
                raise RuntimeError(f"FPN/Multi-scale RPN should have "
                                   f"#anchor_strides ({len(anchor_strides)}) == #sizes ({len(sizes)})")
            if n_dim == 3:
                if len(anchor_strides) != len(depths):
                    raise RuntimeError(f"FPN/Multi-scale RPN should have "
                                       f"#anchor_strides ({len(anchor_strides)}) == #depths ({len(depths)})")

            if n_dim == 2:
                # 2D
                ratio_anchors = [
                    _generate_anchors2d(
                        anchor_stride[0],
                        size if isinstance(size, Sequence) else (size,),
                        aspect_ratios
                    ).float()
                    for anchor_stride, size in zip(anchor_strides, sizes)
                ]
            else:
                # 3D
                ratio_anchors = [
                    _generate_anchors3d(
                        anchor_stride[1],
                        anchor_stride[0],
                        size if isinstance(size, Sequence) else (size,),
                        depth if isinstance(depth, Sequence) else (depth,),
                        aspect_ratios
                    ).float()
                    for anchor_stride, size, depth in zip(anchor_strides, sizes, depths)
                ]

            if custom_anchors:
                if n_dim == 2:
                    # 2D
                    centered_custom_anchors = [
                        _stride_center_manual_anchors2d(
                            anchor_stride[0], anchors if isinstance(anchors[0], Sequence) else (anchors,)
                        ).float()
                        for anchor_stride, anchors in zip(anchor_strides, custom_anchors)
                    ]
                else:
                    # 3D
                    centered_custom_anchors = [
                        _stride_center_manual_anchors3d(
                            anchor_stride[1],
                            anchor_stride[0],
                            anchors if isinstance(anchors[0], Sequence) else (anchors,)
                        ).float()
                        for anchor_stride, anchors in zip(anchor_strides, custom_anchors)
                    ]
            else:
                centered_custom_anchors = []

        # Concatenate ratio anchors and custom anchors
        if centered_custom_anchors:
            cell_anchors = [torch.cat([ra, ca]) for ra, ca in zip(ratio_anchors, centered_custom_anchors)]
        else:
            cell_anchors = ratio_anchors

        assert len(anchor_strides) == len(cell_anchors), "Mismatch between stride and anchors lengths..."
        assert len({len(a) for a in cell_anchors}) == 1, "There can only be a fixed number of anchors per level."
        self.strides = anchor_strides
        # Note: cell_anchors don't need to be part of the model state per se,
        #       but having then in a BufferList ensures that they are automatically on the right device
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_level(self) -> int:
        # Note: there should at least be one level and all levels need to have the same number of anchors.
        return len(self.cell_anchors[0])

    def forward(self, image_list: ImageList, feature_maps: FeatureMaps) -> list[ImageAnchors]:
        """Given an ImageList, return the anchors (per feature level) for each image BxFeatLvlsX2*n_dim."""
        assert image_list.n_dim == self.n_dim
        grid_sizes = [feature_map.shape[2:] for feature_map in feature_maps]  # Exclude batch and channels
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        anchors = []
        for size in image_list.image_sizes:
            anchors_in_image = []
            for lvl, anchors_per_feature_map in enumerate(anchors_over_all_feature_maps):
                boxlist = BoxList(anchors_per_feature_map, size, mode=BoxList.Mode.zyxzyx)
                self._add_visibility_to(boxlist)
                # Add anchor level field
                boxlist.add_field(
                    BoxList.PredictionField.ANCHOR_LVL,
                    torch.full((len(boxlist),), lvl, dtype=torch.long, device=boxlist.boxes.device)
                )
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors

    def _grid_anchors(self, grid_sizes: Sequence[tuple[int, ...]]) -> list[RawAnchorGrid]:
        """
        Anchors as (1 -1, 2 * n_dim) zyxzyx tensors.
        For each anchor (window), shift it to all possible positions in the image.
        :param grid_sizes: zyx ordered.
        """
        anchors = []
        assert len(grid_sizes) == len(self.strides)
        for dhw_size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            device = base_anchors.device
            # Create all possible shifts for each dim
            zyx_shifts = [
                torch.arange(0, dhw_size[dim] * stride[dim], step=stride[dim], dtype=torch.float32, device=device)
                for dim in range(self.n_dim)
            ]
            # Create all possible shifts combinations between dim
            zyx_shifts = torch.meshgrid(*zyx_shifts, indexing="ij")
            zyx_shifts = tuple(shift.reshape(-1) for shift in zyx_shifts)
            # Stack them so that the shift applies both to start and end of anchor
            zyx_shifts = torch.stack(zyx_shifts + zyx_shifts, dim=1)
            # Create all possible combination between shifts and anchors
            anchors.append(
                (zyx_shifts.view(-1, 1, 2 * self.n_dim) + base_anchors.view(1, -1, 2 * self.n_dim))
                .reshape(-1, 2 * self.n_dim)
            )

        return anchors

    def _add_visibility_to(self, boxlist: BoxList):
        """
        Adds a "visibility" field to a BoxList that denotes
        whether a Bounding Box is fully visible (± self.straddle_thresh px).
        """
        assert boxlist.n_dim == self.n_dim
        anchors = boxlist.boxes
        if self.straddle_thresh >= 0:
            indexes_inside = reduce(lambda a, b: a & b, chain(
                (anchors[..., dim] >= -self.straddle_thresh for dim in range(self.n_dim)),
                (anchors[..., dim + self.n_dim] < boxlist.size[dim] + self.straddle_thresh for dim in range(self.n_dim))
            ))
        else:
            indexes_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=anchors.device)
        boxlist.add_field(BoxList.PredictionField.VISIBILITY, indexes_inside)


def build_anchor_generator(cfg: CfgNode, anchor_strides: AnchorStrides) -> AnchorGenerator:
    n_dim = cfg.INPUT.N_DIM
    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS
    anchor_depths = cfg.MODEL.RPN.ANCHOR_DEPTHS
    custom_anchors = cfg.MODEL.RPN.CUSTOM_ANCHORS
    straddle_thresh = cfg.MODEL.RPN.STRADDLE_THRESH

    if len(anchor_strides) > 1:
        assert len(anchor_strides) == len(anchor_sizes), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    if n_dim == 3 and len(anchor_strides) > 1:
        assert len(anchor_strides) == len(anchor_depths), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_DEPTHS)"
    if custom_anchors and len(anchor_strides) > 1:
        assert len(anchor_strides) == len(custom_anchors), "FPN should have len(ANCHOR_STRIDE) == len(CUSTOM_ANCHORS)"

    return AnchorGenerator(
        n_dim,
        anchor_strides,
        anchor_sizes,
        aspect_ratios,
        anchor_depths,
        custom_anchors,
        straddle_thresh
    )


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


def _lengths_centers_to_anchors(lengths: list[np.ndarray], center: Sequence[float]) -> _ANCHOR_T:
    """
    Given a vector of lengths (d_arr, h_arr, w_arr) around a center (z_ctr, y_ctr, x_ctr),
    output a set of anchors (windows).
    Note: can be reversed with _lengths_centers.
    Note: works for ND.
    :param lengths: dhw ordered. All arrays need to have the same length.
    :param center: zyx ordered
    :returns: anchors as array of (z1, y1, x1, z2, y2, x2).
    """
    assert len(lengths) == len(center)
    n_dim = len(lengths)
    lengths = [length[:, np.newaxis] for length in lengths]

    return np.hstack(
        tuple(center[dim] - 0.5 * (lengths[dim] - 1) for dim in range(n_dim)) +  # Starts zyx ordered
        tuple(center[dim] + 0.5 * (lengths[dim] - 1) for dim in range(n_dim))  # Ends zyx ordered
    )


def _lengths_centers(anchor: _ANCHOR_T) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Reverse operation from _lengths_centers_to_anchors.
    Note: works for ND.
    :returns: ((d, h, w), (z_ctr, y_ctr, x_ctr)) for an anchor.
    """
    n_dim = len(anchor) // 2
    lengths = tuple(anchor[n_dim + dim] - anchor[dim] + 1 for dim in range(n_dim))
    centers = tuple(anchor[dim] + 0.5 * (lengths[dim] - 1) for dim in range(n_dim))
    return lengths, centers


def _ratio_enum2d(anchor: _ANCHOR_T, ratios: np.ndarray) -> _ANCHORS_T:
    """
    Enumerate a set of anchors for each aspect ratio with respect to a single anchor.
    I.e. computes the center of the anchors and produces new anchors with varying aspect ratios with the same center.
    Note: preserves anchor area when reshaping.
    """

    (h, w), (cy, cx) = _lengths_centers(anchor)

    size = w * h
    size_ratios = size / ratios
    ws = np.sqrt(size_ratios)
    hs = ws * ratios

    return _lengths_centers_to_anchors([hs, ws], [cy, cx])


def _ratio_enum3d(anchor: _ANCHOR_T, ratios: np.ndarray, ) -> _ANCHORS_T:
    """
    Enumerate a set of anchors for each aspect ratio with respect to a single anchor.
    I.e. computes the center of the anchors and produces new anchors with varying aspect ratios with the same center.
    Note: preserves anchor volume when reshaping.
    """

    (d, h, w), (cz, cy, cx) = _lengths_centers(anchor)

    size = w * h
    size_ratios = size / ratios
    ws = np.sqrt(size_ratios)
    hs = ws * ratios
    ds = np.tile(d, ratios.shape[0])

    return _lengths_centers_to_anchors([ds, hs, ws], [cz, cy, cx])


def _scale_enum2d(anchors: _ANCHORS_T, scales: np.ndarray) -> _ANCHORS_T:
    """Enumerate a set of anchors for each scale with respect to an anchor."""
    lengths, centers = _lengths_centers(anchors)
    # Centers don't need to be rescaled because boxes should be (0,)*n_dim centered
    lengths: list[np.ndarray] = [np.array(length * scales) for length in lengths]
    return _lengths_centers_to_anchors(lengths, list(centers))


def _scale_enum3d(anchors: _ANCHORS_T, scales: np.ndarray, scales_depth: np.ndarray) -> _ANCHORS_T:
    """Enumerate a set of anchors for each scale with respect to an anchor."""
    (d, h, w), (cz, cy, cx) = _lengths_centers(anchors)
    # Centers don't need to be rescaled because boxes should be (0,)*n_dim centered
    lengths: list[np.ndarray] = [d * scales_depth, h * scales, w * scales]
    return _lengths_centers_to_anchors(lengths, [cz, cy, cx])


def _generate_anchors2d(stride: int,
                        sizes: tuple[float, ...],
                        aspect_ratios: tuple[float, ...]) -> torch.Tensor:
    """
    Generates a matrix of anchor boxes in (y1, x1, y2, x2) format.
    Anchors are centered on stride / 2, have ~sqrt areas of the specified sizes, and aspect ratios as given.
    """
    # Generate anchor (reference) windows by enumerating aspect ratios*scales
    scales = np.array(sizes, dtype=float) / stride
    aspect_ratios = np.array(aspect_ratios, dtype=np.float32)
    anchor = np.array([0, 0, stride - 1, stride - 1], dtype=np.float32)
    anchors = _ratio_enum2d(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum2d(anchor, scales) for anchor in anchors])

    return torch.from_numpy(anchors).squeeze()


def _generate_anchors3d(stride: int,
                        stride_depth: int,
                        sizes: tuple[float, ...],
                        depths: tuple[float, ...],
                        aspect_ratios: tuple[float, ...]) -> torch.Tensor:
    """
    Generates a matrix of anchor boxes in (z1, y1, x1, z2, y2, x2) format.
    Anchors are centered on stride / 2, have ~sqrt areas of the specified sizes, and aspect ratios as given.
    Note: one depth per size.
    """
    assert len(sizes) == len(depths)
    # Generate anchor (reference) windows by enumerating aspect ratios*scales
    scales = np.array(sizes, dtype=float) / stride
    scales_depth = np.array(depths, dtype=float) / stride_depth
    aspect_ratios = np.array(aspect_ratios, dtype=np.float32)
    anchor = np.array([0, 0, 0, stride_depth - 1, stride - 1, stride - 1], dtype=np.float32)
    anchors = _ratio_enum3d(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum3d(anchor, scales, scales_depth) for anchor in anchors])

    return torch.from_numpy(anchors).squeeze()


def _stride_center_manual_anchors2d(stride: int, anchor_lengths: Sequence) -> torch.Tensor:
    """
    Anchor lengths can be provided manually in the config.
    However, we still need to compute the appropriate anchor center.
    :param stride:
    :param anchor_lengths: list of shape n x 2, hw ordered.
    :returns: centered anchors.
    """
    anchor_lengths = np.array(anchor_lengths)
    center = (stride - 1) / 2, (stride - 1) / 2
    lengths_per_dim = [anchor_lengths[:, 0], anchor_lengths[:, 1]]
    anchors = _lengths_centers_to_anchors(lengths_per_dim, center)
    return torch.from_numpy(anchors)


def _stride_center_manual_anchors3d(stride: int, stride_depth: int, anchor_lengths: Sequence) -> torch.Tensor:
    """
    Anchor lengths can be provided manually in the config.
    However, we still need to compute the appropriate anchor center.
    :param stride:
    :param anchor_lengths: list of shape n x 3, dhw ordered.
    :returns: centered anchors.
    """
    anchor_lengths = np.array(anchor_lengths)
    center = (stride_depth - 1) / 2, (stride - 1) / 2, (stride - 1) / 2
    lengths_per_dim = [anchor_lengths[:, 0], anchor_lengths[:, 1], anchor_lengths[:, 2]]
    anchors = _lengths_centers_to_anchors(lengths_per_dim, center)
    return torch.from_numpy(anchors)
