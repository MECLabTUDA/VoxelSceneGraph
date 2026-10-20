"""
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

from enum import Enum

import numpy as np


def bbox_2d(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Given a 2D numpy array, returns the bounding box of non-zero elements as (minimums, maxes)."""
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Cast np.int32 to int to avoid JSON serialization issues
    return (int(r_min), int(c_min)), (int(r_max), int(c_max))


def bbox_3d(arr: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Given a 3D numpy array, returns the bounding box of non-zero elements as (minimums, maxes)."""
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    r = np.any(arr, axis=(1, 2))
    c = np.any(arr, axis=(0, 2))
    z = np.any(arr, axis=(0, 1))

    r_min, r_max = np.where(r)[0][[0, -1]]
    c_min, c_max = np.where(c)[0][[0, -1]]
    z_min, z_max = np.where(z)[0][[0, -1]]

    # Cast np.int32 to int to avoid JSON serialization issues
    return (int(r_min), int(c_min), int(z_min)), (int(r_max), int(c_max), int(z_max))


class BoxFormat(Enum):
    """Enumeration of possible bounding box formats. Supports any number of dimensions despite what the names imply."""
    ZYXZYX = "ZYXZYX"
    XYZXYZ = "XYZXYZ"
    ZYXDHW = "ZYXDHW"
    XYZWHD = "XYZWHD"

    @staticmethod
    def format(
            box: list[float] | tuple[float, ...],
            input_format: BoxFormat,
            output_format: BoxFormat
    ) -> tuple[float, ...]:
        """
        Format a bounding box.
        :raises ValueError: if the length of the box is not even.
        """
        if len(box) % 2 == 1:
            raise ValueError("The length of the box should be even.")
        n_dim = len(box) // 2

        # Check whether we should reorder
        if input_format._is_depth_first() != output_format._is_depth_first():
            box = box[:n_dim][::-1] + box[n_dim:][::-1]

        # Check whether we should reformat
        if input_format._contains_length() and not output_format._contains_length():
            box = box[:n_dim] + [x + w for x, w in zip(box[:n_dim], box[n_dim:])]
        elif not input_format._contains_length() and output_format._contains_length():
            box = box[:n_dim] + [x2 - x1 for x1, x2 in zip(box[:n_dim], box[n_dim:])]

        return tuple(box)

    def _is_depth_first(self) -> bool:
        return self in [BoxFormat.ZYXZYX, BoxFormat.ZYXDHW]

    def _contains_length(self) -> bool:
        return self in [BoxFormat.ZYXDHW, BoxFormat.XYZWHD]
