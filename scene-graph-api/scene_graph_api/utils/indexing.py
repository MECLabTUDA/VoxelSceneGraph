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

from enum import Enum
from typing import Iterable, Sequence

_SIZE_T = tuple[int, ...]


class FlipDim(Enum):
    """Enum used as argument for .flip methods (See AbstractSegmentationMask masks and BoxList)."""
    DEPTH = "depth"
    HEIGHT = "height"
    WIDTH = "width"

    def to_idx_from_last(self):
        """
        Converts the FlipDim to an integer i.e. negative index from last dim.
        E.g. WIDTh maps to -1, HEIGHT to -2, DEPTH to -3.
        """
        match self:
            case FlipDim.DEPTH:
                return -3
            case FlipDim.HEIGHT:
                return -2
            case FlipDim.WIDTH:
                return -1


def contiguous_mapping(indexes: Iterable[int], start: int = 0) -> dict[int, int]:
    """
    Compute an index mapping such that the new indices are contiguous and start at 0.
    All identity mappings are excluded from the returned mapping:
    e.g. if mapping[0] = 0, then this conversion will be excluded.
    :param indexes: an iterable containing all int indexes.
    :param start: start value for the indexing (typically 0 or 1).
    :returns: the mapping as a dict.
    """
    return {idx: i + start for i, idx in enumerate(indexes) if i + start != idx}


def sanitize_cropping_box(box: Sequence[float], current_size: _SIZE_T) -> Sequence[int]:
    """
    Given a bounding box used for cropping a BoxList or tensor, make sure that the box is not malformed.
    Return a box safe for indexing.
    :param box: coordinates for cropping
    :param current_size: current size of the BoxList or tensor to check for out-of-bounds issues.
    """
    n_dim = len(current_size)
    # noinspection PyTypeChecker
    box: list[int] = list(map(round, box))
    minimums = box[:n_dim]
    maximums = box[n_dim:]

    for i in range(n_dim):
        minimums[i] = min(max(minimums[i], 0), current_size[i] - 1)
        maximums[i] = min(max(maximums[i], 0), current_size[i] - 1)
        maximums[i] = max(maximums[i], minimums[i])

    return minimums + maximums
