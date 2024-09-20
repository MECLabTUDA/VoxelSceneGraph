"""
Each object instance annotation contains a series of fields,
including the category id and segmentation mask of the object.
The segmentation format depends on whether the instance represents a single object
(iscrowd=0 in which case polygons are used) or a collection of objects (iscrowd=1 in which case RLE is used).
Note that a single object (iscrowd=0) may require multiple polygons, for example if occluded.
Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people).
In addition, an enclosing bounding box is provided for each object
(box coordinates are measured from the top left image corner and are 0-indexed).
Finally, the categories field of the annotation structure stores the mapping of category id
to category and supercategory names. See also the detection task (https://cocodataset.org/#detection-2020).
Source: https://cocodataset.org/#format-data

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

from typing import Literal, TypedDict

import numpy as np
from typing_extensions import NotRequired

from .common import DatasetBase
from ..._masks import AnyRLE

Polygon = list[float]  # Polygons as list of coordinates (int or float)
SegmentationMask = np.ndarray  # dtype = np.uint8


class Category(TypedDict):
    id: int
    name: str
    supercategory: NotRequired[str]


class Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # Bounding box as [(z,) y, x, (d,) h, w]  # FIXME create warning / conversion method
    segmentation: NotRequired[Polygon | AnyRLE]
    area: float
    iscrowd: Literal[0, 1]
    ignore: NotRequired[bool]  # Used during evaluation to ignore some annotations


class MinimalPrediction(TypedDict):
    image_id: int
    category_id: int
    bbox: list[float]
    segmentation: NotRequired[Polygon | AnyRLE]
    score: float


class Dataset(DatasetBase):
    categories: list[Category]
    annotations: list[Annotation]


__all__ = ["Category", "Annotation", "MinimalPrediction", "Dataset"]
