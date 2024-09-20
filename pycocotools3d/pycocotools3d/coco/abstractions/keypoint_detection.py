"""
A keypoint annotation contains all the data of the object annotation and two additional fields.
First, "keypoints" is a length 3k array where k is the total number of keypoints defined for the category.
Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0),
v=1: labeled but not visible, and v=2: labeled and visible.
A keypoint is considered visible if it falls inside the object segment.
"num_keypoints" indicates the number of labeled keypoints (v>0) for a given object
(many objects, e.g. crowds and small objects, will have num_keypoints=0).
Finally, for each category, the categories struct has two additional fields:
- "keypoints," which is a length k array of keypoint names, and
- "skeleton", which defines connectivity via a list of keypoint edge pairs and is used for visualization.
Currently, keypoints are only labeled for the person category (for most medium/large non-crowd person instances).
See also the keypoint task (https://cocodataset.org/#keypoints-2020).
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

from typing import Any

from typing_extensions import NotRequired

from .common import DatasetBase
from .object_detection import Category as ObjectDetectionCategory, Annotation as ObjectDetectionAnnotation


class Category(ObjectDetectionCategory):
    keypoints: str
    skeleton: NotRequired[Any]  # Only required for displaying keypoints


class Annotation(ObjectDetectionAnnotation):
    keypoints: list[float]  # [(z0, ) y0, x0, v0, ...]  # FIXME create warning / conversion method
    num_keypoints: int


class Dataset(DatasetBase):
    categories: list[Category]
    annotations: list[Annotation]


__all__ = ["Category", "Annotation", "Dataset"]
