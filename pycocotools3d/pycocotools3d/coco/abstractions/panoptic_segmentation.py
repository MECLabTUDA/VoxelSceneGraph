"""
For the panoptic task (https://cocodataset.org/#panoptic-2020),
each annotation struct is a per-image annotation rather than a per-object annotation.
Each per-image annotation has two parts:
  1. a PNG that stores the class-agnostic image segmentation and
  2. a JSON struct that stores the semantic information for each image segment.

In more detail:
  1. To match an annotation with an image, use the image_id field (that is annotation.image_id==image.id).
  2. For each annotation, per-pixel segment ids are stored as a single PNG at annotation.file_name.
    The PNGs are in a folder with the same name as the JSON, i.e., annotations/name/ for annotations/name.json.
    Each segment (whether it's a stuff or thing segment) is assigned a unique id.
    Unlabeled pixels (void) are assigned a value of 0.
    Note that when you load the PNG as an RGB image, you will need to compute the ids via ids=R+G*256+B*256^2.
  3. For each annotation, per-segment info is stored in annotation.segments_info.
     segment_info.id stores the unique id of the segment and is used to retrieve
     the corresponding mask from the PNG (ids==segment_info.id).
     category_id gives the semantic category and iscrowd indicates the segment encompasses a group of objects
     (relevant for thing categories only). The bbox and area fields provide additional info about the segment.
  4. The COCO panoptic task has the same thing categories as the detection task,
     whereas the stuff categories differ from those in the stuff task
     (for details see the panoptic evaluation page https://cocodataset.org/#panoptic-eval).
     Finally, each category struct has two additional fields:
     - isthing that distinguishes stuff
     - thing categories and color that is useful for consistent visualization.
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

from typing_extensions import NotRequired

from .common import DatasetBase


class Category(TypedDict):
    id: int
    name: str
    supercategory: str
    isthing: Literal[0, 1]
    color: tuple[int, int, int]  # RGB color for display; Compute the segment id=R+G*256+B*256^2


class SegmentInfo(TypedDict):
    id: int
    category_id: int
    area: NotRequired[int]
    bbox: list[float]  # Bounding box as [(z,) y, x, (d,) h, w]  # FIXME create warning / conversion method
    iscrowd: Literal[0, 1]


class Annotation(TypedDict):
    # One annotation per image (in contrast to object detection)
    image_id: int
    file_name: str
    segments_info: list[SegmentInfo]


class Dataset(DatasetBase):
    categories: list[Category]
    annotations: list[Annotation]


__all__ = ["Category", "SegmentInfo", "Annotation", "Dataset"]
