"""
Custom formatting of relations.

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

from typing import TypedDict

from typing_extensions import NotRequired

from .common import DatasetBase
from .object_detection import Category as ObjectDetectionCategory, Annotation as ObjectDetectionAnnotation
from .panoptic_segmentation import Category as PanopticSegmentationCategory, \
    Annotation as PanopticSegmentationAnnotation


class PredicateCategory(TypedDict):
    id: int
    name: str
    supercategory: NotRequired[str]


SSGCategory = ObjectDetectionCategory

PSGCategory = PanopticSegmentationCategory

# Note about difference in format between SSG and PSG relation annotation:
# With PSG, we only have one Annotation object per image; so we can cram the list of relations directly there
# WIth SSG, we have multiple Annotation objects per image; so we need a separate list of relations

SSGRelationAnnotation = tuple[int, int, int, int]  # image id, subject id, object id, rel id

PSGRelationAnnotation = tuple[int, int, int]  # subject id, object id, rel id

SSGAnnotation = ObjectDetectionAnnotation


class PSGAnnotation(PanopticSegmentationAnnotation):
    relations: list[PSGRelationAnnotation]


class SSGDataset(DatasetBase):
    categories: list[SSGCategory]
    annotations: list[SSGAnnotation]
    predicates: list[PredicateCategory]
    relations: list[SSGRelationAnnotation]


class PSGDataset(DatasetBase):
    categories: list[PSGCategory]
    annotations: list[PSGAnnotation]
    predicates: list[PredicateCategory]


__all__ = [
    "PredicateCategory",
    "SSGCategory", "PSGCategory",
    "SSGAnnotation", "PSGAnnotation",
    "SSGDataset", "PSGDataset",
    "SSGRelationAnnotation", "PSGRelationAnnotation"
]
