"""
These annotations are used to store image captions.
Each caption describes the specified image and each image has at least 5 captions (some images have more).
See also the captioning task (https://cocodataset.org/#captions-2015).
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

from typing import TypedDict, Any

from .common import DatasetBase

Category = Any


class Annotation(TypedDict):
    id: int
    image_id: int
    caption: str


class Dataset(DatasetBase):
    annotations: list[Annotation]
    categories: list[Category]  # "categories" and "annotations" fields are required in all datasets


__all__ = ["Category", "Annotation", "Dataset"]
