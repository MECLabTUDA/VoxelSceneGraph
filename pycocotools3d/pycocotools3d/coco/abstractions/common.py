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

from typing import TypedDict

from typing_extensions import NotRequired


class Info(TypedDict):
    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created: str  # datetime


class License(TypedDict):
    id: int
    name: str
    url: str


class Image(TypedDict):
    id: int
    license: NotRequired[int]  # Index of license for the image
    file_name: str  # Path to image
    coco_url: NotRequired[str]  # Url for image, if it needs to be downloaded
    depth: NotRequired[int]  # Only for 3D
    height: int
    width: int
    flickr_url: NotRequired[str]
    coco_url: NotRequired[str]
    date_captured: NotRequired[str]  # datetime


class DatasetBase(TypedDict):
    info: NotRequired[Info]
    licenses: NotRequired[License]

    images: list[Image]
    # Note: "categories" field is required in all datasets


__all__ = ["Info", "License", "Image", "DatasetBase"]
