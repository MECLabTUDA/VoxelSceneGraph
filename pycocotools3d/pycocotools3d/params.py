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

from abc import ABC
from typing import Literal

import numpy as np

from .IouType import IouType


# noinspection PyPep8Naming
class EvaluationParams(ABC):
    """Parameters for coco evaluation API."""

    def __init__(self, iouType: IouType = IouType.Segmentation):
        self.iouType = iouType  # IouType that should be evaluated

    imgIds: list[int]  # Ids of images
    catIds: list[int]  # Ids of categories
    iouThrs: np.ndarray  # Array of IoU thresholds as float
    iouThrsSummary: np.ndarray  # Array of IoU thresholds to display in the summary; needs to be a subset of iouThrs
    recThrs: np.ndarray  # Array of Recall thresholds as float
    maxDets: list[int]  # List of max number of (bb) detections
    areaRng: list[tuple[int, int]]  # List of area ranges [(min, max)]; "all" should always be in first position
    areaRngLbl: list[str]  # Labels for area ranges; must have the same length
    useCats: Literal[0, 1] = 1  # Whether to compute metrics for each category separately
    kpt_oks_sigmas: np.ndarray | None = None  # Optional array of floats used for keypoint detection


# noinspection PyPep8Naming
class DefaultParams(EvaluationParams):
    """Default parameters used in most/all 2D papers."""

    def __init__(self, iouType: IouType = IouType.Segmentation):
        super().__init__(iouType)
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble. The data point on arange is slightly larger than the true value.
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iouThrsSummary = np.array([.5, .75])
        self.recThrs = np.linspace(0., 1., int(np.round(1. / .01)) + 1, endpoint=True)
        self.useCats = 1

        match iouType:
            case IouType.Segmentation | IouType.BoundingBox:
                self.maxDets = [1, 10, 100]
                self.areaRng = [(0 ** 2, 100000 ** 2), (0 ** 2, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, 100000 ** 2)]
                self.areaRngLbl = ["all", "small", "medium", "large"]
                self.kpt_oks_sigmas = None
            case IouType.Keypoints:
                self.maxDets = [20]
                self.areaRng = [(0, 100000 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, 100000 ** 2)]
                self.areaRngLbl = ["all", "medium", "large"]
                self.kpt_oks_sigmas = np.array([
                    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
                ]) / 10.0
            case _:
                raise RuntimeError(f"iouType ({iouType}) not supported")


# noinspection PyPep8Naming
class Params3D(EvaluationParams):
    """Parameters adjusted for 3D detection."""

    def __init__(self, iouType: IouType = IouType.Segmentation):
        super().__init__(iouType)
        self.iouType = iouType
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble. The data point on arange is slightly larger than the true value.
        self.iouThrs = np.linspace(.1, .5, int(np.round((.5 - .1) / .05)) + 1, endpoint=True)
        self.iouThrsSummary = np.array([.1, .3, .5])
        self.recThrs = np.linspace(0., 1., int(np.round(1. / .01)) + 1, endpoint=True)
        self.useCats = 1
        self.kpt_oks_sigmas = None

        match iouType:
            case IouType.Segmentation | IouType.BoundingBox:
                self.maxDets = [25]
                voxel_vol = 0.9  # Approximate mm3 volume of voxel for head CT
                self.areaRng = [
                    (0, 100000 ** 2),
                    (0, 10000 / voxel_vol),
                    (10000 / voxel_vol, 50000 / voxel_vol),
                    (50000 / voxel_vol, 100000 ** 2)
                ]
                self.areaRngLbl = ["all", "small", "medium", "large"]
            case _:
                raise RuntimeError(f"iouType ({iouType}) not supported")
