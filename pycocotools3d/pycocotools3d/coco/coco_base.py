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

import itertools
import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from logging import Logger, getLogger
from typing import Iterable, Any
from urllib.request import urlretrieve

import numpy as np

from .abstractions import AnyDataset, AnyAnnotation, AnyCategory
from .abstractions.common import Image
from .abstractions.object_detection import MinimalPrediction
from .._masks import CompressedRLE


# Interface for accessing the Microsoft COCO dataset.
# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.
# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).
# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load annotations (anns) with the specified ids.
#  loadCats   - Load categories (cats) with the specified ids.
#  loadImgs   - Load images (imgs) with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help for each function can be accessed by typing: "help COCO>function".
# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]


def _preprocess_input(obj: Any) -> list:
    """Check for array-likeness and wrap as needed to always return an iterable."""
    if obj is None:
        return []
    # Check array-likeness
    return obj if isinstance(obj, Iterable) and not isinstance(obj, str) else [obj]


# noinspection PyPep8Naming
class COCOBase(ABC):
    def __init__(
            self,
            mask_utils_module,
            annotation_file: AnyDataset | str | None = None,
            logger: Logger | None = None
    ):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param mask_utils_module: either mask or mask3d module
        :param annotation_file: either the dataset dict, a path to load or None to only create a COCOBase instance.
        """
        if logger is None:
            logger = getLogger(__file__)
        self.logger = logger

        self._utils_module = mask_utils_module
        # load dataset
        self.dataset: AnyDataset = dict()
        self.anns: dict[int, AnyAnnotation] = dict()
        self.cats: dict[int, AnyCategory] = dict()
        self.imgs: dict[int, Image] = dict()
        self.imgToAnns: dict[int, list[AnyAnnotation]] = defaultdict(list)
        self.catToImgsIds: dict[int, list[int]] = defaultdict(list)
        if annotation_file is not None:
            if isinstance(annotation_file, str):
                self.logger.debug("Loading annotations into memory...")
                tic = time.time()
                with open(annotation_file, "r") as f:
                    dataset = json.load(f)
                assert isinstance(dataset, dict), f"Annotation file format {type(dataset)} not supported"
                self.logger.debug(f"Done (t={time.time() - tic:0.2f}s)")
            else:
                dataset = annotation_file
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        self.logger.debug("Creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

        self.logger.debug("Index created!")

        # Create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgsIds = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """Print information about the annotation file."""
        for key, value in self.dataset["info"].items():
            self.logger.info(f"{key}: {value}")

    def getAnnIds(
            self,
            imgIds: int | list[int] | None = None,
            catIds: int | list[int] | None = None,
            areaRng: float | list[float] | None = None,
            iscrowd: bool | None = None
    ) -> list[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds     : get anns for given image
        :param catIds     : get anns for given category
        :param areaRng    : get anns for given area range (e.g. [0 inf])
        :param iscrowd    : get anns for given crowd label (False or True; or None to get all)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = _preprocess_input(imgIds)
        catIds = _preprocess_input(catIds)
        areaRng = _preprocess_input(areaRng)

        if len(imgIds) == 0:
            anns: list[AnyAnnotation] = self.dataset["annotations"]
        else:
            lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
            anns = list(itertools.chain.from_iterable(lists))
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann["category_id"] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if areaRng[0] < ann["area"] < areaRng[1]]

        if iscrowd is not None:
            return [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        return [ann["id"] for ann in anns]

    def getCatIds(
            self,
            catNms: str | list[str] | None = None,
            supNms: str | list[str] | None = None,
            catIds: str | list[int] | None = None
    ) -> list[int]:
        """
        Filtering parameters. Default skips that filter.
        :param catNms: get category for given category names
        :param supNms: get categories for given super-category names,
                       should only be not None if categories have a "supercategory" field
        :param catIds: get categories for given ids
        :return: integer array of category ids
        """
        catNms = _preprocess_input(catNms)
        supNms = _preprocess_input(supNms)
        catIds = _preprocess_input(catIds)

        cats: list[AnyCategory] = self.dataset["categories"]
        cats = cats if len(catNms) == 0 else [cat for cat in cats if cat["name"] in catNms]
        cats = cats if len(supNms) == 0 else [cat for cat in cats if cat["supercategory"] in supNms]
        cats = cats if len(catIds) == 0 else [cat for cat in cats if cat["id"] in catIds]
        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds: list[int] | None = None, catIds: list[int] | None = None) -> list[int]:
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds: get imgs for given ids
        :param catIds: get imgs with all given categories
        :return: integer array of img ids
        """
        imgIds = _preprocess_input(imgIds)
        catIds = _preprocess_input(catIds)

        if len(imgIds) == 0:
            return list(self.imgs.keys())

        else:
            ids = set()
            for i, catId in enumerate(catIds):
                if i == 0:
                    ids = set(self.catToImgsIds[catId])
                else:
                    ids &= set(self.catToImgsIds[catId])
        return list(ids)

    def loadAnns(self, ids: Iterable[int] | int) -> list[AnyAnnotation]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if isinstance(ids, Iterable):
            return [self.anns[id_] for id_ in ids]
        return [self.anns[ids]]

    def loadCats(self, ids: Iterable[int] | int) -> list[AnyCategory]:
        """
        Load categories with the specified ids.
        :param ids: integer ids specifying categories
        :return: loaded category objects
        """
        if isinstance(ids, Iterable):
            return [self.cats[id_] for id_ in ids]
        return [self.cats[ids]]

    def loadImgs(self, ids: Iterable[int] | int) -> list[Image]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if isinstance(ids, Iterable):
            return [self.imgs[id_] for id_ in ids]
        return [self.imgs[ids]]

    @abstractmethod
    def showAnns(self, anns: list[AnyAnnotation], draw_bbox: bool = False):
        """
        Display the specified annotations.
        :param anns: annotations to display
        :param draw_bbox:
        """
        raise NotImplementedError

    @abstractmethod
    def loadRes(self, resFile: str | np.ndarray | list[MinimalPrediction]) -> "COCOBase":
        """
        Load result file and return a result api object.
        The resFile should contain the following fields:
          - id
          - image_id
          - category_id
          - caption | bbox | segmentation | keypoints (mutually exclusive)
        :param resFile: file name of result file | np.ndarray to convert | annotation dict
        :return: result api object
        """
        raise NotImplementedError

    def download(self, tarDir: str, imgIds: list[int] | None = None):
        """
        Download COCO images from mscoco.org server.
        :param tarDir: COCO results directory name
        :param imgIds: images to be downloaded
        :return: -1 on error else None
        """
        if imgIds is None or len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)

        if not os.path.exists(tarDir):
            os.makedirs(tarDir)

        for i, img in enumerate(imgs):
            tic = time.time()
            f_name = os.path.join(tarDir, img["file_name"])
            if not os.path.exists(f_name):
                urlretrieve(img["coco_url"], f_name)
            self.logger.info(f"Downloaded {i}/{len(imgs)} images (t={time.time() - tic:0.1f}s)")

    @abstractmethod
    def loadNumpyAnnotations(self, data: np.ndarray) -> list[MinimalPrediction]:
        """
        Convert result data from a numpy array [Nx(1+2*n_dim+2))]
        where each row contains {imageID,x1,y1,(z1,)w,h,(d,)score,class}
        :return: annotations (python nested list)
        """
        raise NotImplementedError

    @abstractmethod
    def annToRLE(self, ann: AnyAnnotation) -> CompressedRLE:
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        raise NotImplementedError

    def annToMask(self, ann: AnyAnnotation) -> np.ndarray:
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = self._utils_module.decode(rle)
        return m
