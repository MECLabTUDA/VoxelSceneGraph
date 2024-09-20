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

__author__ = "asanner"

import copy
import json
import time
from logging import Logger

import numpy as np

import pycocotools3d.mask3d as mask_utils3d
from .abstractions import AnyAnnotation
from .abstractions.object_detection import Annotation, MinimalPrediction
from .coco_base import COCOBase
from .._masks import CompressedRLE


# noinspection PyPep8Naming
class COCO3d(COCOBase):
    def __init__(
            self, annotation_file: str | None = None, logger: Logger | None = None
    ):
        super().__init__(mask_utils3d, annotation_file, logger)

    def showAnns(self, anns: list[AnyAnnotation], draw_bbox: bool = False):
        raise NotImplementedError

    def annToRLE(self, ann: Annotation) -> CompressedRLE:
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 3D array)
        """
        if "segmentation" not in ann:
            raise ValueError("Annotation does not have a \"segmentation\" field.")

        t = self.imgs[ann["image_id"]]
        d, h, w = t["depth"], t["height"], t["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self._utils_module.frPyObjects(segm, d, h, w)
            rle = self._utils_module.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = self._utils_module.frPyObjects(segm, d, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    # noinspection DuplicatedCode
    def loadRes(self, resFile: str | np.ndarray | list[MinimalPrediction]) -> "COCO3d":
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
        res = COCO3d()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        self.logger.debug("Loading and preparing results...")
        tic = time.time()
        if isinstance(resFile, str):
            with open(resFile) as f:
                anns = json.load(f)
        elif isinstance(resFile, np.ndarray):
            # In this case, returns a list of FromNumpyAnnotation,
            # but adds the missing keys such that it becomes an Annotation
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert isinstance(anns, list), "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            "Results do not correspond to current coco set"

        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set([ann["image_id"] for ann in anns])
            res.dataset["images"] = [img for img in res.dataset["images"] if img["id"] in imgIds]
            for idx, ann in enumerate(anns):
                ann["id"] = idx + 1

        elif "bbox" in anns[0] and anns[0]["bbox"] != []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                bb = ann["bbox"]
                # z1, z2, y1, y2, x1, x2 = bb[0], bb[0] + bb[3], bb[1], bb[1] + bb[4], bb[2], bb[2] + bb[5]
                # TODO? No support for segmentation as polygon
                # if "segmentation" not in ann:
                #     ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]  # Tod adapt to 3D
                ann["area"] = bb[3] * bb[4] * bb[5]
                ann["id"] = idx + 1
                ann["iscrowd"] = 0

        elif "segmentation" in anns[0]:  # Segmentation only and no bbox -> infer bbox from segmentation
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = self._utils_module.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = self._utils_module.toBbox(ann["segmentation"])
                ann["id"] = idx + 1
                ann["iscrowd"] = 0

        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                s = ann["keypoints"]
                z = s[0::4]
                y = s[1::4]
                x = s[2::4]
                x0, x1, y0, y1, z0, z1 = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
                ann["area"] = (x1 - x0) * (y1 - y0) * (z1 - z0)
                ann["id"] = idx + 1
                ann["bbox"] = [z0, y0, x0, z1 - z0, y1 - y0, x1 - x0]

        self.logger.debug(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def loadNumpyAnnotations(self, data: np.ndarray) -> list[MinimalPrediction]:
        """
        Convert result data from a numpy array [Nx9] where each row contains {imageID,z1,y1,x1,d,h,w,score,class}
        :return: annotations (python nested list)
        """
        self.logger.debug("Converting ndarray to lists...")
        assert isinstance(data, np.ndarray)
        self.logger.debug(data.shape)
        assert (data.shape[1] == 9)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                self.logger.debug(f"{i}/{N}")
            ann += [{
                "image_id": int(data[i, 0]),
                "category_id": int(data[i, 8]),
                "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6]],
                "score": data[i, 7],
            }]
        return ann
