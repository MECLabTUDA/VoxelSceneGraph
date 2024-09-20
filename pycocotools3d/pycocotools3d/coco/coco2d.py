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
from functools import reduce
from logging import Logger

import numpy as np

import pycocotools3d.mask as mask_utils
from .abstractions import AnyAnnotation
from .abstractions.object_detection import Annotation, MinimalPrediction
from .coco_base import COCOBase
from .._masks import CompressedRLE


# noinspection PyPep8Naming
class COCO(COCOBase):
    def __init__(
            self, annotation_file: str | None = None, logger: Logger | None = None
    ):
        super().__init__(mask_utils, annotation_file, logger)

    def showAnns(self, anns: list[AnyAnnotation], draw_bbox: bool = False):
        """
        Display the specified annotations.
        :param anns: annotations to display
        :param draw_bbox:
        """
        if len(anns) == 0:
            return 0

        if "caption" in anns[0]:
            for ann in anns:
                self.logger.info(ann["caption"])
            return

        if "segmentation" not in anns[0] and "keypoints" not in anns[0]:
            raise Exception("datasetType not supported")

        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):
                    # polygon
                    for yx_seg in ann["segmentation"]:
                        # Convert list of xyxyxy... polygons to yxyxyx... polygons
                        xy_seg = reduce(lambda a, b: a + b, [[y, x] for x, y in zip(yx_seg[::2], yx_seg[1::2])])
                        poly = np.array(xy_seg).reshape((len(xy_seg) // 2, 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
                else:
                    # mask
                    t = self.imgs[ann["image_id"]]
                    if isinstance(ann["segmentation"]["counts"], list):
                        rle = self._utils_module.frPyObjects([ann["segmentation"]], t["height"], t["width"])
                    else:
                        rle = [ann["segmentation"]]
                    m = self._utils_module.decode(rle)
                    img = np.ones((m.shape[0], m.shape[1], 3))
                    if ann["iscrowd"] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    else:
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, m * 0.5)))
            if "keypoints" in ann and isinstance(ann["keypoints"], list):
                # turn skeleton into zero-based index
                sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                kp = np.array(ann["keypoints"])
                y = kp[0::3]
                x = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        plt.plot(x[sk], y[sk], linewidth=3, color=c)
                plt.plot(x[v > 0], y[v > 0], "o", markersize=8, markerfacecolor=c, markeredgecolor="k",
                         markeredgewidth=2)
                plt.plot(x[v > 1], y[v > 1], "o", markersize=8, markerfacecolor=c, markeredgecolor=c,
                         markeredgewidth=2)

            if draw_bbox:
                bbox_y, bbox_x, bbox_h, bbox_w = ann["bbox"]
                poly = [[bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4, 2))
                polygons.append(Polygon(np_poly))
                color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
        ax.add_collection(p)

    def annToRLE(self, ann: Annotation) -> CompressedRLE:
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if "segmentation" not in ann:
            raise ValueError("Annotation does not have a \"segmentation\" field.")

        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self._utils_module.frPyObjects(segm, h, w)
            rle = self._utils_module.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = self._utils_module.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    # noinspection DuplicatedCode
    def loadRes(self, resFile: str | np.ndarray | list[MinimalPrediction]) -> "COCO":
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
        res = COCO()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        self.logger.debug("Loading and preparing results...")
        tic = time.time()
        if isinstance(resFile, str):
            with open(resFile) as f:
                anns = json.load(f)
        elif isinstance(resFile, np.ndarray):
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
                y1, y2, x1, x2 = bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[y1, x1, y1, x2, y2, x2, y2, x1]]
                ann["area"] = bb[2] * bb[3]
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
                y = s[0::3]
                x = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = idx + 1
                ann["bbox"] = [y0, x0, y1 - y0, x1 - x0]

        self.logger.debug(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def loadNumpyAnnotations(self, data: np.ndarray) -> list[MinimalPrediction]:
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,y1,x1,h,w,score,class}
        :return: annotations (python nested list)
        """
        self.logger.debug("Converting ndarray to lists...")
        assert isinstance(data, np.ndarray)
        self.logger.debug(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                self.logger.debug(f"{i}/{N}")
            ann += [{
                "image_id": int(data[i, 0]),
                "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                "score": data[i, 5],
                "category_id": int(data[i, 6]),
            }]
        return ann
