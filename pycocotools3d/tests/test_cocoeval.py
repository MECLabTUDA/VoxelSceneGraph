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

import unittest
from copy import deepcopy

from pycocotools3d import IouType
from pycocotools3d.coco import COCO, COCO3d
from pycocotools3d.cocoeval import COCOeval, StatsSummary
from pycocotools3d.params import DefaultParams, Params3D


# noinspection DuplicatedCode
class TestCocoEval(unittest.TestCase):
    ANN_PATH_OBJ_2D = "./test_annotations/dummy_2d_obj_detec.json"
    ANN_PATH_SEG_2D = "./test_annotations/dummy_2d_seg.json"
    ANN_PATH_OBJ_3D = "./test_annotations/dummy_3d_obj_detec.json"
    ANN_PATH_SEG_3D = "./test_annotations/dummy_3d_seg.json"

    def check_perfect_stats(self, stats: StatsSummary, expected: list[float]):
        self.assertEqual(len(expected), len(stats))
        for (k, v), exp in zip(stats.items(), expected):
            with self.subTest(key=k):
                self.assertAlmostEqual(exp, v)

    def test_perfect_object_detection_2d(self):
        coco = COCO(self.ANN_PATH_OBJ_2D)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        pred = coco.loadRes(pred_data)
        coco_eval = COCOeval(coco, pred, IouType.BoundingBox, DefaultParams(IouType.BoundingBox))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # Some metrics are -1 because of the size of the objects (area categories)
        self.check_perfect_stats(coco_eval.stats, [1] * 5 + [-1] * 4 + [1] * 7 + [-1] * 4)

    def test_perfect_segmentation_2d(self):
        coco = COCO(self.ANN_PATH_SEG_2D)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        pred = coco.loadRes(pred_data)
        coco_eval = COCOeval(coco, pred, IouType.BoundingBox, DefaultParams(IouType.BoundingBox))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.check_perfect_stats(coco_eval.stats, [1] * 5 + [-1] * 4 + [1] * 7 + [-1] * 4)

    def test_perfect_object_detection_3d(self):
        coco = COCO3d(self.ANN_PATH_OBJ_3D)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        pred = coco.loadRes(pred_data)
        coco_eval = COCOeval(coco, pred, IouType.BoundingBox, Params3D(IouType.BoundingBox))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.check_perfect_stats(coco_eval.stats, [-1] + [1] * 6 + [-1] * 6 + [1] * 7 + [-1] * 6)

    def test_perfect_segmentation_3d(self):
        coco = COCO3d(self.ANN_PATH_SEG_3D)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        pred = coco.loadRes(pred_data)
        coco_eval = COCOeval(coco, pred, IouType.BoundingBox, Params3D(IouType.BoundingBox))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.check_perfect_stats(coco_eval.stats, [-1] + [1] * 6 + [-1] * 6 + [1] * 7 + [-1] * 6)
