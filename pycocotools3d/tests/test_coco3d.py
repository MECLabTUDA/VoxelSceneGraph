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

from pycocotools3d.coco import COCO3d


# noinspection DuplicatedCode
class TestCoco3d(unittest.TestCase):
    ANN_PATH_OBJ = "./test_annotations/dummy_3d_obj_detec.json"
    ANN_PATH_SEG = "./test_annotations/dummy_3d_seg.json"

    def test___init___object_detection(self):
        coco = COCO3d(self.ANN_PATH_OBJ)
        self.assertEqual(1, len(coco.anns))
        self.assertEqual(3, len(coco.cats))
        self.assertEqual(2, len(coco.imgs))

    def test_loadRes_object_detection(self):
        coco = COCO3d(self.ANN_PATH_OBJ)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        res = coco.loadRes(pred_data)
        self.assertEqual(coco.cats, res.cats)
        self.assertEqual(coco.imgs, res.imgs)

    def test___init___segmentation(self):
        coco = COCO3d(self.ANN_PATH_SEG)
        self.assertEqual(1, len(coco.anns))
        self.assertEqual(3, len(coco.cats))
        self.assertEqual(2, len(coco.imgs))

    def test_loadRes_segmentation(self):
        coco = COCO3d(self.ANN_PATH_SEG)
        pred_data = [deepcopy(ann) for ann in coco.anns.values()]
        for ann in pred_data:
            ann["score"] = 1.
        res = coco.loadRes(pred_data)
        self.assertEqual(coco.cats, res.cats)
        self.assertEqual(coco.imgs, res.imgs)
