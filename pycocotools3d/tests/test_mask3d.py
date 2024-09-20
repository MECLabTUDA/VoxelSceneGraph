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

import numpy as np

import pycocotools3d.mask3d as mask_util3d


# Note: masks are indexed zyx and bounding boxes are zyxdhw ordered

class TestEncodeDecode(unittest.TestCase):
    """Test mask -> RLE -> mask"""

    def testEncodeDecode(self):
        mask = np.zeros((2, 5, 7))
        mask[1, 4, 6] = 1
        enc = mask_util3d.encode(mask)
        decoded = mask_util3d.decode(enc)
        self.assertTrue(np.alltrue(mask == decoded))

    def testEncodeDecodeMultiple(self):
        mask = np.zeros((2, 5, 7, 3))
        mask[1, 2, 1, 0] = 1
        mask[1, 3, 4, 1] = 1
        mask[1, 4, 5, 2] = 1
        enc = mask_util3d.encode(mask)
        decoded = mask_util3d.decode(enc)
        self.assertTrue(np.alltrue(mask == decoded))


class TestBboxToRleToBbox(unittest.TestCase):
    """Test bb -> rle -> bb"""

    def testBboxToRleToBboxFullImage3D(self):
        bb = np.array([[0, 0, 0, 4, 3, 2]], dtype=np.float64)
        d, h, w = 4, 3, 2
        rle = mask_util3d.frPyObjects(bb, d, h, w)
        bb_ = mask_util3d.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))

    def testBboxToRleToBboxNone3D(self):
        bb = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64)
        d, h, w = 4, 3, 2
        rle = mask_util3d.frPyObjects(bb, d, h, w)
        bb_ = mask_util3d.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))

    def testBboxToRleToBboxPartial3D(self):
        bb = np.array([[3, 2, 1, 6, 5, 4]], dtype=np.float64)
        d, h, w = 9, 8, 7
        rle = mask_util3d.frPyObjects(bb, d, h, w)
        bb_ = mask_util3d.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))


# noinspection PyUnresolvedReferences
class TestToBBox(unittest.TestCase):
    """Test mask -> bb"""

    def testToBboxFullImage(self):
        mask = np.array([
            [[0, 1],
             [1, 1]]
        ])
        enc = mask_util3d.encode(mask)
        bbox = mask_util3d.toBbox(enc)
        self.assertTrue((bbox == np.array([0, 0, 0, 1, 2, 2], dtype="float32")).all(), bbox)

    def testToBboxRectImage(self):
        mask = np.array([
            [[0, 0, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [1, 1, 0]],
            [[0, 1, 0],
             [1, 1, 0]],
            [[0, 0, 0],
             [0, 0, 0]]
        ])  # shape: 4, 2, 3
        enc = mask_util3d.encode(mask)
        bbox = mask_util3d.toBbox(enc)
        self.assertTrue(np.all(bbox == np.array([1, 0, 0, 2, 2, 2], dtype="float32")), bbox)

    def testToBboxNonFullImage(self):
        mask = np.zeros((10, 10, 2), dtype=np.uint8)
        mask[2:4, 3:6] = 1
        enc = mask_util3d.encode(mask)
        bbox = mask_util3d.toBbox(enc)
        self.assertTrue((bbox == np.array([2, 3, 0, 2, 3, 2], dtype="float32")).all(), bbox)


class TestIou(unittest.TestCase):
    def test_iou(self):
        bbox1 = np.array([[0, 0, 0, 2, 2, 2]], dtype=np.float64)
        bbox2 = np.array([[1, 1, 1, 2, 2, 2]], dtype=np.float64)
        self.assertAlmostEqual(1 / 15, mask_util3d.iou(bbox1, bbox2, [0]))

    def test_iou_full(self):
        bbox1 = np.array([[0, 0, 0, 2, 2, 2]], dtype=np.float64)
        bbox2 = np.array([[0, 0, 0, 2, 2, 2]], dtype=np.float64)
        self.assertAlmostEqual(1, mask_util3d.iou(bbox1, bbox2, [0]))

    def test_iou_zero(self):
        bbox1 = np.array([[0, 0, 0, 2, 2, 2]], dtype=np.float64)
        bbox2 = np.array([[0, 0, 3, 2, 2, 2]], dtype=np.float64)
        self.assertAlmostEqual(0, mask_util3d.iou(bbox1, bbox2, [0]))
