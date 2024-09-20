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

import pycocotools3d.mask as mask_util


# Note: masks are indexed yx and bounding boxes are zyxdhw ordered

class TestEncodeDecode(unittest.TestCase):
    """Test mask -> RLE -> mask"""

    def testEncodeDecode(self):
        mask = np.zeros((2, 5))
        mask[1, 4] = 1
        enc = mask_util.encode(mask)
        decoded = mask_util.decode(enc)
        self.assertTrue(np.alltrue(mask == decoded))

    def testEncodeDecodeMultiple(self):
        mask = np.zeros((2, 5, 3))
        mask[1, 2, 0] = 1
        mask[1, 3, 1] = 1
        mask[1, 4, 2] = 1
        enc = mask_util.encode(mask)
        decoded = mask_util.decode(enc)
        self.assertTrue(np.alltrue(mask == decoded))


class TestPolyToRle(unittest.TestCase):
    """test poly -> rle vs bb -> rle"""

    def testFrOnePoly(self):
        poly = [0, 1,
                0, 3,
                2, 3,
                2, 1
                ]
        poly_rle = mask_util.frPyObjects(poly, 2, 3)

        mask = np.array([
            [0, 1, 1],
            [0, 1, 1]
        ])
        mask_rle = mask_util.encode(mask)

        self.assertEqual(mask_rle["size"], poly_rle["size"])
        self.assertEqual(mask_rle["counts"], poly_rle["counts"])

    def testFrPolyList(self):
        poly = [0, 1,
                0, 3,
                2, 3,
                2, 1
                ]
        poly_rle = mask_util.frPyObjects([poly], 2, 3)[0]

        mask = np.array([
            [0, 1, 1],
            [0, 1, 1]
        ])
        mask_rle = mask_util.encode(mask)

        self.assertEqual(mask_rle["size"], poly_rle["size"])
        self.assertEqual(mask_rle["counts"], poly_rle["counts"])

    def testFrPolyNumpy(self):
        poly = [np.array([0, 1,
                          0, 3,
                          2, 3,
                          2, 1
                          ])]
        poly_rle = mask_util.frPyObjects(poly, 2, 3)[0]

        mask = np.array([
            [0, 1, 1],
            [0, 1, 1]
        ])
        mask_rle = mask_util.encode(mask)

        self.assertEqual(mask_rle["size"], poly_rle["size"])
        self.assertEqual(mask_rle["counts"], poly_rle["counts"])


class TestBboxToRleToBbox(unittest.TestCase):
    """Test bb -> rle -> bb"""

    def testBboxToRleToBboxOneBbox(self):
        bb = [0, 0, 3, 2]
        h, w = 3, 2
        rle = mask_util.frPyObjects(bb, h, w)
        bb_ = mask_util.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))

    def testBboxToRleToBboxFullImage(self):
        bb = np.array([[0, 0, 3, 2]], dtype=np.float64)
        h, w = 3, 2
        rle = mask_util.frPyObjects(bb, h, w)
        bb_ = mask_util.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))

    def testBboxToRleToBboxNone(self):
        bb = np.array([[0, 0, 0, 0]], dtype=np.float64)
        h, w = 3, 2
        rle = mask_util.frPyObjects(bb, h, w)
        bb_ = mask_util.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))

    def testBboxToRleToBboxPartial(self):
        bb = np.array([[0, 0, 6, 5]], dtype=np.float64)
        h, w = 6, 5
        rle = mask_util.frPyObjects(bb, h, w)
        bb_ = mask_util.toBbox(rle)
        self.assertTrue(np.alltrue(bb == bb_))


# noinspection PyUnresolvedReferences
class TestToBBox(unittest.TestCase):
    """Test mask -> bb"""

    def testToBboxCorner(self):
        mask = np.array([[0, 1],
                         [0, 0]])
        bbox = mask_util.toBbox(mask_util.encode(mask))
        self.assertTrue((bbox == np.array([0, 1, 1, 1], dtype="float32")).all(), bbox)

    def testToBboxRight(self):
        mask = np.array([[0, 1],
                         [0, 1]])
        bbox = mask_util.toBbox(mask_util.encode(mask))
        self.assertTrue((bbox == np.array([0, 1, 2, 1], dtype="float32")).all(), bbox)

    def testToBboxFullImage(self):
        mask = np.array([[0, 1],
                         [1, 1]])
        bbox = mask_util.toBbox(mask_util.encode(mask))
        self.assertTrue((bbox == np.array([0, 0, 2, 2], dtype="float32")).all(), bbox)

    def testToBboxRectImage(self):
        mask = np.array([[0, 1, 0],
                         [1, 1, 0]])
        bbox = mask_util.toBbox(mask_util.encode(mask))
        self.assertTrue((bbox == np.array([0, 0, 2, 2], dtype="float32")).all(), bbox)

    def testToBboxNonFullImage(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 3:6] = 1
        bbox = mask_util.toBbox(mask_util.encode(mask))
        self.assertTrue((bbox == np.array([2, 3, 2, 3], dtype="float32")).all(), bbox)


class TestIou(unittest.TestCase):
    def test_iou(self):
        bbox1 = np.array([[0, 0, 2, 3]], dtype=np.float64)
        bbox2 = np.array([[1, 2, 2, 2]], dtype=np.float64)
        self.assertAlmostEqual(1 / 9., mask_util.iou(bbox1, bbox2, [0]))

    def test_iou_full(self):
        bbox1 = np.array([[0, 0, 2, 3]], dtype=np.float64)
        bbox2 = np.array([[0, 0, 2, 3]], dtype=np.float64)
        self.assertAlmostEqual(1., mask_util.iou(bbox1, bbox2, [0]))

    def test_iou_zero(self):
        bbox1 = np.array([[0, 1, 3, 1]], dtype=np.float64)
        bbox2 = np.array([[1, 0, 3, 1]], dtype=np.float64)
        self.assertAlmostEqual(0., mask_util.iou(bbox1, bbox2, [0]))


class TestMaskUtil(unittest.TestCase):
    def testInvalidRLECounts(self):
        # noinspection SpellCheckingInspection
        rle = {'size': [1024, 1024],
               'counts': 'jd`0=`o06J5L4M3L3N2N2N2N2N1O2N2N101N1O2O0O1O2N100O1O2N100O1O1O1O1O101N1O1O1O1O1O1O101N1O100O1'
                         '01O0O100000000000000001O00001O1O0O2O1N3N1N3N3L5Kh0XO6J4K5L5Id[o5N]dPJ7K4K4M3N2M3N2N1O2N100O2O'
                         '0O1000O01000O101N1O1O2N2N2M3M3M4J7Inml5H[RSJ6L2N2N2N2O000000000000O2O1N2N2Mkm81SRG6L3L3N2O1N2'
                         'N2O0O2O00001O0000000000O2O001N2O0O2N2N3M3L5JRjf6MPVYI8J4L3N3M2N1O2O1N101N1000000O10000001O000'
                         'O101N101N1O2N2N2N3L4L7FWZ_50ne`J0000001O000000001O0000001O1O0N3M3N1O2N2N2O1N2O001N2`RO^O`k0c0'
                         '[TOEak0;\\\\TOJbk07\\\\TOLck03[TO0dk01ZTO2dk0OYTO4gk0KXTO7gk0IXTO8ik0HUTO:kk0ETTO=lk0CRTO>Pl0'
                         '@oSOb0Rl0\\\\OmSOe0Tl0[OjSOg0Ul0YOiSOi0Wl0XOgSOi0Yl0WOeSOk0[l0VOaSOn0kh0cNmYO'}
        with self.assertRaises(ValueError):
            mask_util.decode(rle)

    def testZeroLeadingRLE(self):
        # A foreground segment of length 0 was not previously handled correctly.
        # This input rle has 3 leading zeros.
        # noinspection SpellCheckingInspection
        rle = {'size': [1350, 1080],
               'counts': '000lg0Zb01O00001O00001O001O00001O00001O001O00001O01O2N3M3M3M2N3M3N2M3M2N1O1O1O1O2N1O1O1O2N1O1'
                         'O101N1O1O1O2N1O1O1O2N3M2N1O2N1O2O0O2N1O1O2N1O2N1O2N1O2N1O2N1O2O0O2N1O3M2N1O2N2N2N2N2N1O2N2N2N'
                         '2N1O2N2N2N2N2N1N3N2N00O1O1O1O100000000000000O100000000000000001O0000001O00001O0O5L7I5K4L4L3M2'
                         'N2N2N1O2m]OoXOm`0Sg0j^OVYOTa0lf0c^O]YO[a0ef0\\^OdYOba0bg0N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N3M2'
                         'M4M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2M4M2N2N3M2M4M2N2N3M2M3N3M2N3M2M3N3M2N2N3L3N2N3M2N3L3N2N3M'
                         '5J4M3M4L3M3L5L3M3M4L3L4\\EXTOd6jo0K6J5K6I4M1O1O1O1N2O1O1O001N2O00001O0O101O000O2O00001N101N10'
                         '1N2N101N101N101N2O0O2O0O2O0O2O1N101N2N2O1N2O1N2O1N101N2O1N2O1N2O0O2O1N2N2O1N2O0O2O1N2O1N2N2N1'
                         'N4M2N2M4M2N3L3N2N3L3N3L3N2N3L3N2N3L3M4L3M3M4L3M5K5K5K6J5K5K6J7I7I7Ibijn0'}
        orig_bbox = mask_util.toBbox(rle)
        mask = mask_util.decode(rle)
        rle_new = mask_util.encode(mask)
        new_bbox = mask_util.toBbox(rle_new)
        self.assertTrue(np.equal(orig_bbox, new_bbox).all())
