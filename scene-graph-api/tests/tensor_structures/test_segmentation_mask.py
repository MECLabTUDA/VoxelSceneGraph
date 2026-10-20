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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest

import torch

from scene_graph_api.tensor_structures import AbstractMaskList, PolygonList, BinaryMaskList


class TestSegmentationMask(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        poly = [[
            [306.5, 423.0,
             277.0, 406.5,
             271.5, 400.0,
             277.0, 389.5,
             292.0, 387.5,
             295.0, 384.5,
             220.0, 374.5,
             210.0, 378.5,
             200.5, 391.0,
             199.5, 404.0,
             203.5, 414.0,
             221.0, 425.5,
             297.0, 438.5,
             306.5, 423.0],
            [100, 100,
             100, 200,
             200, 200,
             200, 100]
        ]]
        width = 640
        height = 480
        size = height, width

        self.poly = PolygonList(poly, size)
        self.masks = PolygonList(poly, size).convert_to_binary_mask_list()
        self.masks.masks = self.masks.masks  # Masks generated from a PolygonList will have the uint8 dtype

    @staticmethod
    def _l1(a: AbstractMaskList, b: AbstractMaskList):
        diff = a.get_mask_tensor() - b.get_mask_tensor()
        diff = torch.sum(torch.abs(diff.float())).item()
        return diff

    def test_convert(self):
        m_hat = self.masks.convert_to_polygon_list().convert_to_binary_mask_list()
        p_hat = self.poly.convert_to_binary_mask_list().convert_to_polygon_list()

        diff_mask = self._l1(self.masks, m_hat)
        diff_poly = self._l1(self.poly, p_hat)
        self.assertTrue(diff_mask == diff_poly)
        self.assertGreaterEqual(8169, diff_mask)

    def test_crop(self):
        box = [250, 400, 299, 499]  # yxyx
        diff = self._l1(self.masks.crop(box), self.poly.crop(box))
        self.assertGreaterEqual(1., diff)

    def test_resize(self):
        new_size = 48, 64
        self.masks.masks = self.masks.masks.float()
        m_hat = self.masks.resize(new_size)
        p_hat = self.poly.resize(new_size)

        diff = self._l1(m_hat, p_hat)
        self.assertTrue(self.masks.size == self.poly.size)
        self.assertTrue(m_hat.size == p_hat.size)
        self.assertTrue(self.masks.size != m_hat.size)
        self.assertGreaterEqual(3., diff)

    def test_resize_mask_still_uint8(self):
        masks = BinaryMaskList(torch.tensor([[3, 1], [4, 2]], dtype=torch.uint8), (2, 2))
        m_hat = masks.resize((1, 1))
        self.assertEqual(torch.uint8, m_hat.get_mask_tensor().dtype)

    def test_resize_mask_still_float32(self):
        masks = BinaryMaskList(torch.tensor([[3, 1], [4, 2]], dtype=torch.float32), (2, 2))
        m_hat = masks.resize((1, 1))
        self.assertEqual(torch.float32, m_hat.get_mask_tensor().dtype)

    def test_flip(self):
        diff_hor = self._l1(self.masks.flip(BinaryMaskList.FlipDim.WIDTH),
                            self.poly.flip(BinaryMaskList.FlipDim.WIDTH))

        diff_ver = self._l1(self.masks.flip(BinaryMaskList.FlipDim.HEIGHT),
                            self.poly.flip(BinaryMaskList.FlipDim.HEIGHT))

        self.assertGreaterEqual(53250, diff_hor)
        self.assertGreaterEqual(42494, diff_ver)


class TestSegmentationMask3d(unittest.TestCase):
    def test_crop(self):
        mask = torch.tensor([
            [
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 2, 0],
                 [0, 0, 0]]
            ],
            [
                [[0, 0, 0],
                 [0, 3, 0],
                 [0, 4, 0],
                 [0, 0, 0]]
            ]
        ], dtype=torch.float64).reshape(1, 2, 4, 3)  # 1 instance
        masks = BinaryMaskList(mask, (2, 4, 3))

        box = [0, 1, 1, 1, 2, 1]  # zyxzyx
        cropped = masks.crop(box)
        self.assertEqual((2, 2, 1), cropped.size)
        # noinspection PyTypeChecker
        self.assertTrue(torch.all(torch.tensor([[[[1], [2]], [[3], [4]]]]) == cropped.masks))

    def test_resize(self):
        mask = torch.tensor([
            [
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 2, 0],
                 [0, 0, 0]]
            ],
            [
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]
        ], dtype=torch.float64).reshape(1, 2, 4, 3)  # 1 instance
        masks = BinaryMaskList(mask, (2, 4, 3))

        new_size = 2, 8, 3
        resized = masks.resize(new_size)
        self.assertEqual(new_size, resized.size)
        self.assertEqual((1, 2, 8, 3), resized.masks.shape)

        expected = torch.tensor([
            [
                [[0.0000, 0.0000, 0.0000],
                 [0.0000, 0.2500, 0.0000],
                 [0.0000, 0.7500, 0.0000],
                 [0.0000, 1.2500, 0.0000],
                 [0.0000, 1.7500, 0.0000],
                 [0.0000, 1.5000, 0.0000],
                 [0.0000, 0.5000, 0.0000],
                 [0.0000, 0.0000, 0.0000]],
                [[0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000]]
            ]
        ]).double()
        torch.testing.assert_close(expected, resized.masks)
