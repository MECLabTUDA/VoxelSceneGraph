import unittest
from typing import Type

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomDepthFlip
from scene_graph_prediction.structures import BoxList

_FLIPS = Type[RandomHorizontalFlip | RandomVerticalFlip | RandomDepthFlip]


class TestRandomFlips(unittest.TestCase):
    def _test_build_train(self, cls: _FLIPS):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = .5
        cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = .5
        cfg.INPUT.DEPTH_FLIP_PROB_TRAIN = .5
        transform = cls.build(cfg, True)
        self.assertEqual(transform.prob, .5)

    def _test_build_test(self, cls: _FLIPS):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = .5
        cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = .5
        cfg.INPUT.DEPTH_FLIP_PROB_TRAIN = .5
        transform = cls.build(cfg, False)
        self.assertEqual(transform.prob, 0.)

    def test_build_train(self):
        for cls in [RandomHorizontalFlip, RandomVerticalFlip, RandomDepthFlip]:
            with self.subTest(cls=cls):
                self._test_build_train(cls)

    def test_build_test(self):
        for cls in [RandomHorizontalFlip, RandomVerticalFlip, RandomDepthFlip]:
            with self.subTest(cls=cls):
                self._test_build_test(cls)

    def test_forward_horizontal(self):
        transform = RandomHorizontalFlip(1.)
        target = BoxList(torch.zeros((0, 6)), image_size=(1, 2, 3))
        img = torch.arange(0, 8).view(2, 2, 2)
        trans_img, _ = transform(img, target)
        self.assertEqual(trans_img[0, 0, 0], 1)
        self.assertEqual(trans_img[0, 0, 1], 0)

    def test_forward_vertical(self):
        transform = RandomVerticalFlip(1.)
        target = BoxList(torch.zeros((0, 6)), image_size=(1, 2, 3))
        img = torch.arange(0, 8).view(2, 2, 2)
        trans_img, _ = transform(img, target)
        self.assertEqual(trans_img[0, 0, 0], 2)
        self.assertEqual(trans_img[0, 1, 0], 0)

    def test_forward_depth(self):
        transform = RandomDepthFlip(1.)
        target = BoxList(torch.zeros((0, 6)), image_size=(1, 2, 3))
        img = torch.arange(0, 8).view(2, 2, 2)
        trans_img, _ = transform(img, target)
        self.assertEqual(trans_img[0, 0, 0], 4)
        self.assertEqual(trans_img[1, 0, 0], 0)

    def test_forward_horizontal_with_channel(self):
        transform = RandomHorizontalFlip(1.)
        target = BoxList(torch.zeros((0, 6)), image_size=(1, 2, 3))
        img = torch.arange(0, 8).view(1, 2, 2, 2)
        trans_img, _ = transform(img, target)
        self.assertEqual(trans_img[0, 0, 0, 0], 1)
        self.assertEqual(trans_img[0, 0, 0, 1], 0)

    def test_forward_horizontal_with_batch_channel(self):
        transform = RandomHorizontalFlip(1.)
        target = BoxList(torch.zeros((0, 6)), image_size=(1, 2, 3))
        img = torch.arange(0, 8).view(1, 1, 2, 2, 2)
        trans_img, _ = transform(img, target)
        self.assertEqual(trans_img[0, 0, 0, 0, 0], 1)
        self.assertEqual(trans_img[0, 0, 0, 0, 1], 0)
