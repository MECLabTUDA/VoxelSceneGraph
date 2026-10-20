import unittest

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import Normalize
from scene_graph_prediction.structures import BoxList


class TestNormalize(unittest.TestCase):
    def test_build(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.PIXEL_MEAN = 1, 2, 3
        cfg.INPUT.PIXEL_STD = 4, 5, 6
        transform = Normalize.build(cfg, True)
        self.assertEqual(transform.mean, cfg.INPUT.PIXEL_MEAN)
        self.assertEqual(transform.std, cfg.INPUT.PIXEL_STD)

    def test_build_error_length(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.PIXEL_MEAN = 1, 2, 3
        cfg.INPUT.PIXEL_STD = 4,
        with self.assertRaises(AssertionError):
            Normalize.build(cfg, True)

    def test_build_error_std(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.PIXEL_MEAN = 1, 2, 3
        cfg.INPUT.PIXEL_STD = 1, 0, 2
        with self.assertRaises(AssertionError):
            Normalize.build(cfg, True)

    def test_forward(self):
        transform = Normalize((1, 0), (1, 2))
        img = torch.tensor([0., 1., 0., 1.]).view(2, 2, 1)
        target = BoxList(torch.zeros((0, 4)), image_size=(1, 1))
        trans_img, trans_target = transform(img, target)
        torch.testing.assert_close(trans_img, torch.tensor([-1., 0., 0., .5]).view(2, 2, 1))
        self.assertTrue(trans_target is target)

    def test_forward_wrong_channels(self):
        transform = Normalize((1, 0), (1, 2))
        img = torch.tensor([0., 1., 0., 1.]).view(1, 2, 2)
        target = BoxList(torch.zeros((0, 4)), image_size=(1, 1))
        with self.assertRaises(AssertionError):
            transform(img, target)
