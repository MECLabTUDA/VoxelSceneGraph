import unittest

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import ClipAndRescale
from scene_graph_prediction.structures import BoxList


class TestNormalize(unittest.TestCase):
    def test_build(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.CLIP_MIN = 0
        cfg.INPUT.CLIP_MAX = 100
        transform = ClipAndRescale.build(cfg, True)
        self.assertEqual(transform.min_val, cfg.INPUT.CLIP_MIN)
        self.assertEqual(transform.max_val, cfg.INPUT.CLIP_MAX)

    def test_build_invalid_range(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.CLIP_MIN = 0
        cfg.INPUT.CLIP_MAX = 0
        with self.assertRaises(AssertionError):
            ClipAndRescale.build(cfg, True)

    def test_forward(self):
        transform = ClipAndRescale(0, 100)
        img = torch.tensor([-5, 0, 1, 99, 100, 105]).float()
        target = BoxList(torch.zeros((0, 4)), image_size=(1, 1))
        trans_img, trans_target = transform(img, target)
        torch.testing.assert_close(trans_img, torch.tensor([0, 0, .01, .99, 1, 1]).float())
        self.assertTrue(trans_target is target)
