import unittest

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import ResizeTensor
from scene_graph_prediction.structures import BoxList


class TestResizeTensor(unittest.TestCase):
    def test_build(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.RESIZE = 1, 2, 3
        transform = ResizeTensor.build(cfg, True)
        self.assertEqual(transform.size, (1, 2, 3))

    def test_forward_2d(self):
        transform = ResizeTensor((5, 15))
        img = torch.zeros((1, 2, 5, 5)).float()
        target = BoxList(torch.zeros((0, 4)), image_size=(5, 5))
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape, (1, 2, 5, 15))
        self.assertEqual(trans_target.size, (5, 15))

    def test_forward_3d(self):
        transform = ResizeTensor((5, 15, 25))
        img = torch.zeros((1, 2, 5, 5, 5)).float()
        target = BoxList(torch.zeros((0, 6)), image_size=(5, 5, 5))
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape, (1, 2, 5, 15, 25))
        self.assertEqual(trans_target.size, (5, 15, 25))
