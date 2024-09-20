import unittest

import torch
from PIL import Image
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import ToTensor
from scene_graph_prediction.structures import BoxList


class TestToTensor(unittest.TestCase):
    def test_build(self):
        self.assertTrue(isinstance(ToTensor.build(CfgNode(), False), ToTensor))

    def test_forward_no_op(self):
        transform = ToTensor()
        target = BoxList(torch.zeros((0, 4)), image_size=(5, 5))
        img = torch.zeros((5, 5))
        trans_img, trans_target = transform(img, target)
        self.assertTrue(trans_img is img)
        self.assertTrue(trans_target is target)

    def test_forward_from_image(self):
        transform = ToTensor()
        target = BoxList(torch.zeros((0, 4)), image_size=(5, 5))
        img = Image.new("RGB", (10, 5))
        trans_img, trans_target = transform(img, target)
        self.assertTrue(isinstance(trans_img, torch.Tensor))
        self.assertEqual(trans_img.shape, (3, 5, 10))
        self.assertTrue(trans_target is target)
