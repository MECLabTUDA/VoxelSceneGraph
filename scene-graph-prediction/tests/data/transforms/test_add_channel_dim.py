import unittest

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import AddChannelDim
from scene_graph_prediction.structures import BoxList


class TestResizeTensor(unittest.TestCase):
    def test_build(self):
        # Check that no exception is raised
        AddChannelDim.build(CfgNode(), True)

    def test_forward(self):
        transform = AddChannelDim()
        img = torch.zeros((2, 15, 15)).float()
        target = BoxList(torch.zeros((0, 4)), image_size=(15, 15))
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape, (1, 2, 15, 15))
        self.assertEqual(trans_target.size, (15, 15))
