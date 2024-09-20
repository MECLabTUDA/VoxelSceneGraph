import unittest

import torch
from PIL import Image
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import ResizeImage2D
from scene_graph_prediction.structures import BoxList


class TestResizeImage2D(unittest.TestCase):
    def setUp(self):
        # Build the config
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.MIN_SIZE_TRAIN = 5
        cfg.INPUT.MAX_SIZE_TRAIN = 25
        cfg.INPUT.MIN_SIZE_TEST = 35
        cfg.INPUT.MAX_SIZE_TEST = 45
        self.cfg = cfg

    def test_build_train(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        self.assertEqual(self.cfg.INPUT.MIN_SIZE_TRAIN, transform.min_size)
        self.assertEqual(self.cfg.INPUT.MAX_SIZE_TRAIN, transform.max_size)

    def test_build_test(self):
        transform = ResizeImage2D.build(self.cfg, is_train=False)
        self.assertEqual(self.cfg.INPUT.MIN_SIZE_TEST, transform.min_size)
        self.assertEqual(self.cfg.INPUT.MAX_SIZE_TEST, transform.max_size)

    def test_forward_tensor_no_change(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 20, 15
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        img = torch.zeros((1,) + base_img_size).float()
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape[1:], base_img_size)
        self.assertEqual(trans_target.size, base_img_size)

    def test_forward_tensor_too_big(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 50, 20
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        img = torch.zeros((1,) + base_img_size).float()
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape, (1, 25, 10))
        self.assertEqual(trans_target.size, (25, 10))

    def test_forward_tensor_too_small(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 4, 4
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        img = torch.zeros((1,) + base_img_size).float()
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.shape, (1, 5, 5))
        self.assertEqual(trans_target.size, (5, 5))

    def test_forward_image_no_change(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 20, 15
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        # noinspection PyTypeChecker
        img = Image.new("RGB", base_img_size[::-1])
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.size, base_img_size[::-1])
        self.assertEqual(trans_target.size, base_img_size)

    def test_forward_image_too_big(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 50, 20
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        # noinspection PyTypeChecker
        img = Image.new("RGB", base_img_size[::-1])
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.size, (10, 25))
        self.assertEqual(trans_target.size, (25, 10))

    def test_forward_image_too_small(self):
        transform = ResizeImage2D.build(self.cfg, is_train=True)
        base_img_size = 4, 4
        target = BoxList(torch.zeros((0, 4)), image_size=base_img_size)
        # noinspection PyTypeChecker
        img = Image.new("RGB", base_img_size[::-1])
        trans_img, trans_target = transform(img, target)
        self.assertEqual(trans_img.size, (5, 5))
        self.assertEqual(trans_target.size, (5, 5))
