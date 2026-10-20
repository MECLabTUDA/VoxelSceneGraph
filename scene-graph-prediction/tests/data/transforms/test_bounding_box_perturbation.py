import unittest
from copy import deepcopy

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import BoundingBoxPerturbation
from scene_graph_prediction.structures import BoxList


class TestResizeTensor(unittest.TestCase):
    def test_build_train(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.MAX_BB_SHIFT = 1, 2, 3
        cfg.INPUT.N_DIM = 3
        transform = BoundingBoxPerturbation.build(cfg, True)
        self.assertEqual(transform.max_shift, cfg.INPUT.MAX_BB_SHIFT)
        self.assertEqual(transform.n_dim, cfg.INPUT.N_DIM)

    def test_build_test(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.MAX_BB_SHIFT = 1, 2, 3
        cfg.INPUT.N_DIM = 3
        transform = BoundingBoxPerturbation.build(cfg, False)
        self.assertEqual(transform.max_shift, (0, 0, 0))
        self.assertEqual(transform.n_dim, cfg.INPUT.N_DIM)

    def test_build_bb_shift_int(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.MAX_BB_SHIFT = 1
        cfg.INPUT.N_DIM = 2
        transform = BoundingBoxPerturbation.build(cfg, True)
        self.assertEqual(transform.max_shift, (1, 1))

    def test_build_bb_shift_negative(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.MAX_BB_SHIFT = -1, 1, 1
        cfg.INPUT.N_DIM = 2
        with self.assertRaises(AssertionError):
            BoundingBoxPerturbation.build(cfg, True)

    def test_forward(self):
        # Check that no exception is raised
        # The parameters are such that the probability of having no perturbation is negligible
        transform = BoundingBoxPerturbation(3, 10)
        img = torch.zeros((2,) * 10).float()
        boxes = torch.randint(7, 12, (100, 20)).float()  # Make sure that bboxes cannot be shifted outside of the image
        boxes[:, 10:] = boxes[:, :10] + 5  # Make sure that the boxes cannot be/get malformed
        target = BoxList(boxes, image_size=(25,) * 10)
        boxes = deepcopy(boxes)  # Copy required as BoxList copies are not deep
        trans_img, trans_target = transform(img, target)
        self.assertTrue(trans_img is img)
        self.assertFalse(trans_target is target)
        self.assertFalse(torch.allclose(trans_target.boxes, boxes))
        # Check that the boxes are all clipped
        self.assertTrue(torch.all(trans_target.boxes[:10] > 0))
        self.assertTrue(torch.all(trans_target.boxes[10:] < 25))

    def test_forward_target_wrong_ndim(self):
        transform = BoundingBoxPerturbation(2, 10)
        img = torch.zeros((2, 15, 15)).float()
        target = BoxList(torch.zeros(0, 22), image_size=(25,) * 11)
        with self.assertRaises(AssertionError):
            transform(img, target)
