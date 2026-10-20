import unittest

import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import RandomAffine
from scene_graph_prediction.structures import BoxList


class TestResizeTensor(unittest.TestCase):
    # noinspection DuplicatedCode
    def test_build_train_2d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 2,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1), (1., 1.2)
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        transform = RandomAffine.build(cfg, True)
        transform = RandomAffine.build(cfg, True)
        self.assertEqual(transform.max_translate, cfg.INPUT.AFFINE_MAX_TRANSLATE)
        self.assertEqual(transform.scale_range, cfg.INPUT.AFFINE_SCALE_RANGE)
        self.assertEqual(transform.max_rotate, cfg.INPUT.AFFINE_MAX_ROTATE)
        self.assertEqual(transform.n_dim, cfg.INPUT.N_DIM)

    # noinspection DuplicatedCode
    def test_build_train_3d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 2, 3
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1), (1., 1.2), (1., 1.3)
        cfg.INPUT.AFFINE_MAX_ROTATE = 3, 4, 5
        cfg.INPUT.N_DIM = 3
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        transform = RandomAffine.build(cfg, True)
        transform = RandomAffine.build(cfg, True)
        self.assertEqual(transform.max_translate, cfg.INPUT.AFFINE_MAX_TRANSLATE)
        self.assertEqual(transform.scale_range, cfg.INPUT.AFFINE_SCALE_RANGE)
        self.assertEqual(transform.max_rotate, cfg.INPUT.AFFINE_MAX_ROTATE)
        self.assertEqual(transform.n_dim, cfg.INPUT.N_DIM)

    # noinspection DuplicatedCode
    def test_build_test_2d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 2,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1), (1., 1.2)
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        transform = RandomAffine.build(cfg, True)
        transform = RandomAffine.build(cfg, False)
        self.assertEqual(transform.max_translate, (0, 0))
        self.assertEqual(transform.scale_range, ((1., 1.), (1., 1.)))
        self.assertEqual(transform.max_rotate, (0.,))

    # noinspection DuplicatedCode
    def test_build_test_3d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 2, 3
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1), (1., 1.2), (1., 1.3)
        cfg.INPUT.AFFINE_MAX_ROTATE = 3, 4, 5
        cfg.INPUT.N_DIM = 3
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        transform = RandomAffine.build(cfg, True)
        transform = RandomAffine.build(cfg, False)
        self.assertEqual(transform.max_translate, (0, 0, 0))
        self.assertEqual(transform.scale_range, ((1., 1.), (1., 1.), (1., 1.)))
        self.assertEqual(transform.max_rotate, (0., 0., 0.))

    # noinspection DuplicatedCode
    def test_build_len1(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        transform = RandomAffine.build(cfg, True)
        self.assertEqual(transform.max_translate, (1, 1))
        self.assertEqual(transform.scale_range, ((1., 1.1), (1., 1.1)))

    # noinspection DuplicatedCode
    def test_build_ndim_unsupported(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 4
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    # noinspection DuplicatedCode
    def test_build_invalid_max_translate(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 1, 1
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    # noinspection DuplicatedCode
    def test_build_invalid_scale_range(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 1
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.1), (1., 1.1), (1., 1.1)
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    # noinspection DuplicatedCode
    def test_build_invalid_scale_range_order(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1, 1
        cfg.INPUT.AFFINE_SCALE_RANGE = (1.3, 1.),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3,
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    # noinspection DuplicatedCode
    def test_build_invalid_rotate_2d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3, 3
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    # noinspection DuplicatedCode
    def test_build_invalid_rotate_3d(self):
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.AFFINE_MAX_TRANSLATE = 1,
        cfg.INPUT.AFFINE_SCALE_RANGE = (1., 1.),
        cfg.INPUT.AFFINE_MAX_ROTATE = 3, 3
        cfg.INPUT.N_DIM = 2
        cfg.MODEL = CfgNode()
        cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION = True
        cfg.MODEL.MASK_ON = False
        with self.assertRaises(AssertionError):
            RandomAffine.build(cfg, True)

    def test_forward(self):
        # Check that no exception is raised
        transform = RandomAffine(2, (2, 2), ((0.9, 1.1), (0.9, 1.1)), (10,))
        img = torch.randint(1, 5, (1, 10, 10)).float()
        target = BoxList(torch.zeros((1, 4)), image_size=(10, 10))
        target.LABELMAP = torch.zeros(10, 10).float()
        trans_img, trans_target = transform(img, target)
        self.assertFalse(torch.allclose(trans_img, img))
        self.assertFalse(trans_target is target)
