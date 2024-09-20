import unittest

import numpy as np
import torch

# noinspection PyProtectedMember
from scene_graph_prediction.modeling.region_proposal._common.anchor_generator import _lengths_centers_to_anchors, \
    _lengths_centers, _ratio_enum2d, _ratio_enum3d, _generate_anchors2d, _generate_anchors3d, AnchorGenerator, \
    _stride_center_manual_anchors2d, _stride_center_manual_anchors3d
from scene_graph_prediction.structures import BufferList, BoxList


class TestAnchorGeneratorHelpers(unittest.TestCase):
    def test__mk_anchors(self):
        """Make sure that the generated anchors have the correct format."""
        zyx_centers = [0., 50., 100.]
        lengths = [
            np.array([1, 2, 11]),  # Depths
            np.array([1, 2, 21]),  # Heights
            np.array([1, 2, 31]),  # Widths
        ]

        anchors = _lengths_centers_to_anchors(lengths, zyx_centers)
        expected = np.array([
            0, 50, 100, 0, 50, 100,
            -.5, 49.5, 99.5, .5, 50.5, 100.5,
            -5, 40, 85, 5, 60, 115
        ], dtype=float).reshape((-1, 6))

        np.testing.assert_almost_equal(anchors, expected)

    def test__lengths_centers(self):
        """Test reverse-operation from _lengths_centers_to_anchors."""
        anchors = np.array([
            0, 50, 100, 0, 50, 100,
            -.5, 49.5, 99.5, .5, 50.5, 100.5,
            -5, 40, 85, 5, 60, 115
        ], dtype=float).reshape((-1, 6))

        for anchor in anchors:
            with self.subTest(anchor):
                lengths, centers = _lengths_centers(anchor)
                new_anchor = _lengths_centers_to_anchors([np.array([length]) for length in lengths], centers)[0]
                np.testing.assert_almost_equal(new_anchor, anchor)

    def test__ratio_enum_1_2D(self):
        anchors = np.array([
            0, 0, 10, 10,  # Should not be changed as the ratio is already 1.
            0, 0, 17, 1,  # Should be reshaped to a 6x6 anchor, but moved to be around the center
            0, 0, 1, 17,  # Should be reshaped to a 6x6 anchor, but moved to be around the center
        ], dtype=float).reshape((-1, 4))

        expected_anchors = np.array([
            0, 0, 10, 10,
            6, -2, 11, 3,
            -2, 6, 3, 11,
        ], dtype=float).reshape((-1, 1, 4))

        for anchor, expected in zip(anchors, expected_anchors):
            with self.subTest(anchor=anchor, expected=expected):
                new_anchors = _ratio_enum2d(anchor, np.array([1.]))
                np.testing.assert_almost_equal(new_anchors, expected)

    def test__ratio_enum(self):
        anchors = np.array([
            0, 0, 9, 9,  # 10x10=100
            0, 0, 3, 8,  # 4x9=36
        ], dtype=float).reshape((-1, 4))

        expected_anchors = np.array([
            2.5, -5., 6.5, 14.,  # 5x20=100
            -5., 2.5, 14., 6.5,  # 20x5=100

            0.5, -1.5, 2.5, 9.5,  # 3x12=36
            -4., 3., 7., 5.,  # 12x3=36
        ], dtype=float).reshape((-1, 2, 4))

        for anchor, expected in zip(anchors, expected_anchors):
            with self.subTest(anchor=anchor, expected=expected):
                new_anchors = _ratio_enum2d(anchor, np.array([.25, 4.]))
                np.testing.assert_almost_equal(new_anchors, expected)

    def test__ratio_enum_1_3D(self):
        anchors = np.array([
            0, 0, 0, 10, 10, 10,  # Should not be changed as the ratio is already 1.
            0, 0, 0, 1, 1, 17,
        ], dtype=float).reshape((-1, 6))

        expected_anchors = np.array([
            0, 0, 0, 10, 10, 10,
            0., -2., 6., 1., 3., 11.,
        ], dtype=float).reshape((-1, 1, 6))

        for anchor, expected in zip(anchors, expected_anchors):
            with self.subTest(anchor=anchor, expected=expected):
                new_anchors = _ratio_enum3d(anchor, np.array([1.]))
                np.testing.assert_almost_equal(new_anchors, expected)

    def test__ratio_enum3d(self):
        anchors = np.array([
            0, 0, 0, 9, 9, 9,  # 10x10x10=1000
        ], dtype=float).reshape((-1, 6))

        expected_anchors = np.array([
            0., 2.5, -5., 9., 6.5, 14.,
            0., -5., 2.5, 9., 14., 6.5,
        ], dtype=float).reshape((-1, 2, 6))

        for anchor, expected in zip(anchors, expected_anchors):
            with self.subTest(anchor=anchor, expected=expected):
                new_anchors = _ratio_enum3d(anchor, np.array([.25, 4.]))
                np.testing.assert_almost_equal(new_anchors, expected)

    def test__generate_anchors2D(self):
        """Test results."""
        anchors = _generate_anchors2d(16, (512,), (1., .5, 2.))
        expected = torch.tensor([[-248.0000, -248.0000, 263.0000, 263.0000],
                                 [-173.0193, -354.0387, 188.0193, 369.0387],
                                 [-354.0387, -173.0193, 369.0387, 188.0193]], dtype=torch.float64)
        torch.testing.assert_close(anchors, expected, atol=1e-3, rtol=1e-7)

    def test__generate_anchors3D(self):
        """Test results."""
        anchors = _generate_anchors3d(16, 4, (512,), (2,), (1., .5, 2.))
        expected = torch.tensor([[1.0000, -248.0000, -248.0000, 2.0000, 263.0000, 263.0000],
                                 [1.0000, -173.0193, -354.0387, 2.0000, 188.0193, 369.0387],
                                 [1.0000, -354.0387, -173.0193, 2.0000, 369.0387, 188.0193]], dtype=torch.float64)
        torch.testing.assert_close(anchors, expected, atol=1e-3, rtol=1e-7)

    def test__stride_center_manual_anchors2d(self):
        anchors = _stride_center_manual_anchors2d(16, ((4, 8),))
        expected = torch.tensor([[6., 4., 9., 11.]], dtype=torch.float64)
        torch.testing.assert_close(anchors, expected, atol=1e-3, rtol=1e-7)

    def test__stride_center_manual_anchors3d(self):
        anchors = _stride_center_manual_anchors3d(16, 4, ((2, 4, 8),))
        expected = torch.tensor([[1., 6., 4., 2., 9., 11.]], dtype=torch.float64)
        torch.testing.assert_close(anchors, expected, atol=1e-3, rtol=1e-7)


class TestAnchorGenerator(unittest.TestCase):
    # 2D + RPN
    def test_init_2d_rpn_no_anchors(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2),),
            sizes=tuple(),
        )
        self.assertEqual(2, gen.n_dim)
        self.assertEqual(((2, 2),), gen.strides)
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level...
        self.assertEqual(0, gen.cell_anchors[0].shape[0])  # ...but no anchor

    def test_init_2d_rpn_anchors_no_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2),),
            sizes=(50,),
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios

    def test_init_2d_rpn_only_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2),),
            sizes=tuple(),
            custom_anchors=((10, 10),)
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(1, gen.cell_anchors[0].shape[0])  # 1 custom

    def test_init_2d_rpn_anchors_with_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2),),
            sizes=(50,),
            custom_anchors=((10, 10),)
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(4, gen.cell_anchors[0].shape[0])  # 3 ratios + 1 custom

    # 2D + FPN
    def test_init_2d_fpn_no_anchors(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2), (4, 4)),
            sizes=(tuple(), tuple()),
        )
        self.assertEqual(2, gen.n_dim)
        self.assertEqual(((2, 2), (4, 4)), gen.strides)
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels...
        self.assertEqual(0, gen.cell_anchors[0].shape[0])  # ...but no anchor
        self.assertEqual(0, gen.cell_anchors[1].shape[0])  # ...but no anchor

    def test_init_2d_fpn_anchors_no_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2), (4, 4)),
            sizes=((50,), (100,)),
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios
        self.assertEqual(3, gen.cell_anchors[1].shape[0])  # 3 ratios

    def test_init_2d_fpn_anchors_no_custom_sizes_as_int(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2), (4, 4)),
            sizes=(50, 100),
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios
        self.assertEqual(3, gen.cell_anchors[1].shape[0])  # 3 ratios

    def test_init_2d_fpn_only_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2), (4, 4)),
            sizes=(tuple(), tuple()),
            custom_anchors=((10, 10), (20, 20))
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(1, gen.cell_anchors[0].shape[0])  # 1 custom
        self.assertEqual(1, gen.cell_anchors[1].shape[0])  # 1 custom

    def test_init_2d_fpn_anchors_with_custom(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2), (4, 4)),
            sizes=(50, 100),
            custom_anchors=((10, 10), (20, 20))
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(4, gen.cell_anchors[0].shape[0])  # 3 ratios + 1 custom
        self.assertEqual(4, gen.cell_anchors[1].shape[0])  # 3 ratios + 1 custom

    def test_init_2d_fpn_invalid_sizes_strides_lengths(self):
        with self.assertRaises(RuntimeError):
            AnchorGenerator(
                n_dim=2,
                anchor_strides=((2, 2), (4, 4)),
                sizes=(50,),
            )

    # 3D + RPN
    def test_init_3d_rpn_no_anchors(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2),),
            sizes=tuple(),
            depths=tuple(),
        )
        self.assertEqual(3, gen.n_dim)
        self.assertEqual(((2, 2, 2),), gen.strides)
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level...
        self.assertEqual(0, gen.cell_anchors[0].shape[0])  # ...but no anchor

    def test_init_3d_rpn_anchors_no_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2),),
            sizes=(50,),
            depths=(2,),
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios

    def test_init_3d_rpn_only_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2),),
            sizes=tuple(),
            custom_anchors=((10, 10, 10),),
            depths=tuple(),
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(1, gen.cell_anchors[0].shape[0])  # 1 custom

    def test_init_3d_rpn_anchors_with_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2),),
            sizes=(50,),
            depths=(2,),
            custom_anchors=((10, 10, 10),)
        )
        self.assertEqual(1, len(gen.cell_anchors))  # 1 level
        self.assertEqual(4, gen.cell_anchors[0].shape[0])  # 3 ratios + 1 custom

    # 3D + FPN
    def test_init_3d_fpn_no_anchors(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=(tuple(), tuple()),
            depths=(tuple(), tuple()),
        )
        self.assertEqual(3, gen.n_dim)
        self.assertEqual(((2, 2, 2), (4, 4, 4)), gen.strides)
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels...
        self.assertEqual(0, gen.cell_anchors[0].shape[0])  # ...but no anchor
        self.assertEqual(0, gen.cell_anchors[1].shape[0])  # ...but no anchor

    def test_init_3d_fpn_anchors_no_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=((50,), (100,)),
            depths=(2, 4),
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios
        self.assertEqual(3, gen.cell_anchors[1].shape[0])  # 3 ratios

    def test_init_3d_fpn_anchors_no_custom_sizes_as_int(self):
        gen = AnchorGenerator(
            n_dim=2,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=(50, 100),
            depths=(2, 4),
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(3, gen.cell_anchors[0].shape[0])  # 3 ratios
        self.assertEqual(3, gen.cell_anchors[1].shape[0])  # 3 ratios

    def test_init_3d_fpn_only_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=(tuple(), tuple()),
            depths=(tuple(), tuple()),
            custom_anchors=((10, 10, 10), (20, 20, 20))
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(1, gen.cell_anchors[0].shape[0])  # 1 custom
        self.assertEqual(1, gen.cell_anchors[1].shape[0])  # 1 custom

    def test_init_3d_fpn_anchors_with_custom(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=(50, 100),
            depths=(2, 4),
            custom_anchors=((10, 10, 10), (20, 20, 20))
        )
        self.assertEqual(2, len(gen.cell_anchors))  # 2 levels
        self.assertEqual(4, gen.cell_anchors[0].shape[0])  # 3 ratios + 1 custom
        self.assertEqual(4, gen.cell_anchors[1].shape[0])  # 3 ratios + 1 custom

    def test_init_3d_fpn_invalid_sizes_strides_lengths(self):
        with self.assertRaises(RuntimeError):
            AnchorGenerator(
                n_dim=3,
                anchor_strides=((2, 2, 2), (4, 4, 4)),
                sizes=(50,),
            )

    def test_init_3d_fpn_invalid_sizes_strides_depths(self):
        with self.assertRaises(RuntimeError):
            AnchorGenerator(
                n_dim=3,
                anchor_strides=((2, 2, 2), (4, 4, 4)),
                sizes=(50, 100),
            )

    def test_num_anchors_per_level(self):
        gen = AnchorGenerator(
            n_dim=3,
            anchor_strides=((2, 2, 2), (4, 4, 4)),
            sizes=(50, 100),
            depths=(2, 4),
            custom_anchors=((10, 10, 10), (20, 20, 20))
        )
        self.assertEqual(4, gen.num_anchors_per_level())

    def test__grid_anchors(self):
        gen = AnchorGenerator(n_dim=2, anchor_strides=((2, 2),), sizes=tuple())
        # Dummy values
        gen.strides = ((2., 2.),)
        gen.cell_anchors = BufferList(torch.tensor([[0., 1., 3., 7.]]))  # 4x8 anchor
        grid_anchors = gen._grid_anchors(((2, 3),))
        expected = torch.tensor([[0., 1., 3., 7.],
                                 [0., 3., 3., 9.],
                                 [0., 5., 3., 11.],
                                 [2., 1., 5., 7.],
                                 [2., 3., 5., 9.],
                                 [2., 5., 5., 11.]])
        self.assertEqual(1, len(grid_anchors))
        torch.testing.assert_close(grid_anchors[0], expected, atol=1e-3, rtol=1e-7)

    def test__grid_anchors_invalid_grid_sizes(self):
        gen = AnchorGenerator(n_dim=2, anchor_strides=((2, 2),), sizes=tuple())
        # Dummy values
        gen.strides = ((2., 2.),)
        gen.cell_anchors = BufferList(torch.tensor([[0., 1., 3., 7.]]))  # 4x8 anchor
        with self.assertRaises(AssertionError):
            gen._grid_anchors(((10, 10), (20, 20)))  # Expected only one level but got two

    # noinspection DuplicatedCode
    def test__add_visibility_to(self):
        gen = AnchorGenerator(n_dim=2, anchor_strides=((2, 2),), sizes=tuple(), straddle_thresh=0)
        boxes = BoxList(
            boxes=torch.tensor([
                [0., 0., 19., 19.],  # Fully inside
                [-1., 3., 3., 9.],  # Outside
                [0., 5., 3., 20.]  # Outside
            ]),
            image_size=(20, 20),
        )
        gen._add_visibility_to(boxes)
        self.assertTrue(boxes.has_field(BoxList.PredictionField.VISIBILITY))
        visibility = boxes.get_field(BoxList.PredictionField.VISIBILITY, raise_missing=True)
        expected = torch.tensor([1, 0, 0]).bool()
        torch.testing.assert_close(visibility, expected, atol=1e-3, rtol=1e-7)

    # noinspection DuplicatedCode
    def test__add_visibility_to_negative_thr(self):
        gen = AnchorGenerator(n_dim=2, anchor_strides=((2, 2),), sizes=tuple(), straddle_thresh=-1)
        boxes = BoxList(
            boxes=torch.tensor([
                [0., 0., 19., 19.],
                [-1., 3., 3., 9.],
                [0., 5., 3., 20.]
            ]),
            image_size=(20, 20),
        )
        gen._add_visibility_to(boxes)
        self.assertTrue(boxes.has_field(BoxList.PredictionField.VISIBILITY))
        visibility = boxes.get_field(BoxList.PredictionField.VISIBILITY, raise_missing=True)
        # Should all be visible
        self.assertTrue(torch.all(visibility == 1))
