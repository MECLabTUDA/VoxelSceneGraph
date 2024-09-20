import unittest

import torch

from scene_graph_api.utils.tensor import affine_transformation_grid, compute_bounding_box


# noinspection DuplicatedCode
class TestAffineTransformationGrid(unittest.TestCase):
    def test_no_op_2d(self):
        tensor = torch.tensor(list(range(4))).float().view((2, 2))
        grid = affine_transformation_grid(tensor.shape)
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        torch.testing.assert_close(transformed, tensor)

    def test_no_op_3d(self):
        tensor = torch.tensor(list(range(8))).float().view((2, 2, 2))
        grid = affine_transformation_grid(tensor.shape)
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        torch.testing.assert_close(transformed, tensor)

    def test_translate_2d(self):
        tensor = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
        ]).float()
        grid = affine_transformation_grid(tensor.shape, translate=(-1, 1))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [0, 0, 1, 2],
            [0, 0, 3, 4],
            [0, 0, 0, 0],
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_scale_2d_y_only(self):
        tensor = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]).float()
        grid = affine_transformation_grid(tensor.shape, scale=(2, 1))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([[1.0000, 0.7500, 0.2500, 0.0000],
                                 [1.0000, 0.7500, 0.2500, 0.0000],
                                 [0.0000, 0.0000, 0.0000, 0.0000],
                                 [0.0000, 0.0000, 0.0000, 0.0000]]).T
        torch.testing.assert_close(transformed, expected)

    def test_scale_2d_x_only(self):
        tensor = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]).float()
        grid = affine_transformation_grid(tensor.shape, scale=(1, 2))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.7500, 1.0000, 1.0000, 0.7500],
            [0.7500, 1.0000, 1.0000, 0.7500],
            [0.0000, 0.0000, 0.0000, 0.0000]
        ])
        torch.testing.assert_close(transformed, expected)

    def test_scale_2d_smaller(self):
        tensor = torch.ones((10, 10))
        grid = affine_transformation_grid(tensor.shape, scale=(.5, .5))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.2500, 0.5000, 0.5000, 0.5000, 0.5000, 0.2500, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.2500, 0.5000, 0.5000, 0.5000, 0.5000, 0.2500, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ])
        torch.testing.assert_close(transformed, expected)

    def test_rotate_2d(self):
        tensor = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 4, 5, 6, 0],
            [0, 7, 8, 9, 0],
            [0, 0, 0, 0, 0],
        ]).float()
        grid = affine_transformation_grid(tensor.shape, rotate=(90,))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 3, 6, 9, 0],
            [0, 2, 5, 8, 0],
            [0, 1, 4, 7, 0],
            [0, 0, 0, 0, 0],
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_rotate_3d_around_z_only(self):
        tensor = torch.tensor([
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
            [[9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]],
            [[18, 19, 20],
             [21, 22, 23],
             [24, 25, 26]]
        ]).float()
        grid = affine_transformation_grid(tensor.shape, rotate=(90, 0, 0))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [[2, 5, 8],
             [1, 4, 7],
             [0, 3, 6]],
            [[11, 14, 17],
             [10, 13, 16],
             [9, 12, 15]],
            [[20, 23, 26],
             [19, 22, 25],
             [18, 21, 24]]
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_rotate_3d_around_y_only(self):
        tensor = torch.tensor([
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
            [[9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]],
            [[18, 19, 20],
             [21, 22, 23],
             [24, 25, 26]]
        ]).float()
        grid = affine_transformation_grid(tensor.shape, rotate=(0, 90, 0))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [[2, 11, 20],
             [5, 14, 23],
             [8, 17, 26]],
            [[1, 10, 19],
             [4, 13, 22],
             [7, 16, 25]],
            [[0, 9, 18],
             [3, 12, 21],
             [6, 15, 24]]
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_rotate_3d_around_x_only(self):
        tensor = torch.tensor([
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
            [[9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]],
            [[18, 19, 20],
             [21, 22, 23],
             [24, 25, 26]]
        ]).float()
        grid = affine_transformation_grid(tensor.shape, rotate=(0, 0, 90))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [[18, 19, 20],
             [9, 10, 11],
             [0, 1, 2]],
            [[21, 22, 23],
             [12, 13, 14],
             [3, 4, 5]],
            [[24, 25, 26],
             [15, 16, 17],
             [6, 7, 8]]
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_rotate_2d_all_at_once(self):
        tensor = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ]).float()
        grid = affine_transformation_grid(tensor.shape, scale=(2, 1), rotate=(90,), translate=(-1, 0))
        transformed = torch.nn.functional.grid_sample(tensor[None, None], grid, align_corners=False)[0, 0]
        expected = torch.tensor([
            [1.5000, 2.0000, 2.0000, 1.5000],
            [0.7500, 1.0000, 1.0000, 0.7500],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000]
        ]).float()
        torch.testing.assert_close(transformed, expected)

    def test_invalid_tensor_size(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 0))

    def test_invalid_scale_value(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 1), scale=(0, 0))

    def test_translate_length(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 1), translate=(1, 1, 1))

    def test_scale_length(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 1), scale=(1, 1, 1))

    def test_rotate_length_2d(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 1), rotate=(1, 1))

    def test_rotate_length_3d(self):
        with self.assertRaises(AssertionError):
            affine_transformation_grid((1, 1, 1), rotate=(1, 1))


class TestComputeBoundingBox(unittest.TestCase):
    def test_non_zero_2d(self):
        tensor = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]).bool()
        bb = compute_bounding_box(tensor)
        torch.testing.assert_close(bb, torch.tensor([1, 1, 2, 1]))

    def test_non_zero_3d(self):
        tensor = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0] * 3).view((3, 3, 3)).bool()
        bb = compute_bounding_box(tensor)
        torch.testing.assert_close(bb, torch.tensor([0, 1, 0, 2, 1, 2]))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_non_zero_cuda(self):
        tensor = torch.ones((3, 3)).cuda().bool()
        bb = compute_bounding_box(tensor)
        torch.testing.assert_close(bb, torch.tensor([0, 0, 2, 2]).cuda())

    def test_zero(self):
        """Important to check that the tensor is not all zero for the torch.where calls."""
        tensor = torch.zeros((2, 2)).bool()
        bb = compute_bounding_box(tensor)
        torch.testing.assert_close(bb, torch.tensor([0, 0, -1, -1]))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_zero_cuda(self):
        tensor = torch.zeros((2, 2)).bool().cuda()
        bb = compute_bounding_box(tensor)
        torch.testing.assert_close(bb, torch.tensor([0, 0, -1, -1]).cuda())
