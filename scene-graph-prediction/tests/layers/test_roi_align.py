# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch

from scene_graph_prediction.layers.c_layers import roi_align_forward, roi_align_forward_3d, roi_align_backward, \
    roi_align_backward_3d


class TestROIAlign(unittest.TestCase):
    def setUp(self):
        # 1x1x8x8 feature tensor
        self.features = torch.tensor([[[[0.0887, 0.5108, 0.8036, 0.6360, 0.5853, 0.7555, 0.0553, 0.1363],
                                        [0.6111, 0.6958, 0.6473, 0.7341, 0.4384, 0.5886, 0.0918, 0.4425],
                                        [0.8974, 0.1671, 0.3077, 0.1904, 0.0285, 0.6594, 0.5133, 0.4410],
                                        [0.9345, 0.7155, 0.0675, 0.9358, 0.6166, 0.1721, 0.1548, 0.7775],
                                        [0.8592, 0.9111, 0.1174, 0.1242, 0.4167, 0.7418, 0.7754, 0.7989],
                                        [0.8776, 0.0069, 0.1957, 0.7012, 0.4614, 0.5890, 0.0199, 0.7100],
                                        [0.2145, 0.3121, 0.1924, 0.8593, 0.0336, 0.4559, 0.2707, 0.1136],
                                        [0.9737, 0.2433, 0.6548, 0.8046, 0.4264, 0.9011, 0.7527, 0.9859]]]]).cuda()

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_2d_grid(self):
        # Test with dummy positional embeddings
        horizontal = torch.arange(0, 10).repeat(10).view(10, 10).float().cuda()
        vertical = horizontal.T
        features = torch.stack([vertical, horizontal])[None]
        # Full ROI test
        rois = torch.tensor([[0, 0, 0, 9, 9]], dtype=torch.float32).cuda()
        aligned = roi_align_forward(features, rois, 1, 10, 10, 0)
        torch.testing.assert_close(aligned, features)
        # Crop test
        rois = torch.tensor([[0, 2, 4, 5, 7]], dtype=torch.float32).cuda()
        aligned = roi_align_forward(features, rois, 1, 4, 4, 0)
        expected_vertical = torch.arange(2, 6).repeat(4).view(4, 4).T.float().cuda()
        expected_horizontal = torch.arange(4, 8).repeat(4).view(4, 4).float().cuda()
        torch.testing.assert_close(aligned[0, 0], expected_vertical)
        torch.testing.assert_close(aligned[0, 1], expected_horizontal)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_3d_grid(self):
        # Test with dummy positional embeddings
        depth = torch.arange(0, 10).repeat(100).view(10, 10, 10).permute(*torch.arange(3 - 1, -1, -1))
        vertical = torch.arange(0, 10).repeat(10).view(10, 10).T.repeat(10, 1).view(10, 10, 10)
        horizontal = torch.arange(0, 10).repeat(100).view(10, 10, 10)
        features = torch.stack([depth, vertical, horizontal])[None].float().cuda()
        # Full ROI test
        rois = torch.tensor([[0, 0, 0, 0, 9, 9, 9]], dtype=torch.float32).cuda()
        aligned = roi_align_forward_3d(features, rois, 1, 1, 10, 10, 10, 0)
        torch.testing.assert_close(aligned, features)
        # Crop test
        rois = torch.tensor([[0, 2, 4, 6, 5, 7, 9]], dtype=torch.float32).cuda()
        aligned = roi_align_forward_3d(features, rois, 1, 1, 4, 4, 4, 0)
        expected_depth = torch.arange(2, 6).repeat(16).view(4, 4, 4).T.float().cuda()
        expected_vertical = torch.arange(4, 8).repeat(4).view(4, 4).T.repeat(4, 1).view(4, 4, 4).float().cuda()
        expected_horizontal = torch.arange(6, 10).repeat(16).view(4, 4, 4).float().cuda()
        torch.testing.assert_close(aligned[0, 0], expected_depth)
        torch.testing.assert_close(aligned[0, 1], expected_vertical)
        torch.testing.assert_close(aligned[0, 2], expected_horizontal)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_2d_vs_3d(self):
        """Check that 2D == 3D."""

        rois = torch.tensor([[0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        aligned = roi_align_forward(self.features, rois, 1, 8, 8, 0)

        rois3d = torch.tensor([[0, 0, 0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        aligned3d = roi_align_forward_3d(self.features[None], rois3d, 1, 1, 1, 8, 8, 0)

        torch.testing.assert_close(aligned, aligned3d[0])

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_3d(self):
        """Check that 1x8x8 == 8x1x8 == 8x8x1."""

        rois3d_d = torch.tensor([[0, 0, 0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        aligned3d_d = roi_align_forward_3d(self.features.reshape((1, 1, 1, 8, 8)), rois3d_d, 1, 1, 1, 8, 8, 0)
        rois3d_h = torch.tensor([[0, 0, 0, 0, 7, 0, 7]], dtype=torch.float32).cuda()  # Full RoI
        aligned3d_h = roi_align_forward_3d(self.features.reshape((1, 1, 8, 1, 8)), rois3d_h, 1, 1, 8, 1, 8, 0)
        rois3d_w = torch.tensor([[0, 0, 0, 0, 7, 7, 0]], dtype=torch.float32).cuda()  # Full RoI
        aligned3d_w = roi_align_forward_3d(self.features.reshape((1, 1, 8, 8, 1)), rois3d_w, 1, 1, 8, 8, 1, 0)

        torch.testing.assert_close(aligned3d_d.reshape(-1), aligned3d_h.reshape(-1))
        torch.testing.assert_close(aligned3d_d.reshape(-1), aligned3d_w.reshape(-1))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_backward_2d_vs_3d(self):
        """Check that 2D == 3D."""

        rois = torch.tensor([[0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        grad_input = roi_align_backward(self.features, rois, 1, 8, 8, 1, 1, 8, 8, 1)

        rois3d = torch.tensor([[0, 0, 0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        grad_input3d = roi_align_backward_3d(self.features[None], rois3d, 1, 1, 1, 8, 8, 1, 1, 1, 8, 8, 1)

        torch.testing.assert_close(grad_input, grad_input3d[0])

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_roi_align_backward_3d(self):
        """Check that 1x8x8 == 8x1x8 == 8x8x1."""

        rois3d_d = torch.tensor([[0, 0, 0, 0, 0, 7, 7]], dtype=torch.float32).cuda()  # Full RoI
        grad_input3d_d = roi_align_backward_3d(self.features.reshape((1, 1, 1, 8, 8)), rois3d_d,
                                               1, 1, 1, 8, 8, 1, 1, 1, 8, 8, 1)
        rois3d_h = torch.tensor([[0, 0, 0, 0, 7, 0, 7]], dtype=torch.float32).cuda()  # Full RoI
        grad_input3d_h = roi_align_backward_3d(self.features.reshape((1, 1, 8, 1, 8)), rois3d_h,
                                               1, 1, 8, 1, 8, 1, 1, 8, 1, 8, 1)
        rois3d_w = torch.tensor([[0, 0, 0, 0, 7, 7, 0]], dtype=torch.float32).cuda()  # Full RoI
        grad_input3d_w = roi_align_backward_3d(self.features.reshape((1, 1, 8, 8, 1)), rois3d_w,
                                               1, 1, 8, 8, 1, 1, 1, 8, 8, 1, 1)

        torch.testing.assert_close(grad_input3d_d.reshape(-1), grad_input3d_h.reshape(-1))
        torch.testing.assert_close(grad_input3d_d.reshape(-1), grad_input3d_w.reshape(-1))
