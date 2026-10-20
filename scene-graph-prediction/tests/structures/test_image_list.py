import unittest

import torch

from scene_graph_prediction.structures import ImageList


class TestImageList(unittest.TestCase):
    def test_to_image_list_from_image_list(self):
        imgs = ImageList(torch.empty((1, 3, 5, 7, 9)), [(5, 7, 9)])
        imgs2 = ImageList.to_image_list(imgs, 3, size_divisible=(0, 0, 0))
        self.assertEqual(imgs.n_dim, imgs2.n_dim)
        self.assertTrue(imgs2.tensors is imgs.tensors)
        self.assertEqual(imgs.image_sizes, imgs2.image_sizes)

    def test_to_image_list_single_tensor_size_div_all_zero(self):
        tensor = torch.empty((3, 5, 7, 9))  # CxDxHxW
        imgs = ImageList.to_image_list(tensor, 3, size_divisible=(0, 0, 0))
        self.assertEqual(1, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((3, 5, 7, 9), imgs.tensors[0].shape)

    def test_to_image_list_single_tensor_size_div_one_zero(self):
        tensor = torch.empty((3, 5, 7, 9))  # CxDxHxW
        imgs = ImageList.to_image_list(tensor, 3, size_divisible=(0, 16, 32))
        self.assertEqual(1, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((3, 5, 16, 32), imgs.tensors[0].shape)  # Check padding

    def test_to_image_list_single_tensor_size_div(self):
        tensor = torch.empty((3, 5, 7, 9))  # CxDxHxW
        imgs = ImageList.to_image_list(tensor, 3, size_divisible=(8, 16, 32))
        self.assertEqual(1, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((3, 8, 16, 32), imgs.tensors[0].shape)  # Check padding

    def test_to_image_list_single_tensor_size_div_already_good(self):
        tensor = torch.empty((3, 6, 10, 15))  # CxDxHxW
        imgs = ImageList.to_image_list(tensor, 3, size_divisible=(3, 5, 5))
        self.assertEqual(1, len(imgs))
        self.assertEqual((6, 10, 15), imgs.image_sizes[0])
        self.assertEqual((3, 6, 10, 15), imgs.tensors[0].shape)

    def test_to_image_list_list_tensors_size_div_all_zero(self):
        tensors = [torch.empty((3, 5, 7, 9)), torch.empty((3, 10, 14, 18))]
        imgs = ImageList.to_image_list(tensors, 3, size_divisible=(0, 0, 0))
        self.assertEqual(2, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((10, 14, 18), imgs.image_sizes[1])
        # Check that the smaller tensor has been padded to the size of the largest one
        self.assertEqual((3, 10, 14, 18), imgs.tensors[0].shape)
        self.assertEqual((3, 10, 14, 18), imgs.tensors[1].shape)

    def test_to_image_list_list_tensors_size_div(self):
        tensors = [torch.empty((3, 5, 7, 9)), torch.empty((3, 10, 14, 18))]
        imgs = ImageList.to_image_list(tensors, 3, size_divisible=(4, 5, 20))
        self.assertEqual(2, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((10, 14, 18), imgs.image_sizes[1])
        # Check padding
        self.assertEqual((3, 12, 15, 20), imgs.tensors[0].shape)
        self.assertEqual((3, 12, 15, 20), imgs.tensors[1].shape)

    def test_to_image_list_list_tensor_batch(self):
        tensors = torch.empty((2, 3, 5, 7, 9))
        imgs = ImageList.to_image_list(tensors, 3, size_divisible=(4, 5, 20))
        self.assertEqual(2, len(imgs))
        self.assertEqual((5, 7, 9), imgs.image_sizes[0])
        self.assertEqual((5, 7, 9), imgs.image_sizes[1])
        # Check padding
        self.assertEqual((3, 8, 10, 20), imgs.tensors[0].shape)
        self.assertEqual((3, 8, 10, 20), imgs.tensors[1].shape)

    def test_ith_image_as_image_list(self):
        imgs = ImageList(torch.randn((3, 3, 5, 7, 9)), [(5, 7, 9)] * 3)
        img = imgs.ith_image_as_image_list(1)
        self.assertEqual(imgs.n_dim, img.n_dim)
        self.assertEqual((1, 3, 5, 7, 9), img.tensors.shape)
        torch.testing.assert_close(imgs.tensors[1], img.tensors[0])
