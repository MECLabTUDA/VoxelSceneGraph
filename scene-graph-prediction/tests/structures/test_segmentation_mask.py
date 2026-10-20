import unittest

import torch

from scene_graph_prediction.structures import BinaryMaskList, MaskListView


class TestMaskListView(unittest.TestCase):
    def test_indexing(self):
        # Test we can do whatever operations on the view, and it is behaving like we would expect the mask list to
        masks = BinaryMaskList(torch.rand((4, 2, 3, 4)).float(), (2, 3, 4))
        view = MaskListView(masks)

        masks = masks[::2].resize((4, 6, 8)).crop((1, 1, 1, 2, 3, 4))
        view = view[::2].resize((4, 6, 8)).crop((1, 1, 1, 2, 3, 4))

        m_tensor = masks.get_mask_tensor()
        v_tensor = view.get_mask_tensor()
        self.assertTrue(torch.allclose(m_tensor, v_tensor))
