import unittest

import torch

from scene_graph_prediction.modeling.region_proposal.retinaunet.loss import RetinaUNetSegLossComputation


class TestBackbones(unittest.TestCase):
    def test_dice_loss_perfect(self):
        loss = RetinaUNetSegLossComputation(2, 2)
        gt_seg = torch.tensor([[1, 0], [1, 1]]).view(4).long()
        logits = torch.tensor([[[-1, 1], [-1, -1]], [[1, -1], [1, 1]]]).view(2, 4).permute((1, 0)).float() * 2
        # noinspection PyTypeChecker
        self.assertLessEqual(loss._squared_dice_loss(logits, gt_seg), 0.015)

    def test_dice_loss_pred_all_bg(self):
        loss = RetinaUNetSegLossComputation(2, 2)
        gt_seg = torch.tensor([[1, 0], [1, 1]]).view(4).long()
        logits = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]).view(2, 4).permute((1, 0)).float() * 2
        # noinspection PyTypeChecker
        self.assertGreaterEqual(loss._squared_dice_loss(logits, gt_seg), 0.95)
