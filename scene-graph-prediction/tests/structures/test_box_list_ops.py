# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest

import torch

from scene_graph_prediction.structures import BoxList, BoxListOps


# noinspection DuplicatedCode
class TestBoxListOps(unittest.TestCase):

    def test_generalized_iou(self):
        boxes1 = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ])
        boxlist1 = BoxList(boxes1, (20, 20, 20), BoxList.Mode.zyxzyx)
        boxes2 = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
        ])
        boxlist2 = BoxList(boxes2, (20, 20, 20), BoxList.Mode.zyxzyx)
        gen_iou = BoxListOps.generalized_iou(boxlist1, boxlist2)
        # Compare values to implementation of nnDetection
        self.assertAlmostEqual(gen_iou[0].item(), 1., places=3)
        self.assertAlmostEqual(gen_iou[1].item(), -.75, places=3)
        self.assertAlmostEqual(gen_iou[2].item(), -.9259, places=3)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_nms_classwise_cuda(self):
        boxes = torch.tensor([
            [0, 0, 8, 9],  # Base box, label 0, high score
            [0, 0, 9, 9],  # Box full overlap with box 0, but label 1, high score
            [5, 5, 14, 14],  # Box overlap with box 1, label 1, low score
        ]).float().cuda()
        scores = torch.tensor([.1, .9, .2]).cuda()
        labels = torch.tensor([0, 1, 1]).long().cuda()

        boxlist = BoxList(boxes, (15, 15), BoxList.Mode.zyxzyx)
        boxlist.PRED_SCORES = scores
        boxlist.PRED_LABELS = labels

        # Check that plain nms fails
        post_nms, keep = BoxListOps.nms(
            boxlist,
            .1, -1,
            BoxList.PredictionField.PRED_SCORES
        )
        self.assertNotEqual(2, len(post_nms))  # Removes one too many boxes

        # Check that class-wise nms works
        orig_boxes = boxes.clone()
        post_nms, keep = BoxListOps.nms_classwise(
            boxlist,
            .1, -1,
            BoxList.PredictionField.PRED_SCORES,
            BoxList.PredictionField.PRED_LABELS
        )

        self.assertEqual(2, len(post_nms))
        # Note: also checks that boxes are ordered by descending score
        torch.testing.assert_close(keep, torch.tensor([1, 0]).cuda())

        # Check that any offset that has been added, has also been removed
        torch.testing.assert_close(boxlist.boxes, orig_boxes)
        torch.testing.assert_close(post_nms.boxes, orig_boxes[keep])
