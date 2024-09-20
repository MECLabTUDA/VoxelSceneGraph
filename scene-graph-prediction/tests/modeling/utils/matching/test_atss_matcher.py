import unittest

import torch

from scene_graph_prediction.modeling.utils import ATSSMatcher
from scene_graph_prediction.structures import BoxList


class TestATSSMatcher(unittest.TestCase):
    def test_basic(self):
        matcher = ATSSMatcher(num_anchors_per_lvl=1, num_candidates=1)
        target = BoxList(
            torch.tensor([
                [10, 10, 19, 19],
                [0, 0, 4, 9],
            ]).float(),
            (50, 50)
        )
        anchors = BoxList(
            torch.tensor([
                [0, 0, 4, 9],
                [0, 0, 19, 19],
                [20, 20, 30, 30],
            ]).float(),
            (50, 50)
        )
        anchors.add_field(BoxList.PredictionField.ANCHOR_LVL, torch.tensor([0, 0, 0]))
        matches = matcher(target, anchors)

        expected_matches = torch.tensor([1, 0, -1]).long()
        torch.testing.assert_close(matches, expected_matches)

    def test_keep_best_match(self):
        matcher = ATSSMatcher(num_anchors_per_lvl=1, num_candidates=1)
        target = BoxList(
            torch.tensor([
                [0, 0, 4, 9],
                [0, 0, 4, 10],
            ]).float(),
            (50, 50)
        )
        anchors = BoxList(
            torch.tensor([
                [0, 0, 4, 9],
            ]).float(),
            (50, 50)
        )
        anchors.add_field(BoxList.PredictionField.ANCHOR_LVL, torch.tensor([0]))
        matches = matcher(target, anchors)

        expected_matches = torch.tensor([0]).long()
        torch.testing.assert_close(matches, expected_matches)
