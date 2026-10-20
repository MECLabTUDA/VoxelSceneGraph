import unittest

import torch

from scene_graph_prediction.modeling.utils import IoUMatcher
from scene_graph_prediction.structures import BoxList


class TestIoUMatcher(unittest.TestCase):
    def test_no_target(self):
        matcher = IoUMatcher(.5, .25, always_keep_best_match=False)
        target = BoxList(torch.empty((0, 4)), (50, 50))
        proposal = BoxList(torch.empty((1, 4)), (50, 50))
        with self.assertRaises(ValueError):
            matcher(target, proposal)

    def test_no_proposal(self):
        matcher = IoUMatcher(.5, .25, always_keep_best_match=False)
        target = BoxList(torch.empty((1, 4)), (50, 50))
        proposal = BoxList(torch.empty((0, 4)), (50, 50))
        with self.assertRaises(ValueError):
            matcher(target, proposal)

    def test_box_decoder_no_keep(self):
        matcher = IoUMatcher(.5, .25, always_keep_best_match=False)

        # Predictions are: great (0), borderline (2), in between (-2), below (-1), multiple good (3, 1)
        sim_matrix = torch.tensor([
            .7, 0, 0, 0, 0,
            0, 0, .25, 0, .6,
            0, .5, 0, 0, 0,
            0, 0, 0, .2, .7,
        ]).view(4, 5)  # 4 GT boxes and 5 pred

        matches = matcher._parse_quality_matrix(sim_matrix)
        expected_matches = torch.tensor([0, 2, -2, -1, 3])
        torch.testing.assert_close(matches, expected_matches)

    def test_box_decoder_keep(self):
        matcher = IoUMatcher(.5, .25, always_keep_best_match=True)

        # Pred box 0 is the best match of GT box 1
        # Pred box 1 is the best match of GT boxes 0 and 2; only he last best match (2) is kept
        sim_matrix = torch.tensor([
            0, .2,
            .2, 0,
            0, .2,
        ]).view(3, 2)  # 4 GT boxes and 2 pred

        matches = matcher._parse_quality_matrix(sim_matrix)
        expected_matches = torch.tensor([1, 2])
        torch.testing.assert_close(matches, expected_matches)
