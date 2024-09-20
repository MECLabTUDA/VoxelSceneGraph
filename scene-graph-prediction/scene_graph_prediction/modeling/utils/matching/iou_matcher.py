# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.abstractions.matcher import Matcher as AbstractMatcher
from scene_graph_prediction.structures import BoxList, BoxListOps


class IoUMatcher(AbstractMatcher):
    def __init__(self, high_threshold: float, low_threshold: float = .33, always_keep_best_match: bool = False):
        """
        :param high_threshold: Quality values greater than or equal to this value are candidate matches.
        :param low_threshold: A lower quality threshold used to stratify matches into three levels:
                               1) matches >= high_threshold
                               2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                               3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
        :param always_keep_best_match: If True, keep the best match for each GT box, even if the matching score is low.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.always_keep_best_match = always_keep_best_match

    def _match(self, target: BoxList, proposal: BoxList) -> torch.LongTensor:
        """
        Matches elements between groundtruth (M boxes) and proposal (N boxes). The tensor needs at least one element.
        :return: matches tensor: An N tensor where N[i] is a matched gt in [0, M - 1] or
                                  a negative value indicating that prediction could not be matched.
        """
        return self._parse_quality_matrix(BoxListOps.iou(target, proposal))

    def _parse_quality_matrix(self, match_quality_matrix: torch.FloatTensor) -> torch.LongTensor:
        """Private method split off of self._match to allow for easier testing..."""
        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)

        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.always_keep_best_match:
            # noinspection PyUnboundLocalVariable
            self._set_low_quality_matches(matches, match_quality_matrix)

        return matches

    @staticmethod
    def _set_low_quality_matches(matches: torch.LongTensor, match_quality_matrix: torch.FloatTensor):
        """
        Produces additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that
        have maximum overlap with it (excluding ties);
        for each prediction in that set, if it is unmatched,
        then match it to the ground-truth with which it has the highest quality value.
        """
        # For each gt, find the prediction with which it has the highest quality
        _, best_pred_idx = match_quality_matrix.max(dim=1)
        matches[best_pred_idx] = torch.arange(len(best_pred_idx)).to(matches)
