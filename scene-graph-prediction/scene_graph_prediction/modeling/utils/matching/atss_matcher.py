# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from scene_graph_prediction.modeling.abstractions.matcher import Matcher as AbstractMatcher
from scene_graph_prediction.structures import BoxList, BoxListOps


class ATSSMatcher(AbstractMatcher):
    """
    Compute matches according to ATTS for a single image. Adapted from:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/atss_assigner.py
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/matcher/atss.py
    """
    _INF = 1000

    def __init__(self, num_anchors_per_lvl: int, num_candidates: int = 4):
        assert num_candidates > 0
        self.num_anchors_per_lvl = num_anchors_per_lvl
        self.num_candidates = num_candidates
        self.min_dist = 0.01

    def _match(self, target: BoxList, anchors: BoxList) -> torch.LongTensor:
        """
        Matches elements between groundtruth (M boxes) and proposal (N boxes). The tensor needs at least one element.
        Note: requires the ANCHOR_LVL field.
        :returns: An N tensor where N[i] is a matched gt in [0, M - 1] or
                  a negative value indicating that prediction could not be matched.
        """
        centers_target = BoxListOps.centers(target)
        centers_anchors = BoxListOps.centers(anchors)
        center_dists: torch.Tensor = (centers_target[:, None] - centers_anchors[None]).pow(2).sum(-1).sqrt()

        # Select candidates based on center distance
        candidate_idxs = []
        anchor_levels = anchors.ANCHOR_LVL
        for lvl in torch.unique(anchor_levels):
            # noinspection PyTypeChecker
            selectable_anchors = torch.nonzero(anchor_levels == lvl).squeeze(1)
            max_k = min(self.num_candidates * self.num_anchors_per_lvl, selectable_anchors.numel())
            _, selected_idxs = center_dists[:, selectable_anchors].topk(max_k, dim=1, largest=False)
            candidate_idxs.append(selectable_anchors[selected_idxs])  # N x max_k

        candidate_idxs = torch.cat(candidate_idxs, dim=1)  # N x #candidates

        match_quality_matrix = BoxListOps.iou(target, anchors)
        candidate_overlaps = match_quality_matrix.gather(1, candidate_idxs)  # N x #candidates

        # Compute adaptive iou threshold
        overlaps_mean_per_gt = candidate_overlaps.mean(dim=1)  # N
        # If there's only one candidate for a given target box, then the std is nan
        overlaps_std_per_gt = torch.nan_to_num(candidate_overlaps.std(dim=1))  # N
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[:, None]  # N x #candidates

        # In case one anchor is assigned to multiple boxes, use box with highest IoU
        # We do this by first offsetting the candidate_idxs, such that they now point to appropriate score in sim matrix
        num_anchors = len(anchors)
        for target_idx in range(len(target)):
            candidate_idxs[target_idx, :] += target_idx * num_anchors
        # We initialize a score matrix with -INF scores (-1 would be low enough with IoU)
        # and then copy the scores for positive matches
        overlaps_inf = torch.full_like(match_quality_matrix, -self._INF).view(-1)
        # noinspection PyUnresolvedReferences
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = match_quality_matrix.view(-1)[index]
        overlaps_inf = overlaps_inf.view_as(match_quality_matrix)

        # Then a max call along dim 0 to only keep the best match per target box
        matched_vals, matches = overlaps_inf.max(dim=0)
        # Finally, all negative boxes get the appropriate BELOW_LOW_THRESHOLD match
        matches[matched_vals == -self._INF] = self.BELOW_LOW_THRESHOLD

        return matches
