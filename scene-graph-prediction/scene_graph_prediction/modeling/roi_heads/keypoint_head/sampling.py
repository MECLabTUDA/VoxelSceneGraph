from functools import reduce

import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.keypoint_head import KeypointHeadTargets
from scene_graph_prediction.modeling.abstractions.matcher import Matcher
from scene_graph_prediction.modeling.abstractions.region_proposal import RPNProposals
from scene_graph_prediction.modeling.utils import BalancedSampler


class KeypointRCNNSampling:
    """Note: supports ND."""

    def __init__(self, n_dim: int, fg_bg_sampler: BalancedSampler):
        self.n_dim = n_dim
        self.fg_bg_sampler = fg_bg_sampler

    def _prepare_targets(
            self, proposals: RPNProposals, targets: KeypointHeadTargets
    ) -> tuple[list[torch.LongTensor], list[torch.Tensor]]:
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_idxs = proposals_per_image.MATCHED_IDXS
            matched_targets = targets_per_image[matched_idxs.clamp(min=0)]

            labels_per_image = matched_targets.LABELS.long()

            keypoints_per_image = matched_targets.KEYPOINTS
            within_box = self._within_box(keypoints_per_image.keypoints, matched_targets.boxes)
            vis_kp = keypoints_per_image.keypoints[..., self.n_dim] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0

            labels_per_image[~is_visible] = Matcher.BELOW_LOW_THRESHOLD

            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)

        # Note: the labels or only used for the sampler
        return labels, keypoints

    def subsample(self, proposals: RPNProposals, targets: KeypointHeadTargets) -> list[torch.BoolTensor]:
        """This method performs the positive/negative sampling, and return the sampled proposals."""

        labels, keypoints = self._prepare_targets(proposals, targets)

        # Add corresponding label and regression_targets information to the bounding boxes
        for keypoints_per_image, proposals_per_image in zip(keypoints, proposals):
            proposals_per_image.KEYPOINTS = keypoints_per_image

        sampled_pos_masks, _ = self.fg_bg_sampler(labels)
        return sampled_pos_masks

    def _within_box(self, points: torch.Tensor, boxes: torch.Tensor) -> torch.LongTensor:
        """
        Validate which keypoints are contained inside any detected box.
        :param points: NxKx(n_dim + 1)
        :param boxes: Nx(2*n_dim)
        :returns: NxK
        """

        within = [points[..., dim] >= boxes[:, dim, None] for dim in range(self.n_dim)] + \
                 [points[..., dim] <= boxes[:, self.n_dim + dim, None] + 1 for dim in range(self.n_dim)]

        # noinspection PyTypeChecker
        return reduce(lambda a, b: a & b, within)


def build_roi_keypoint_samp_processor(cfg: CfgNode) -> KeypointRCNNSampling:
    return KeypointRCNNSampling(
        cfg.INPUT.N_DIM,
        BalancedSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    )
