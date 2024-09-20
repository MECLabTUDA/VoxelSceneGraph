import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.keypoint_head import KeypointHeadTargets, KeypointLogits
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList


class KeypointRCNNLossComputation(torch.nn.Module):
    """Note: supports ND."""

    def __init__(self, discretization_size: int):
        super().__init__()
        self.discretization_size = discretization_size

    def forward(self, proposals: KeypointHeadTargets, keypoint_logits: KeypointLogits) -> torch.Tensor:
        heatmaps = []
        valid = []
        for proposals_per_image in proposals:
            kp = proposals_per_image.KEYPOINTS

            proposals = proposals_per_image.convert(BoxList.Mode.zyxzyx)
            heatmaps_per_image, valid_per_image = kp.to_heat_map(
                kp.keypoints, proposals.boxes, self.discretization_size
            )

            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))

        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0)
        valid = torch.nonzero(valid).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't  accept empty tensors, so handle it separately
        if keypoint_targets.numel() == 0 or len(valid) == 0:
            return keypoint_logits.sum() * 0

        n, k, *lengths = keypoint_logits.shape
        keypoint_logits = keypoint_logits.view(n * k, np.prod(lengths))

        keypoint_loss = torch.nn.functional.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])

        return keypoint_loss


def build_roi_keypoint_loss_evaluator(cfg: CfgNode) -> KeypointRCNNLossComputation:
    return KeypointRCNNLossComputation(cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION)
