import torch
from torchvision.transforms.functional import resize, InterpolationMode
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.keypoint_head import KeypointHeadTargets, KeypointLogits, \
    KeypointHeadProposals
from scene_graph_prediction.structures import PersonKeypoints


# TODO test for non-regression in Keypointer after conversion to torchvision implementation
class _Keypointer(torch.nn.Module):
    """
    Projects a set of masks in an image on the locations specified by the bounding boxes.
    Note: support 2D and 3D.
    """

    def __init__(self, n_dim: int):
        super().__init__()
        self.n_dim = n_dim
        assert self.n_dim in [2, 3]

    def forward(
            self, masks: KeypointLogits, boxes: KeypointHeadTargets
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Extract predicted keypoint locations from heatmaps.
        :returns: Output has shape (#rois, n_dim + 1, #keypoints), (#rois, #keypoints) with the n_dim + 2 rows (per roi)
                  corresponding to `return ((x, y, z,... logit), prob)` for each keypoint.
        """
        predictions, scores = [], []
        for mask, box in zip(masks, boxes):
            pred, score = self._heatmaps_to_keypoints(mask[None].detach(), box.boxes)
            predictions.append(pred)
            scores.append(score)

        return predictions, scores

    def _heatmaps_to_keypoints(self, maps: KeypointLogits, rois: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Only supports a batch size of 1."""

        # This function converts a discrete image coordinate in a HEATMAP_SIZE x HEATMAP_SIZE image
        # to a continuous keypoint coordinate. We maintain consistency with keypoints_to_heatmap_labels by using the
        # conversion from Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

        offsets = [rois[:, dim] for dim in range(self.n_dim)]

        lengths = [rois[:, self.n_dim + dim] - rois[:, dim] for dim in range(self.n_dim)]
        lengths = [torch.maximum(length, torch.Tensor([1])) for length in lengths]
        lengths_ceil = [torch.ceil(length) for length in lengths]

        min_size = 0  # cfg.KRCNN.INFERENCE_MIN_SIZE
        num_keypoints = maps.shape[-1]
        predictions = torch.zeros((len(rois), self.n_dim + 1, num_keypoints), dtype=torch.float32)
        end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32)
        for i in range(len(rois)):
            if min_size > 0:
                roi_map_lengths = [int(torch.maximum(length[i], torch.Tensor([min_size]))) for length in lengths]
            else:
                roi_map_lengths = [length[i] for length in lengths_ceil]
            roi_map = resize(maps[i], list(roi_map_lengths), interpolation=InterpolationMode.BICUBIC)

            pos = roi_map.reshape(num_keypoints, -1).argmax(axis=1)  # 1D pixel pos in 1 channel image
            if self.n_dim == 2:
                # 2D
                w = roi_map.shape[2]
                zyx_int = pos // w, pos % w
            else:
                # 3D
                h, w = roi_map.shape[-2], roi_map.shape[-1]
                zyx_int = pos // (h * w), (pos / w) % h, pos % w,

            lengths_correction = [lengths[dim][i] / roi_map_lengths[dim] for dim in range(self.n_dim)]
            zyx = [(zyx_int[dim] + .5) * lengths_correction[dim] for dim in range(self.n_dim)]
            for dim in range(self.n_dim):
                predictions[i, dim, :] = zyx[dim] + offsets[dim]
            predictions[i, self.n_dim, :] = 1
            end_scores[i, :] = roi_map[(torch.arange(num_keypoints),) + tuple(zyx_int)]

        return predictions, end_scores


class KeypointPostProcessor(torch.nn.Module):
    def __init__(self, keypointer: _Keypointer):
        super().__init__()
        self.keypointer = keypointer
        self.n_dim = self.keypointer.n_dim

    def forward(self, x: KeypointLogits, boxes: KeypointHeadTargets) -> KeypointHeadProposals:
        mask_prob, scores = self.keypointer(x, boxes)

        for prob, bbox, score in zip(mask_prob, boxes, scores):
            # FIXME choose the appropriate set of keypoints here
            prob = PersonKeypoints(prob, bbox.size)
            prob.KEYPOINT_LOGITS = score
            bbox.KEYPOINTS = prob

        return boxes


def build_roi_keypoint_post_processor(cfg: CfgNode) -> KeypointPostProcessor:
    return KeypointPostProcessor(_Keypointer(cfg.INPUT.N_DIM))
