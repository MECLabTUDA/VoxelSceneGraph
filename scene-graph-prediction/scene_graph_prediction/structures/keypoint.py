from functools import reduce

import torch
from scene_graph_api.tensor_structures import Keypoints as _Keypoints, PersonKeypoints as _PersonKeypoints
from typing_extensions import Self

_SIZE_T = tuple[int, ...]


class Keypoints(_Keypoints):
    """Note: supports ND."""

    def resize(self, size: _SIZE_T) -> Self:
        assert self.n_dim == len(size), f"Dimension mismatch, expected {self.n_dim} but got {len(size)}."
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        resized_data = self.keypoints.clone()
        for i in range(self.n_dim):
            resized_data[..., i] *= ratios[i]
        keypoints = type(self)(resized_data, size)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    # Tensor-like methods
    def to(self, *args, **kwargs) -> Self:
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def to_heat_map(
            self,
            rois: torch.Tensor,
            heatmap_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self.size) == rois.dim(), f"Dimension mismatch, expected {len(self.size)} but got {rois.dim()}."
        if rois.numel() == 0:
            return rois.new().long(), rois.new().long()

        offsets = [rois[:, i] for i in range(self.n_dim)]
        scales = [heatmap_size / (rois[:, i + self.n_dim] - rois[:, i]) for i in range(self.n_dim)]

        offsets = [o[:, None] for o in offsets]
        scales = [s[:, None] for s in scales]

        # noinspection PyTypeChecker
        coordinates = [self.keypoints[..., i] for i in range(self.n_dim)]

        coord_boundary_indexes = [coordinates[i] == rois[:, i + self.n_dim][:, None] for i in range(self.n_dim)]

        coordinates = [(coord - offset) * scale for coord, offset, scale in zip(coordinates, offsets, scales)]
        coordinates = [c.floor().long() for c in coordinates]

        for i in range(self.n_dim):
            coordinates[i][coord_boundary_indexes[i]] = heatmap_size - 1

        dim_valid_loc = [c >= 0 for c in coordinates] + [c < heatmap_size for c in coordinates]
        valid_loc = reduce(lambda a, b: a & b, dim_valid_loc)
        # noinspection PyTypeChecker
        vis = self.keypoints[..., self.n_dim] > 0
        valid = (valid_loc & vis) > 0

        linear_index = sum(coordinates[i] * heatmap_size ** i for i in range(self.n_dim))
        heatmaps = linear_index * valid

        return heatmaps, valid


def _create_flip_indices(names: list[str], flip_map: dict[str, str]):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


def _kp_connections(keypoints):
    return [
        [keypoints.index("left_eye"), keypoints.index("right_eye")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
    ]


class PersonKeypoints(_PersonKeypoints):
    FLIP_MAP = {
        "left_eye": "right_eye",
        "left_ear": "right_ear",
        "left_shoulder": "right_shoulder",
        "left_elbow": "right_elbow",
        "left_wrist": "right_wrist",
        "left_hip": "right_hip",
        "left_knee": "right_knee",
        "left_ankle": "right_ankle"
    }

    FLIP_INDICES = _create_flip_indices(_PersonKeypoints.NAMES, FLIP_MAP)
