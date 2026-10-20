"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Union, Hashable, Any

import torch

_SIZE_T = tuple[int, ...]


class Keypoints:
    def __init__(self, keypoints: list | torch.Tensor, size: _SIZE_T):
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]

        if num_keypoints:
            # 2D shape is (y, x, logit)
            # 3D shape is (z, y, x, logit)
            keypoints = keypoints.view(num_keypoints, -1, len(size) + 1)

        self.keypoints = keypoints
        self.n_dim = len(size)

        self.size = size
        self.extra_fields = {}

    def __getitem__(self, item: Union[int, list[int], list[bool], torch.Tensor]) -> "Keypoints":
        """Like a tensor, it accepts an index, a list/tensor of ints or a list/tensor of bools."""
        keypoints = type(self)(self.keypoints[item], self.size)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field: Hashable, field_data: Any):
        self.extra_fields[field] = field_data

    def get_field(self, field: Hashable) -> Any:
        return self.extra_fields[field]

    def has_field(self, field: Hashable) -> bool:
        return field in self.extra_fields

    def __repr__(self) -> str:
        fields_repr = ",".join(sorted(list(self.extra_fields.keys())))
        return f"{self.__class__.__name__}(num_instances={len(self.keypoints)}, image_size={self.size}, " \
               f"fields=[{fields_repr}])"


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

    CONNECTIONS = [
        [NAMES.index('left_eye'), NAMES.index('right_eye')],
        [NAMES.index('left_eye'), NAMES.index('nose')],
        [NAMES.index('right_eye'), NAMES.index('nose')],
        [NAMES.index('right_eye'), NAMES.index('right_ear')],
        [NAMES.index('left_eye'), NAMES.index('left_ear')],
        [NAMES.index('right_shoulder'), NAMES.index('right_elbow')],
        [NAMES.index('right_elbow'), NAMES.index('right_wrist')],
        [NAMES.index('left_shoulder'), NAMES.index('left_elbow')],
        [NAMES.index('left_elbow'), NAMES.index('left_wrist')],
        [NAMES.index('right_hip'), NAMES.index('right_knee')],
        [NAMES.index('right_knee'), NAMES.index('right_ankle')],
        [NAMES.index('left_hip'), NAMES.index('left_knee')],
        [NAMES.index('left_knee'), NAMES.index('left_ankle')],
        [NAMES.index('right_shoulder'), NAMES.index('left_shoulder')],
        [NAMES.index('right_hip'), NAMES.index('left_hip')],
    ]
