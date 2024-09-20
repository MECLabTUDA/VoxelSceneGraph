from abc import ABC, abstractmethod
from typing import Sequence

import torch

# Backbone (Encoder)
Images = torch.Tensor  # ImageList.tensors attribute i.e. multiple images in a single tensor
FeatureMaps = Sequence[torch.Tensor]  # List of feature maps at each scale
# Downscale factor for each feature map level for each dimension
AnchorStrides = tuple[tuple[int, ...], ...]


class Backbone(torch.nn.Module, ABC):
    """Interface for backbones. Features are returned as a list of tensors."""

    # Note for later if serialization fails:
    # Backbones used to be wrapped in a Sequential module with key "body"

    out_channels: int
    n_dim: int

    @abstractmethod
    def forward(self, x: Images) -> FeatureMaps:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_strides(self) -> AnchorStrides:
        """
        :returns: for each feature map level return by self.forward(),
                  a tuple with the spatial stride for each dimension in the image.
        """
        raise NotImplementedError
