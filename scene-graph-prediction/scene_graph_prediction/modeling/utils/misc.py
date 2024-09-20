from enum import Enum
from typing import Sequence

import torch


def cat(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Efficient version of torch.cat that avoids a copy if there is only a single element in a list."""
    assert isinstance(tensors, Sequence)
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(list(tensors), dim)


class ROIHeadName(Enum):
    Attribute = "ROI_ATTRIBUTE_HEAD"
    BoundingBox = "ROI_BOX_HEAD"
    Keypoint = "ROI_KEYPOINT_HEAD"
    Mask = "ROI_MASK_HEAD"
    Relation = "ROI_RELATION_HEAD"

    def to_arch_head_name(self) -> str:
        return {
            self.Attribute: "attr",
            self.BoundingBox: "bbox",
            self.Keypoint: "kpts",
            self.Mask: "mask",
            self.Relation: "rel",
        }[self]
