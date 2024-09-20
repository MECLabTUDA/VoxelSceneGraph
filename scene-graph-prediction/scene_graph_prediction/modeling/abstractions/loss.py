from typing import TypedDict, Any

import torch
from typing_extensions import NotRequired

# Since it's the easiest way to save information during training,
# we also allow it to contain additional information that is not a loss term
# These should have keys starting with an underscore
LossDict = dict[str, torch.Tensor] | dict[str, Any]


class AttributeHeadLossDict(TypedDict):
    loss_attribute: torch.Tensor


class BoxHeadLossDict(TypedDict):
    loss_classifier: torch.Tensor
    loss_box_reg: torch.Tensor


class KeypointHeadLossDict(TypedDict):
    loss_kp: torch.Tensor


class MaskHeadLossDict(TypedDict):
    loss_mask: torch.Tensor


class RelationHeadLossDict(TypedDict):
    loss_rel: torch.Tensor
    loss_refine_obj: torch.Tensor
    loss_refine_att: torch.Tensor


class RPNLossDict(TypedDict):
    loss_objectness: torch.Tensor
    loss_rpn_box_reg: torch.Tensor
    loss_rpn_seg: NotRequired[torch.Tensor]
