from yacs.config import CfgNode

from .mask_head import ROIMaskHead
from ...abstractions.backbone import AnchorStrides
from ...abstractions.mask_head import ROIMaskHead as AbstractROIMaskHead


def build_roi_mask_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIMaskHead:
    return ROIMaskHead(cfg, in_channels, anchor_strides)
