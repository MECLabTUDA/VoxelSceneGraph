from yacs.config import CfgNode

from .default import ROIMaskHead
from .retinaunet_hybrid import ROIMaskHeadHybrid
from ...abstractions.backbone import AnchorStrides
from ...abstractions.mask_head import ROIMaskHead as AbstractROIMaskHead
from ...registries import ROI_MASK_HEADS


@ROI_MASK_HEADS.register("ROIMaskHead")
def _build_default_roi_mask_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    return ROIMaskHead(cfg, in_channels, anchor_strides)


@ROI_MASK_HEADS.register("ROIMaskHeadHybrid")
def _build_default_roi_mask_head_hybrid(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    return ROIMaskHeadHybrid(cfg, in_channels, anchor_strides)


def build_roi_mask_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIMaskHead:
    """Construct a new mask head."""
    return ROI_MASK_HEADS[cfg.MODEL.MASK_HEAD](cfg, in_channels, anchor_strides)
