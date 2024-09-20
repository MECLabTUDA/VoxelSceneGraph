from yacs.config import CfgNode

from .default import ROIBoxHead, ROIRelationReadyBoxHead
from .retinaunet_hybrid import ROIBoxHeadHybrid, ROIRelationReadyBoxHeadHybrid
from ...abstractions.backbone import AnchorStrides
from ...abstractions.box_head import ROIBoxHead as AbstractROIBoxHead
from ...registries import ROI_HEADS


@ROI_HEADS.register("ROIBoxHead")
def _build_default_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    if cfg.MODEL.RELATION_ON:
        return ROIRelationReadyBoxHead(cfg, in_channels, anchor_strides)
    return ROIBoxHead(cfg, in_channels, anchor_strides)


@ROI_HEADS.register("ROIBoxHeadHybrid")
def _build_hybrid_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    if cfg.MODEL.RELATION_ON:
        return ROIRelationReadyBoxHeadHybrid(cfg, in_channels, anchor_strides)
    return ROIBoxHeadHybrid(cfg, in_channels, anchor_strides)


def build_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIBoxHead:
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config.
    """
    return ROI_HEADS[cfg.MODEL.BOX_HEAD](cfg, in_channels, anchor_strides)
