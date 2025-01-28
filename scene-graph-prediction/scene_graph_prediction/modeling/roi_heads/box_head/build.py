from yacs.config import CfgNode

from .default import ROIBoxHead, ROIRelationReadyBoxHead
from .retinaunet_hybrid import ROIBoxHeadHybrid, ROIRelationReadyBoxHeadHybrid
from .retinaunet_hybrid_segmentation_grounded import ROIBoxHeadHybridSegGrounded, \
    ROIRelationReadyBoxHeadHybridSegGrounded
from ...abstractions.backbone import AnchorStrides
from ...abstractions.box_head import ROIBoxHead as AbstractROIBoxHead
from ...registries import ROI_BOX_HEADS


@ROI_BOX_HEADS.register("ROIBoxHead")
def _build_default_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    if cfg.MODEL.RELATION_ON:
        return ROIRelationReadyBoxHead(cfg, in_channels, anchor_strides)
    return ROIBoxHead(cfg, in_channels, anchor_strides)


@ROI_BOX_HEADS.register("ROIBoxHeadHybrid")
def _build_hybrid_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    if cfg.MODEL.RELATION_ON:
        return ROIRelationReadyBoxHeadHybrid(cfg, in_channels, anchor_strides)
    return ROIBoxHeadHybrid(cfg, in_channels, anchor_strides)


@ROI_BOX_HEADS.register("ROIBoxHeadHybridSegGrounded")
def _build_hybrid_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    if cfg.MODEL.RELATION_ON:
        return ROIRelationReadyBoxHeadHybridSegGrounded(cfg, in_channels, anchor_strides)
    return ROIBoxHeadHybridSegGrounded(cfg, in_channels, anchor_strides)


def build_roi_box_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIBoxHead:
    """Construct a new box head."""
    return ROI_BOX_HEADS[cfg.MODEL.BOX_HEAD](cfg, in_channels, anchor_strides)
