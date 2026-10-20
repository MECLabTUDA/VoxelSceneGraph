from yacs.config import CfgNode

from .keypoint_head import ROIKeypointHead
from ...abstractions.backbone import AnchorStrides
from ...abstractions.keypoint_head import ROIKeypointHead as AbstractROIKeypointHead


def build_roi_keypoint_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIKeypointHead:
    return ROIKeypointHead(cfg, in_channels, anchor_strides)
