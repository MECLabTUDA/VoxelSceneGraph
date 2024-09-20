from yacs.config import CfgNode

from .attribute_head import ROIAttributeHead
from ...abstractions.attribute_head import ROIAttributeHead as AbstractROIAttributeHead
from ...abstractions.backbone import AnchorStrides


def build_roi_attribute_head(
        cfg: CfgNode,
        in_channels: int,
        anchor_strides: AnchorStrides
) -> AbstractROIAttributeHead:
    """
    Construct a new attribute head.
    By default, uses ROIAttributeHead. But if it turns out not to be enough,
    just register a new class and make it a parameter in the config.
    """
    return ROIAttributeHead(cfg, in_channels, anchor_strides)
