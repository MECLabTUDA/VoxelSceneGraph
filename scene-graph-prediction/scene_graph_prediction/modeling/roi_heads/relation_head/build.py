from yacs.config import CfgNode

from .relation_head import ROIRelationHead
from ...abstractions.backbone import AnchorStrides
from ...abstractions.relation_head import ROIRelationHead as AbstractROIRelationHead


def build_roi_relation_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> AbstractROIRelationHead:
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config.
    """
    # Check that masks get predicted when using masks for relation prediction
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_MASKS:
        assert cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION or cfg.MODEL.MASK_ON, \
            ("Predicting relations using masks requires maks to be predicted "
             "(at least MODEL.REQUIRE_SEMANTIC_SEGMENTATION or MODEL.MASK_ON must be on).")

    return ROIRelationHead(cfg, in_channels, anchor_strides)
