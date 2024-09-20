from yacs.config import CfgNode

from .base_detector import BaseDetector
from .retinanet import RetinaNet
from ..backbone.fpn import FPN
from ..region_proposal import RetinaUNetModule
from ..roi_heads import build_roi_heads


class RetinaUNet(RetinaNet):
    """
    RetinaUNet detector. Uses FPN as the backbone.
    The segmentation is done through a last convolution layer.
    Note: can also be used as an RPN for a two-stage method (see cfg.MODEL.RETINANET.TWO_STAGE).
    """

    def __init__(self, cfg: CfgNode):
        # MASK_ON is not required anymore, e.g. we do detection through segmentation but no evaluation on the seg
        # assert cfg.MODEL.MASK_ON, "Masks need to be on for Retina-UNet."
        assert cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION, "Masks need to be provided as semantic segmentation."

        # Note: it's important to skip RetinaNet's constructor to avoid building the backbone/RPN/ROIHeads twice
        backbone = FPN(cfg.INPUT.N_DIM, cfg.INPUT.N_CHANNELS)
        BaseDetector.__init__(
            self,
            cfg,
            backbone,
            RetinaUNetModule(cfg, backbone.out_channels, backbone.feature_strides),
            build_roi_heads(
                cfg,
                in_channels=backbone.out_channels,
                anchor_strides=backbone.feature_strides,
                is_rpn_only=cfg.MODEL.RPN_ONLY,
                has_boxes=cfg.MODEL.RETINANET.TWO_STAGE or cfg.MODEL.ROI_HEADS_ONLY,
                has_masks=False,  # The segmentation is already handled by the RetinaUNetModule
                has_keypoints=cfg.MODEL.KEYPOINT_ON,
                has_attributes=cfg.MODEL.ATTRIBUTE_ON,
                has_relations=cfg.MODEL.RELATION_ON,
            )
        )
