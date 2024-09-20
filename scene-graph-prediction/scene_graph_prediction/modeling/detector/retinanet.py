from yacs.config import CfgNode

from .base_detector import BaseDetector
from ..backbone.fpn import FPN
from ..region_proposal import RetinaNetModule
from ..roi_heads import build_roi_heads


class RetinaNet(BaseDetector):
    """
    RetinaNet detector. Uses FPN as the backbone.
    Note: can also be used as an RPN for a two-stage method (see cfg.MODEL.RETINANET.TWO_STAGE).
    """

    def __init__(self, cfg: CfgNode):
        assert not cfg.MODEL.MASK_ON, "Masks need to be off. Consider using RetinaUNet instead."

        if cfg.MODEL.RELATION_ON:
            # One stage methods cannot be evaluated easily with sgcls:
            #  It would require a matching of gt boxes with anchors to get logits
            from scene_graph_prediction.data.evaluation import SGGEvaluationMode
            assert SGGEvaluationMode.build(cfg) != SGGEvaluationMode.SceneGraphClassification, "Mode not supported."

        backbone = FPN(cfg.INPUT.N_DIM, cfg.INPUT.N_CHANNELS)
        super().__init__(
            cfg,
            backbone,
            RetinaNetModule(cfg, backbone.out_channels, backbone.feature_strides),
            build_roi_heads(
                cfg,
                in_channels=backbone.out_channels,
                anchor_strides=backbone.feature_strides,
                is_rpn_only=cfg.MODEL.RPN_ONLY,
                has_boxes=cfg.MODEL.RETINANET.TWO_STAGE or cfg.MODEL.ROI_HEADS_ONLY,
                has_masks=cfg.MODEL.MASK_ON,
                has_keypoints=cfg.MODEL.KEYPOINT_ON,
                has_attributes=cfg.MODEL.ATTRIBUTE_ON,
                has_relations=cfg.MODEL.RELATION_ON,
            )
        )
