from yacs.config import CfgNode

from .base_detector import BaseDetector
from ..backbone.no_op import NoOpBackbone
from ..region_proposal.from_precomputations import ProposalsFromPrecomputations
from ..roi_heads import build_roi_heads


class FromPrecomputationsDetector(BaseDetector):
    """NoOpBackbone with an ProposalsFromPrecomputations RPN (or one-stage detector)."""

    def __init__(self, cfg: CfgNode):
        if cfg.MODEL.RELATION_ON:
            # One stage methods cannot be evaluated easily with sgcls:
            #  It would require a matching of gt boxes with anchors to get logits
            from scene_graph_prediction.data.evaluation import SGGEvaluationMode
            assert SGGEvaluationMode.build(cfg) != SGGEvaluationMode.SceneGraphClassification, "Mode not supported."

        assert not cfg.MODEL.RPN_ONLY, "cfg.MODEL.RPN_ONLY does not make any sense! The RPN is not trainable..."

        backbone = NoOpBackbone(cfg)
        rpn = ProposalsFromPrecomputations(cfg, backbone.out_channels, backbone.feature_strides)
        super().__init__(
            cfg,
            backbone,
            rpn,
            build_roi_heads(
                cfg,
                in_channels=backbone.out_channels,
                anchor_strides=backbone.feature_strides,
                detector_is_one_stage=rpn.is_one_stage_detector(),
                is_rpn_only=False,
                has_boxes=cfg.MODEL.BOX_ON,
                has_masks=cfg.MODEL.MASK_ON,
                has_keypoints=cfg.MODEL.KEYPOINT_ON,
                has_attributes=cfg.MODEL.ATTRIBUTE_ON,
                has_relations=cfg.MODEL.RELATION_ON,
            )
        )
