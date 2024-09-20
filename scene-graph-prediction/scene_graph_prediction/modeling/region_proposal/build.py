import torch
from yacs.config import CfgNode

from .retinanet import RetinaNetModule
from .retinaunet import RetinaUNetModule
from .retinaunet_hybrid import RetinaUNetHybridModule
from .two_stage import RPNModule, RPNHeadConvRegressor, FBNetRPNHead
from ..abstractions.backbone import AnchorStrides
from ..abstractions.region_proposal import RPN, RPNHead
from ..backbone.fbnet import build_fbnet_builder
from ..registries import RPN_HEADS, RPNS


@RPN_HEADS.register("FBNet.rpn_head")
def _build_fbnet_rpn_head(cfg: CfgNode, in_channels: int, _: int, num_anchors: int) -> torch.nn.Sequential:
    assert cfg.INPUT.N_DIM == 2
    builder, model_arch = build_fbnet_builder(cfg)
    builder.last_depth = in_channels

    rpn_feature = FBNetRPNHead(cfg, in_channels, builder, model_arch)
    rpn_regressor = RPNHeadConvRegressor(cfg, rpn_feature.out_channels, num_anchors)

    class FBNetRpnHeadWrapper(torch.nn.Sequential, RPNHead):
        """Quick wrapper to return a module that explicitly is a RPNHead."""

        def __init__(self):
            torch.nn.Sequential.__init__(self, rpn_feature, rpn_regressor)
            self.n_dim = 2

    return FBNetRpnHeadWrapper()


# Register RPN
RPNS.register("RPN", RPNModule)
# Note: RetinaNet modules are registered here, but expect specific backbones
RPNS.register("RetinaNet", RetinaNetModule)
RPNS.register("RetinaUNet", RetinaUNetModule)
RPNS.register("RetinaUNetHybrid", RetinaUNetHybridModule)


def build_rpn(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides) -> RPN:
    assert cfg.MODEL.REGION_PROPOSAL in RPNS, \
        f"cfg.MODEL.REGION_PROPOSAL: {cfg.MODEL.REGION_PROPOSAL} are not registered in registry"
    assert not (cfg.MODEL.RPN_ONLY and cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE), \
        "RPN only mode and the optimized training pipeline are incompatible."

    return RPNS[cfg.MODEL.REGION_PROPOSAL](cfg, in_channels, anchor_strides)
