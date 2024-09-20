import logging

import torch
from yacs.config import CfgNode

from ...abstractions.backbone import FeatureMaps
from ...abstractions.region_proposal import RPNHead, FeatureMapsObjectness, FeatureMapsBoundingBoxRegression
from ...backbone.fbnet.fbnet_builder import FBNetBuilder, get_blocks
from ...backbone.fbnet.fbnet_model_def import UnifiedArch, ExpandedBlockCfgWithStageOp
from ...registries import RPN_HEADS


class RPNHeadConvRegressor(RPNHead):
    """
    A single conv HEAD for region proposal.
    Note: supports 2D only.
    Note: supports RPN and FPN.
    """

    def __init__(self, _: CfgNode, in_channels: int, num_anchors: int):
        """
        :param _: Config (not used).
        :param in_channels: Number of input channels.
        :param num_anchors: Number of anchors to be predicted.
        """
        super().__init__()
        self.cls_logits = torch.nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x: FeatureMaps) -> tuple[FeatureMapsObjectness, FeatureMapsBoundingBoxRegression]:
        logits = list(map(self.cls_logits, x))
        bbox_reg = list(map(self.bbox_pred, x))

        return logits, bbox_reg


def _get_rpn_stage(arch_def: UnifiedArch, num_blocks: int) -> list[ExpandedBlockCfgWithStageOp]:
    rpn_stage = arch_def.get("rpn")
    ret = get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logging.getLogger(__name__).warning(f'Use last {num_blocks} blocks in {ret} as rpn')

        block_count = len(ret["stages"])
        assert num_blocks <= block_count, f"use block {num_blocks}, block count {block_count}"
        blocks = range(block_count - num_blocks, block_count)
        ret = get_blocks(ret, block_indices=blocks)

    return ret["stages"]


class FBNetRPNHead(torch.nn.Module):
    """
    Region proposal HEAD using FBNet.
    Note: supports RPN and FPN.
    """

    def __init__(self, cfg: CfgNode, in_channels: int, builder: FBNetBuilder, arch_def: UnifiedArch):
        # in_channels is only used for assert
        super().__init__()
        assert in_channels == builder.last_depth, in_channels

        rpn_bn_type: str = cfg.MODEL.FBNET.RPN_BN_TYPE
        if rpn_bn_type:
            builder.bn_type = rpn_bn_type

        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)

        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x: FeatureMaps) -> FeatureMaps:
        return [self.head(y) for y in x]


@RPN_HEADS.register("SingleConvRPNHead")
class SingleConvRPNHead(RPNHead):
    """Adds a simple RPN Head with classification and regression heads."""

    def __init__(self, cfg: CfgNode, in_channels: int, __: int, num_anchors: int):
        super().__init__()

        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        n_features = 256 if self.n_dim == 2 else 128
        conv_module = torch.nn.Conv2d if self.n_dim == 2 else torch.nn.Conv3d
        self.conv = conv_module(in_channels, n_features, kernel_size=3, stride=1, padding=1)
        self.cls_logits = conv_module(n_features, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = conv_module(n_features, num_anchors * 2 * self.n_dim, kernel_size=1, stride=1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features: FeatureMaps) -> tuple[FeatureMapsObjectness, FeatureMapsBoundingBoxRegression]:
        logits = []
        bbox_reg = []
        for feature in features:
            t = torch.nn.functional.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
