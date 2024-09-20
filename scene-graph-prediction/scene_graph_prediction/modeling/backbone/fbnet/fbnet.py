import copy
from collections import OrderedDict

import torch

from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.utils.logger import setup_logger
from .fbnet_builder import unify_arch_def, FBNetBuilder, get_blocks, get_num_stages
from .fbnet_model_def import MODEL_ARCH, ModelArch, UnifiedArch, ExpandedBlockCfgWithStageOp
from ...abstractions.backbone import Backbone, FeatureMaps, Images, AnchorStrides
from ...utils import pooler, ROIHeadName


# FIXME clean and improve FBNet thingy

def build_fbnet_builder(cfg: CfgNode) -> tuple[FBNetBuilder, UnifiedArch]:
    bn_type = cfg.MODEL.FBNET.BN_TYPE
    if bn_type == "gn":
        bn_type = bn_type, cfg.GROUP_NORM.NUM_GROUPS
    factor = cfg.MODEL.FBNET.SCALE_FACTOR

    arch_name: str = cfg.MODEL.FBNET.ARCH
    assert arch_name in MODEL_ARCH, arch_name

    arch_def: ModelArch = MODEL_ARCH[arch_name]
    arch_def: UnifiedArch = unify_arch_def(arch_def)

    width_divisor = cfg.MODEL.FBNET.WIDTH_DIVISOR
    dw_skip_bn = cfg.MODEL.FBNET.DW_CONV_SKIP_BN
    dw_skip_relu = cfg.MODEL.FBNET.DW_CONV_SKIP_RELU

    setup_logger("build_fbnet_builder", "", distributed_rank=1) \
        .info(f"Building fbnet model with arch {arch_name} (without scaling):\n{arch_def}")

    builder = FBNetBuilder(
        width_ratio=factor,
        bn_type=bn_type,
        width_divisor=width_divisor,
        dw_skip_bn=dw_skip_bn,
        dw_skip_relu=dw_skip_relu,
    )

    return builder, arch_def


def _get_trunk_cfg(arch_def: UnifiedArch) -> UnifiedArch:
    """Get all stages except the last one."""
    num_stages = get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    return get_blocks(arch_def, stage_indices=trunk_stages)


class FBNetTrunk(Backbone):
    def __init__(self, builder: FBNetBuilder, arch_def: UnifiedArch, dim_in: int, n_dim: int):
        super().__init__()
        assert n_dim == 2
        self.n_dim = n_dim
        self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])
        self.out_channels = builder.last_depth

    def forward(self, x: Images) -> FeatureMaps:
        return [self.stages(self.first(x))]

    @property
    def feature_strides(self) -> AnchorStrides:
        # TODO checkout feature maps sizes...
        raise NotImplementedError("TODO...")


def _get_head_stage(arch: UnifiedArch, head_name: ROIHeadName, blocks: list[int]) -> list[ExpandedBlockCfgWithStageOp]:
    assert head_name.to_arch_head_name() in arch
    # Linter confusion between Literal and str
    # noinspection PyTypedDict
    head_stage = arch.get(head_name.to_arch_head_name(), None)
    ret = get_blocks(arch, stage_indices=head_stage, block_indices=blocks)
    return ret["stages"]


class FBNetROIHead(ROIKeypointFeatureExtractor):
    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            builder: FBNetBuilder,
            arch_def: UnifiedArch,
            head_name: ROIHeadName,
            use_blocks: list[int],
            stride_init: int,
            last_layer_scale: int,
            anchor_strides: AnchorStrides,
    ):
        # in_channels is only used for assert
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)
        n_dim = cfg.INPUT.N_DIM
        assert n_dim == 2
        self.n_dim = n_dim

        super().__init__(cfg, in_channels)

        self.pooler = pooler.build_pooler(cfg, head_name, anchor_strides)

        stage = _get_head_stage(arch_def, head_name, use_blocks)

        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            # Linter having a brainfuck on (int, int, int, int)
            # noinspection PyUnresolvedReferences
            stage[0]["block"][3] = stride_init
        blocks = builder.add_blocks(stage)

        last_info = copy.deepcopy(arch_def["last"])
        tmp = list(last_info)
        tmp[1] = last_layer_scale
        last_info = tuple(tmp)
        # noinspection PyTypeChecker
        last = builder.add_last(last_info)

        self.head = torch.nn.Sequential(OrderedDict([("blocks", blocks), ("last", last)]))
        self.representation_size = builder.last_depth
        self.out_channels = builder.last_depth

    def forward(self, x: torch.Tensor, proposals) -> torch.Tensor:
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@ROI_BOX_FEATURE_EXTRACTORS.register("FBNet.roi_head")
def add_roi_head(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides, _: bool = False, __: bool = False):
    builder, model_arch = build_fbnet_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[bbox]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name=ROIHeadName.BoundingBox,
        use_blocks=cfg.MODEL.FBNET.DET_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.DET_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.DET_HEAD_LAST_SCALE,
        anchor_strides=anchor_strides
    )


@ROI_KEYPOINT_FEATURE_EXTRACTORS.register("FBNet.roi_head_keypoints")
def add_roi_head_keypoints(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    builder, model_arch = build_fbnet_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[keypoints]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name=ROIHeadName.Keypoint,
        use_blocks=cfg.MODEL.FBNET.KPTS_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.KPTS_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.KPTS_HEAD_LAST_SCALE,
        anchor_strides=anchor_strides
    )


@ROI_MASK_FEATURE_EXTRACTORS.register("FBNet.roi_head_mask")
def add_roi_head_mask(cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
    builder, model_arch = build_fbnet_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[mask]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name=ROIHeadName.Mask,
        use_blocks=cfg.MODEL.FBNET.MASK_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.MASK_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.MASK_HEAD_LAST_SCALE,
        anchor_strides=anchor_strides
    )
