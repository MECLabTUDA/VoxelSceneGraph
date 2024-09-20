# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from typing import Callable, NamedTuple, Sequence, Type

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import FrozenBatchNorm, DFConv
from scene_graph_prediction.modeling.utils.build_layers import build_group_norm
from scene_graph_prediction.utils.registry import Registry
from ..abstractions.backbone import Backbone, FeatureMaps, Images, AnchorStrides


# ResNet stage specification
class StageSpec(NamedTuple):
    index: int  # Index of the stage, eg 1, 2, ..,. 5
    block_count: int  # Number of residual blocks in the stage
    return_features: bool  # True => return the last feature map from this stage


# -------------------------------------------------------------------------------------------------------------------- #
# Standard ResNet models
# -------------------------------------------------------------------------------------------------------------------- #

# ResNet-50 (including all stages)
_ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
_ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
_ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
_ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
_ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
    # for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True))
)
# ResNet-101-FPN (including all stages)
_ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
_ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)


class ResNet(Backbone):
    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]

        self.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

        # If we want to use the cfg in forward(), then we should make a copy of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Construct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            module = _build_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                self.n_dim,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at: int):
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, f"layer{stage_index}")
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x: Images) -> FeatureMaps:
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs

    @property
    def feature_strides(self) -> AnchorStrides:
        if self.n_dim == 2:
            # Add more than there can be levels as ResNets are flexible
            strides = (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)
            return tuple(stride for stride, stage_name in zip(strides, self.stages) if self.return_features[stage_name])
        strides = (1, 4, 4), (2, 8, 8), (4, 16, 16), (8, 32, 32), (16, 64, 64)
        return tuple(stride for stride, stage_name in zip(strides, self.stages) if self.return_features[stage_name])

class ResNetHead(torch.nn.Module):
    def __init__(self,
                 n_dim: int,
                 block_module_name: str,
                 stages: Sequence[StageSpec],
                 num_groups: int = 1,
                 width_per_group: int = 64,
                 stride_in_1x1: bool = True,
                 stride_init: int | None = None,
                 res2_out_channels: int = 256,
                 dilation: int = 1,
                 dcn_config: dict | None = None):
        super().__init__()
        assert n_dim in [2, 3]

        if dcn_config is None:
            dcn_config = {}

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module_name]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _build_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                n_dim,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


class _Bottleneck(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 bottleneck_channels: int,
                 out_channels: int,
                 n_dim: int,
                 num_groups: int,
                 stride_in_1x1: bool,
                 stride: int,
                 dilation: int,
                 norm_func: Callable,
                 dcn_config: dict | None
                 ):
        super().__init__()

        assert n_dim in [2, 3]
        self.n_dim = n_dim
        conv_module = torch.nn.Conv2d if n_dim == 2 else torch.nn.Conv3d

        if dcn_config is None:
            dcn_config = {}

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1

            self.downsample = torch.nn.Sequential(
                conv_module(in_channels, out_channels, kernel_size=1, stride=down_stride, bias=False),
                norm_func(out_channels),
            )

            for m in self.downsample.modules():
                if isinstance(m, conv_module):
                    torch.nn.init.kaiming_uniform_(m.weight, a=1)

        if dilation > 1:
            stride = 1  # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = conv_module(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)

        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv(
                bottleneck_channels,
                bottleneck_channels,
                n_dim=n_dim,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = conv_module(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            torch.nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = conv_module(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_func(out_channels)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.nn.functional.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return torch.nn.functional.relu_(out)


class _BottleneckWithFixedBatchNorm(_Bottleneck):
    def __init__(
            self,
            in_channels: int,
            bottleneck_channels: int,
            out_channels: int,
            n_dim: int,
            num_groups: int = 1,
            stride_in_1x1: bool = True,
            stride: int = 1,
            dilation: int = 1,
            dcn_config: dict | None = None
    ):
        super().__init__(in_channels=in_channels,
                         bottleneck_channels=bottleneck_channels,
                         out_channels=out_channels,
                         n_dim=n_dim,
                         num_groups=num_groups,
                         stride_in_1x1=stride_in_1x1,
                         stride=stride,
                         dilation=dilation,
                         norm_func=FrozenBatchNorm,
                         dcn_config=dcn_config
                         )


class _BottleneckWithGN(_Bottleneck):
    def __init__(self,
                 in_channels: int,
                 bottleneck_channels: int,
                 out_channels: int,
                 n_dim: int,
                 num_groups: int = 1,
                 stride_in_1x1: bool = True,
                 stride: int = 1,
                 dilation: int = 1,
                 dcn_config: dict | None = None):
        super().__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            n_dim=n_dim,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=build_group_norm,
            dcn_config=dcn_config
        )


class _BaseStem(torch.nn.Module):
    def __init__(self, cfg: CfgNode, norm_func: Callable):
        super().__init__()

        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        self.n_channels = cfg.INPUT.N_CHANNELS

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        if self.n_dim == 2:
            self.conv1 = torch.nn.Conv2d(self.n_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = torch.nn.Conv3d(self.n_channels, out_channels,
                                         kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = norm_func(out_channels)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu_(x)
        if self.n_dim == 2:
            return torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return torch.nn.functional.max_pool3d(x, kernel_size=3, stride=(1, 2, 2), padding=1)


class _StemWithFixedBatchNorm(_BaseStem):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg, norm_func=FrozenBatchNorm)


class _StemWithGN(_BaseStem):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg, norm_func=build_group_norm)


def _build_stage(transformation_module: Type[_BottleneckWithGN] | Type[_BottleneckWithFixedBatchNorm],
                 in_channels: int,
                 bottleneck_channels: int,
                 out_channels: int,
                 n_dim: int,
                 block_count: int,
                 num_groups: int,
                 stride_in_1x1: bool,
                 first_stride: int,
                 dilation: int = 1,
                 dcn_config: dict | None = None) -> torch.nn.Sequential:
    """
    :param transformation_module:
    :param in_channels:
    :param bottleneck_channels:
    :param out_channels:
    :param n_dim:
    :param block_count:
    :param num_groups:
    :param stride_in_1x1:
    :param first_stride:
    :param dilation:
    :param dcn_config: dict containing optionally the following keys:
            stage_with_dcn: bool
            deformable_groups: int
            with_modulated_dcn: bool
    """
    if dcn_config is None:
        dcn_config = {}

    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                n_dim,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels

    return torch.nn.Sequential(*blocks)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": _BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": _BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": _StemWithFixedBatchNorm,
    "StemWithGN": _StemWithGN,
})

# FIXME needs to be cleaned
_STAGE_SPECS = Registry({
    "R-50-C4": _ResNet50StagesTo4,
    "R-50-C5": _ResNet50StagesTo5,
    "R-101-C4": _ResNet101StagesTo4,
    "R-101-C5": _ResNet101StagesTo5,
    "R-50-FPN": _ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": _ResNet50FPNStagesTo5,
    "R-101-FPN": _ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": _ResNet101FPNStagesTo5,
    "R-152-FPN": _ResNet152FPNStagesTo5,
})
