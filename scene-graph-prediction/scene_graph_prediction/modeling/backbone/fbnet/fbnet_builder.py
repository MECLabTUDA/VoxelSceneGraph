import copy
from collections import OrderedDict
from typing import Literal, Sequence, Protocol, Iterable

import torch

from scene_graph_prediction.layers import FrozenBatchNorm
from .fbnet_model_def import ModelArch, ArchStages, StageCfg, BlockCfg, ExpandedBlockCfg, UnifiedArch, \
    ExpandedBlockCfgWithStageOp


def _get_divisible_by(num: float, divisible_by: int) -> int:
    """
    Returns the closest multiple of divisible_by smaller than num.
    If it is 0, then return divisible_by.
    """
    if divisible_by < 0:
        return int(num)

    num = int(num)
    if divisible_by == 0:
        return num

    ret = num - (num % divisible_by)

    if ret == 0:
        return divisible_by
    return ret


class _Primitive(Protocol):
    output_depth: int

    def __call__(self, c_in: int, c_out: int, expansion: int, stride: int, **kwargs) -> torch.nn.Module: ...


_PRIMITIVES: dict[str, _Primitive] = {
    "skip": lambda c_in, c_out, expansion, stride, **kwargs: _IdentityOrConv(c_in, c_out, stride),
    "ir_k3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, **kwargs
    ),
    "ir_k5": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, kernel=5, **kwargs
    ),
    "ir_k7": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, kernel=7, **kwargs
    ),
    "ir_k1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, kernel=1, **kwargs
    ),
    "shuffle": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "basic_block": lambda c_in, c_out, expansion, stride, **kwargs: _CascadeConv3x3(c_in, c_out, stride),
    "shift_5x5": lambda c_in, c_out, expansion, stride, **kwargs: _ShiftBlock5x5(c_in, c_out, expansion, stride),
    # layer search 2
    "ir_k3_e1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=3, **kwargs
    ),
    "ir_k3_e3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=3, **kwargs
    ),
    "ir_k3_e6": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=3, **kwargs
    ),
    "ir_k3_s4": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "ir_k5_e1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=5, **kwargs
    ),
    "ir_k5_e3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=5, **kwargs
    ),
    "ir_k5_e6": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=5, **kwargs
    ),
    "ir_k5_s4": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
    ),
    # layer search se
    "ir_k3_e1_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e3_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e6_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_s4_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 4, stride, kernel=3, se=True, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "ir_k5_e1_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e3_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e6_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_s4_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 4, stride, kernel=5, se=True, shuffle_type="mid", pw_group=4, **kwargs
    ),
    # layer search 3 (in addition to layer search 2)
    "ir_k3_s2": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k5_s2": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k3_s2_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=3, se=True, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k5_s2_se": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, se=True, **kwargs
    ),
    # layer search 4 (in addition to layer search 3)
    "ir_k3_sep": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e6": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=3, cdw=True, **kwargs
    ),
    # layer search 5 (in addition to layer search 4)
    "ir_k7_e1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=7, **kwargs
    ),
    "ir_k7_e3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=7, **kwargs
    ),
    "ir_k7_e6": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=7, **kwargs
    ),
    "ir_k7_sep": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, expansion, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e1": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 1, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e3": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 3, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e6": lambda c_in, c_out, expansion, stride, **kwargs: _IRFBlock(
        c_in, c_out, 6, stride, kernel=7, cdw=True, **kwargs
    ),
}


class _IdentityOrConv(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv = None
        if c_in != c_out or stride != 1:
            self.conv = ConvBNRelu(
                c_in,
                c_out,
                kernel_size=1,
                stride=stride,
                pad=0,
                no_bias=True,
                use_relu=True,
                bn_type="bn",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            return self.conv(x)
        return x


class _CascadeConv3x3(torch.nn.Sequential):
    def __init__(self, c_in: int, c_out: int, stride: Literal[1, 2]):
        """If stride == 1 and c_in == c_out, then add input to conv output."""
        assert stride in [1, 2]
        super().__init__(
            torch.nn.Conv2d(c_in, c_in, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(c_in),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(c_out)
        )
        self.res_connect = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.res_connect:
            y += x
        return y


class _Shift(torch.nn.Module):
    def __init__(self, n_classes: int, kernel_size: int, stride: Literal[1, 2], padding: int):
        assert stride in [1, 2]

        super().__init__()
        self.n_classes = n_classes
        kernel = torch.zeros((n_classes, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = n_classes // ksq + n_classes % ksq
                else:
                    num_ch = n_classes // ksq
                kernel[ch_idx: ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = torch.nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.kernel,
            self.bias,
            (self.stride, self.stride),
            (self.padding, self.padding),
            self.dilation,
            self.n_classes,  # groups
        )


class _ShiftBlock5x5(torch.nn.Sequential):
    def __init__(self, c_in: int, c_out: int, expansion: int, stride: Literal[1, 2]):
        """If stride == 1 and c_in == c_out, then add input to conv output."""
        assert stride in [1, 2]
        c_mid = _get_divisible_by(c_in * expansion, 8)

        super().__init__(
            # pw
            torch.nn.Conv2d(c_in, c_mid, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(c_mid),
            torch.nn.ReLU(inplace=True),
            # shift
            _Shift(c_mid, 5, stride, 2),
            # pw-linear
            torch.nn.Conv2d(c_mid, c_out, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(c_out)
        )

        self.res_connect = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)

        if self.res_connect:
            y += x
        return y


class _ChannelShuffle(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        n, c, h, w = x.size()  # type: int, int, int, int
        assert c % self.n_classes == 0, f"Incompatible group size {self.n_classes} for input channel {c}"

        return x.view(n, self.n_classes, int(c / self.n_classes), h, w) \
            .permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)


_BN_T = Literal["bn", "af", "gn", None]


class ConvBNRelu(torch.nn.Sequential):
    def __init__(
            self,
            input_depth: int,
            output_depth: int,
            kernel_size: int,
            stride: int,
            pad: int,
            no_bias: bool,
            use_relu: bool,
            bn_type: _BN_T,
            group: int = 1,
            *args,
            **kwargs
    ):
        super().__init__()

        if isinstance(bn_type, (tuple, list)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        else:
            assert bn_type != "gn"
            gn_group = None

        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = torch.nn.Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )

        torch.nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            torch.nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            self.add_module("bn", torch.nn.BatchNorm2d(output_depth))
        elif bn_type == "gn":
            self.add_module("bn", torch.nn.GroupNorm(num_groups=gn_group, num_channels=output_depth))
        elif bn_type == "af":
            self.add_module("bn", FrozenBatchNorm(output_depth))

        if use_relu:
            self.add_module("relu", torch.nn.ReLU(inplace=True))


class _SEModule(torch.nn.Module):
    def __init__(self, c: int):
        super().__init__()
        reduction = 4
        mid = max(c // reduction, 8)

        self.op = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c, mid, 1, 1, 0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid, c, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.op(x)


class _Upsample(torch.nn.Module):
    def __init__(self,
                 scale_factor: float | Sequence[float],
                 mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]):
        super().__init__()
        self.scale = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode,
                                               align_corners=None)


_STRIDE_T = Literal[1, 2, 4, -1, -2, -4]


def _get_upsample_op(stride: _STRIDE_T | tuple[_STRIDE_T]) -> tuple[_Upsample | None, int]:
    _possible_strides = [1, 2, 4, -1, -2, -4]
    assert stride in _possible_strides \
           or (isinstance(stride, Sequence) and all(x in [-1, -2, -4] for x in stride))

    if isinstance(stride, tuple):
        return _Upsample(scale_factor=[-x for x in stride], mode="nearest"), 1
    if stride < 0:
        return _Upsample(scale_factor=-stride, mode="nearest"), 1
    return None, stride


class _IRFBlock(torch.nn.Module):
    def __init__(
            self,
            input_depth: int,
            output_depth: int,
            expansion: int,
            stride: _STRIDE_T,
            bn_type: _BN_T = "bn",
            kernel: Literal[1, 3, 5, 7] = 3,
            width_divisor: int = 1,
            shuffle_type: Literal["mid"] | None = None,
            pw_group: int = 1,
            se: bool = False,
            cdw: bool = False,
            dw_skip_bn: bool = False,
            dw_skip_relu: bool = False
    ):
        super().__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = _get_divisible_by(input_depth * expansion, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel_size=1,
            stride=1,
            pad=0,
            no_bias=True,
            use_relu=True,
            bn_type=bn_type,
            group=pw_group,
        )

        # negative stride to do up-sampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = torch.nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel_size=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=True,
                use_relu=True,
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel_size=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=True,
                use_relu=not dw_skip_relu,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = torch.nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel_size=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=True,
                use_relu=not dw_skip_relu,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel_size=1,
            stride=1,
            pad=0,
            no_bias=True,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = _ChannelShuffle(pw_group)

        self.se4 = _SEModule(output_depth) if se else torch.nn.Sequential()

        self.output_depth = output_depth

    # noinspection DuplicatedCode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


def _expand_arch_stages(arch_stages: BlockCfg) -> list[BlockCfg]:
    """For a single block."""
    ret = []
    for idx in range(arch_stages[2]):
        cur = arch_stages[0], arch_stages[1], 1, 1 if idx >= 1 else arch_stages[3]
        ret.append(cur)
    return ret


def _expand_stage_cfg(stage_cfg: StageCfg) -> list[BlockCfg]:
    """For a single stage."""
    ret = []
    for x in stage_cfg:
        ret += _expand_arch_stages(x)
    return ret


def _arch_stages_to_list(arch_stages: list) -> list[ExpandedBlockCfg]:
    ret = []
    for stage_idx, stage in enumerate(arch_stages):
        stage = _expand_stage_cfg(stage)
        for block_idx, block in enumerate(stage):
            cur = {"stage_idx": stage_idx, "block_idx": block_idx, "block": block}
            ret.append(cur)
    return ret


def _add_to_arch(arch: list[ExpandedBlockCfg], info: list[list[str]]):
    """ arch = [{block_0}, {block_1}, ...]
        info = [
            # stage 0
            [
                block0_info,
                block1_info,
                ...
            ], ...
        ]
        convert to:
        arch = [
            {
                block_0,
                name: block0_info,
            },
            {
                block_1,
                name: block1_info,
            }, ...
        ]
    """
    assert isinstance(arch, list) and all(isinstance(x, dict) for x in arch)
    assert isinstance(info, list) and all(isinstance(x, list) for x in info)

    idx = 0
    for stage_idx, stage in enumerate(info):
        for block_idx, block in enumerate(stage):
            assert arch[idx]["stage_idx"] == stage_idx and arch[idx]["block_idx"] == block_idx, \
                f"Index ({stage_idx}, {block_idx}) does not match for block {arch[idx]}"

            assert "stage_op_type" not in arch[idx]
            # ExpandedBlockCfg is converted to ExpandedBlockCfgWithStageOp
            # noinspection PyTypedDict
            arch[idx]["stage_op_type"] = block
            idx += 1


def unify_arch_def(arch_def: ModelArch) -> UnifiedArch:
    """
    Unify the arch_def to:
        {
            ...,
            "arch": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    ...
                },
                {}, ...
            ]
        }
    """
    assert "arch_stages" in arch_def and "stages" in arch_def["arch_stages"]
    assert "stage_op_types" in arch_def
    assert "stages" not in arch_def

    # Copy fields from ArchStages
    # The linter does not support iterating over TypedDict 's keys
    stages: ArchStages = arch_def["arch_stages"]
    # noinspection PyTypedDict
    ret = {x: stages[x] for x in stages}

    # list stages, init as list[ExpandedBlockCfgWithStageOp]
    ret["stages"] = _arch_stages_to_list(stages["stages"])

    # Expand stage op types into stages, converted to list[ExpandedBlockCfgWithStageOp]
    _add_to_arch(ret["stages"], arch_def["stage_op_types"])

    return ret


def get_num_stages(arch_def: UnifiedArch) -> int:
    ret = 0
    for x in arch_def["stages"]:
        ret = max(x["stage_idx"], ret)
    return ret + 1


def get_blocks(arch_def: UnifiedArch,
               stage_indices: Iterable[int] | None = None,
               block_indices: Iterable[int] | None = None) -> UnifiedArch:
    ret = copy.deepcopy(arch_def)
    ret["stages"] = []
    for block in arch_def["stages"]:
        if (stage_indices and block["stage_idx"] in stage_indices) or \
                (block_indices and block["block_idx"] in block_indices):
            ret["stages"].append(block)

    return ret


class FBNetBuilder:
    def __init__(
            self,
            width_ratio: float,
            bn_type: _BN_T = "bn",
            width_divisor: int = 1,
            dw_skip_bn: bool = False,
            dw_skip_relu: bool = False,
    ):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.bn_type = bn_type
        self.width_divisor = width_divisor
        self.dw_skip_bn = dw_skip_bn
        self.dw_skip_relu = dw_skip_relu

    def add_first(self, stage_info: list[int], dim_in: int = 3, pad: bool = True) -> ConvBNRelu:
        # stage_info: [c, s, kernel], key "first" in ArchStages, UnifiedArch
        assert len(stage_info) >= 2
        channel = stage_info[0]
        stride = stage_info[1]
        out_depth = _get_divisible_by(channel * self.width_ratio, self.width_divisor)
        kernel = 3
        if len(stage_info) > 2:
            kernel = stage_info[2]

        self.last_depth = out_depth

        return ConvBNRelu(
            dim_in,
            out_depth,
            kernel_size=kernel,
            stride=stride,
            pad=kernel // 2 if pad else 0,
            no_bias=True,
            use_relu=True,
            bn_type=self.bn_type,
        )

    def add_blocks(self, blocks: list[ExpandedBlockCfgWithStageOp]) -> torch.nn.Sequential:
        """blocks: [{}, {}, ...]"""
        assert isinstance(blocks, list) and all(isinstance(x, dict) for x in blocks), blocks

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            block_idx = block["block_idx"]
            stage_op_type = block["stage_op_type"]
            tcns = block["block"]
            n = tcns[2]
            assert n == 1
            nn_block = self._add_ir_block(tcns, stage_op_type)
            nn_name = f"xif{stage_idx}_{block_idx}"
            assert nn_name not in modules
            modules[nn_name] = nn_block

        return torch.nn.Sequential(modules)

    def add_last(self, stage_info: tuple[int, float] | tuple[int, float, int]) -> torch.nn.Module:
        """Skip last layer if channel_scale == 0, use the same output channel if channel_scale < 0."""
        # stage_info: [c, s, kernel], key "first" in ArchStages, UnifiedArch
        assert len(stage_info) == 2
        channels = stage_info[0]
        channel_scale = stage_info[1]

        if channel_scale == 0.0:
            return torch.nn.Sequential()

        if channel_scale > 0:
            last_channel = int(channels * self.width_ratio) if self.width_ratio > 1.0 else channels
            last_channel = int(last_channel * channel_scale)
        else:
            last_channel = int(self.last_depth * -channel_scale)
        last_channel = _get_divisible_by(last_channel, self.width_divisor)

        # Can never happen because _get_divisible_by will return a value larger than 0
        # if last_channel == 0:
        #     return torch.nn.Sequential()

        dim_in = self.last_depth
        self.last_depth = last_channel

        return ConvBNRelu(
            dim_in,
            last_channel,
            kernel_size=1,
            stride=1,
            pad=0,
            no_bias=True,
            use_relu=True,
            bn_type=self.bn_type,
        )

    def _add_ir_block(self, tcns: BlockCfg, stage_op_type: str, **kwargs) -> torch.nn.Module:
        t, c, n, s = tcns
        assert n == 1
        out_depth = _get_divisible_by(c * self.width_ratio, self.width_divisor)
        dim_in = self.last_depth

        op = _PRIMITIVES[stage_op_type](
            dim_in,
            out_depth,
            expansion=t,
            stride=s,
            bn_type=self.bn_type,
            width_divisor=self.width_divisor,
            dw_skip_bn=self.dw_skip_bn,
            dw_skip_relu=self.dw_skip_relu,
            **kwargs
        )

        self.last_depth = op.output_depth
        return op
