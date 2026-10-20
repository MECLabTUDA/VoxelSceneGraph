# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

from enum import Enum

import torch

from scene_graph_prediction.config import cfg


class NormType(Enum):
    Group = "Group"
    Instance = "Instance"


def _get_group_gn(dim: int, dim_per_gp: int, num_groups: int) -> int:
    """Returns the number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, f"dim: {dim}, dim_per_gp: {dim_per_gp}"
        return dim // dim_per_gp

    assert dim % num_groups == 0, f"dim: {dim}, num_groups: {num_groups}"
    return num_groups


def build_group_norm(out_channels: int, affine: bool = True, divisor: int = 1) -> torch.nn.GroupNorm:
    """Supports ND natively."""
    assert out_channels % divisor == 0
    out_channels //= divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON  # default: 1e-5
    return torch.nn.GroupNorm(
        _get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def build_instance_norm(n_dim: int, out_channels: int) -> torch.nn.InstanceNorm2d | torch.nn.InstanceNorm3d:
    """Supports only 2D and 3D."""
    norm_module = torch.nn.InstanceNorm2d if n_dim == 2 else torch.nn.InstanceNorm3d
    eps = cfg.MODEL.GROUP_NORM.EPSILON  # default: 1e-5
    return norm_module(out_channels, affine=True, eps=eps)


# noinspection DuplicatedCode
def build_conv3x3(
        n_dim: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        stride: int = 1,
        use_gn: bool = False,
        use_relu: bool = False,
        kaiming_init: bool = True
) -> torch.nn.Module:
    assert n_dim in [2, 3]
    conv_module = torch.nn.Conv2d if n_dim == 2 else torch.nn.Conv3d
    conv = conv_module(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )

    if kaiming_init:
        torch.nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)

    if not use_gn:
        torch.nn.init.constant_(conv.bias, 0)

    modules = [conv]
    if use_gn:
        modules.append(build_group_norm(out_channels))
    if use_relu:
        modules.append(torch.nn.ReLU(inplace=True))

    if len(modules) > 1:
        return torch.nn.Sequential(*modules)
    return conv


def build_fc(dim_in: int, hidden_dim: int, use_gn: bool = False) -> torch.nn.Module:
    """Caffe2 implementation uses XavierFill, which corresponds to kaiming_uniform_ in PyTorch."""
    if use_gn:
        fc = torch.nn.Linear(dim_in, hidden_dim, bias=False)
        torch.nn.init.kaiming_uniform_(fc.weight, a=1)
        return torch.nn.Sequential(fc, build_group_norm(hidden_dim))

    fc = torch.nn.Linear(dim_in, hidden_dim)
    torch.nn.init.kaiming_uniform_(fc.weight, a=1)
    torch.nn.init.constant_(fc.bias, 0)
    return fc


# noinspection DuplicatedCode
def build_conv(
        n_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] | int,
        stride: tuple[int, ...] | int,
        padding: tuple[int, ...] | int,
        norm: NormType | None = None,
        activation: bool = True,
        transposed: bool = False
) -> torch.nn.Module:
    assert n_dim in [2, 3]
    if not transposed:
        conv_module = torch.nn.Conv2d if n_dim == 2 else torch.nn.Conv3d
    else:
        conv_module = torch.nn.ConvTranspose2d if n_dim == 2 else torch.nn.ConvTranspose3d
    has_bias = norm is None
    # noinspection PyTypeChecker
    conv = conv_module(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=has_bias
    )
    # Caffe2 implementation uses XavierFill, which corresponds to kaiming_uniform_ in PyTorch
    torch.nn.init.kaiming_uniform_(conv.weight, a=1)
    if has_bias:
        torch.nn.init.constant_(conv.bias, 0)

    module = [conv]
    if norm == NormType.Group:
        module.append(build_group_norm(out_channels))
    elif norm == NormType.Instance:
        module.append(build_instance_norm(n_dim, out_channels))

    if activation:
        module.append(torch.nn.ReLU(inplace=True))

    if len(module) > 1:
        module = torch.nn.Sequential(*module)
        module.out_channels = out_channels
        return module
    return conv
