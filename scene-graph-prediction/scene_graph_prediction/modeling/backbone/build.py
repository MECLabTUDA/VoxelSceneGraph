# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from yacs.config import CfgNode

from .fbnet import FBNetTrunk, build_fbnet_builder
from .fpn import FPN2to6
from .resnet import ResNet
from .unet_3d import UNet3D
from .vgg import VGG16
from ..abstractions.backbone import Backbone
from ..registries import BACKBONES


# Note: this use of Sequential modules with "body" and "fpn" is fucking ugly
# However, a Sequential module is required for BackboneFPN
# Which means that if we want to copy weights around easily, we need this prefix everywhere
# So see Wrappers for Backbone and BackboneFPN that include proper typing,
#  Should be replaced with a proper interface combining Backbone and FPN


@BACKBONES.register("VGG-16")
def _build_vgg_fpn_backbone(cfg: CfgNode) -> Backbone:
    return VGG16(cfg)


@BACKBONES.register("R-50-C4")
@BACKBONES.register("R-50-C5")
@BACKBONES.register("R-101-C4")
@BACKBONES.register("R-101-C5")
def _build_resnet_backbone(cfg: CfgNode) -> Backbone:
    return ResNet(cfg)


@BACKBONES.register("FBNet")
def _build_fbnet(cfg: CfgNode, dim_in: int = 3) -> Backbone:
    assert cfg.INPUT.N_DIM == 2
    builder, arch_def = build_fbnet_builder(cfg)
    body = FBNetTrunk(builder, arch_def, dim_in, cfg.INPUT.N_DIM)
    return body


@BACKBONES.register("UNet")
def _build_unet(cfg: CfgNode) -> Backbone:
    return UNet3D(cfg)


@BACKBONES.register("R-50-FPN")
def _build_retinanet_fpn50(cfg: CfgNode) -> Backbone:
    return FPN2to6(cfg.INPUT.N_DIM, cfg.INPUT.N_CHANNELS)


def build_backbone(cfg: CfgNode) -> Backbone:
    assert cfg.MODEL.BACKBONE.CONV_BODY in BACKBONES, \
        f"cfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    return BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
