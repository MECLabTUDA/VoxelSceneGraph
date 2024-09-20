# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Type

from yacs.config import CfgNode

from scene_graph_prediction.utils.registry import Registry
from .transforms import *

_transform_schemes: Registry[str, list[Type[AbstractTransform]]] = Registry()

# Default scheme for 2D PIL Images
_transform_schemes.register("default", [
    ColorJitter,
    ResizeImage2D,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
    Normalize,
])

# Default scheme for 3D Nifti volumes: just resize and normalize
_transform_schemes.register("HeadCT", [
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomDepthFlip,
    ClipAndRescale,
    AddChannelDim,
    RandomAffine
])


def build_transforms(cfg: CfgNode, is_train: bool = True) -> Compose:
    transform_scheme = _transform_schemes[cfg.INPUT.TRANSFORM_SCHEME]
    return Compose(list(map(lambda trans: trans.build(cfg, is_train), transform_scheme)))
