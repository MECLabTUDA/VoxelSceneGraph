# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import pickle
import re
from collections import OrderedDict
from typing import Any

import torch
from yacs.config import CfgNode

from scene_graph_prediction.utils.registry import Registry


def _rename_basic_resnet_weights(layer_keys: list[str]) -> list[str]:
    """Handles renaming of layers between two naming styles."""
    for old, new in [
        ('"_"', '"."'),
        ('".w"', '".weight"'),
        ('".bn"', '"_bn"'),
        ('".b"', '".bias"'),
        ('"_bn.s"', '"_bn.scale"'),
        ('".biasranch"', '".branch"'),
        ('"bbox.pred"', '"bbox_pred"'),
        ('"cls.score"', '"cls_score"'),
        ('"res.conv1_"', '"conv1_"'),

        # RPN / Faster RCNN
        ('".biasbox"', '".bbox"'),
        ('"conv.rpn"', '"rpn.conv"'),
        ('"rpn.bbox.pred"', '"rpn.bbox_pred"'),
        ('"rpn.cls.logits"', '"rpn.cls_logits"'),

        # Affine-Channel -> BatchNorm renaming
        ('"_bn.scale"', '"_bn.weight"'),

        # Make torchvision-compatible
        ('"conv1_bn."', '"bn1."'),

        ('"res2."', '"layer1."'),
        ('"res3."', '"layer2."'),
        ('"res4."', '"layer3."'),
        ('"res5."', '"layer4."'),

        ('".branch2a."', '".conv1."'),
        ('".branch2a_bn."', '".bn1."'),
        ('".branch2b."', '".conv2."'),
        ('".branch2b_bn."', '".bn2."'),
        ('".branch2c."', '".conv3."'),
        ('".branch2c_bn."', '".bn3."'),

        ('".branch1."', '".downsample.0."'),
        ('".branch1_bn."', '".downsample.1."'),

        # GroupNorm
        ('"conv1.gn.s"', '"bn1.weight"'),
        ('"conv1.gn.bias"', '"bn1.bias"'),
        ('"conv2.gn.s"', '"bn2.weight"'),
        ('"conv2.gn.bias"', '"bn2.bias"'),
        ('"conv3.gn.s"', '"bn3.weight"'),
        ('"conv3.gn.bias"', '"bn3.bias"'),

        ('"downsample.0.gn.s"', '"downsample.1.weight"'),
        ('"downsample.0.gn.bias"', '"downsample.1.bias"')
    ]:
        layer_keys = [k.replace(old, new) for k in layer_keys]
    return layer_keys


def _rename_fpn_weights(layer_keys: list[str], stage_names: list[str]) -> list[str]:
    for mapped_idx, stage_name in enumerate(stage_names, 1):
        suffix = ""
        if mapped_idx < 4:
            suffix = ".lateral"
        layer_keys = [
            k.replace(f"fpn.inner.layer{stage_name}.sum{suffix}", f"fpn_inner{mapped_idx}")
            for k in layer_keys
        ]
        layer_keys = [
            k.replace(f"fpn.layer{stage_name}.sum", f"fpn_layer{mapped_idx}")
            for k in layer_keys
        ]

    layer_keys = [k.replace("rpn.conv.fpn2", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox_pred.fpn2", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls_logits.fpn2", "rpn.cls_logits") for k in layer_keys]

    return layer_keys


def _rename_weights_for_resnet(weights: dict[str, Any], stage_names: list[str]):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # for X-101, rename output to fc1000 to avoid conflicts afterward
    layer_keys = [k if k != "pred_b" else "fc1000_b" for k in layer_keys]
    layer_keys = [k if k != "pred_w" else "fc1000_w" for k in layer_keys]

    # performs basic renaming: _ -> . , etc
    layer_keys = _rename_basic_resnet_weights(layer_keys)

    # FPN
    layer_keys = _rename_fpn_weights(layer_keys, stage_names)

    # Mask R-CNN
    layer_keys = [k.replace("mask.fcn.logits", "mask_fcn_logits") for k in layer_keys]
    layer_keys = [k.replace(".[mask].fcn", "mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "conv5_mask") for k in layer_keys]

    # Keypoint R-CNN
    layer_keys = [k.replace("kps.score.lowres", "kps_score_lowres") for k in layer_keys]
    layer_keys = [k.replace("kps.score", "kps_score") for k in layer_keys]
    layer_keys = [k.replace("conv.fcn", "conv_fcn") for k in layer_keys]

    # Rename for our RPN structure
    layer_keys = [k.replace("rpn.", "rpn.head.") for k in layer_keys]

    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights")

    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        if "_momentum" in k:
            continue
        w = torch.from_numpy(v)
        logger.info(f"C2 name: {k} mapped name: {key_map[k]}")
        new_weights[key_map[k]] = w

    return new_weights


def _load_c2_pickled_weights(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def _rename_conv_weights_for_deformable_conv_layers(state_dict: dict, cfg: CfgNode) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Remapping conv weights for deformable conv weights")
    layer_keys = sorted(state_dict.keys())
    for ix, stage_with_dcn in enumerate(cfg.MODEL.RESNETS.STAGE_WITH_DCN, 1):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            pattern = f".*layer{ix}.*conv2.*"
            r = re.match(pattern, old_key)
            if r is None:
                continue
            for param in ["weight", "bias"]:
                if old_key.find(param) == -1:
                    continue
                new_key = old_key.replace(f"conv2.{param}", f"conv2.conv.{param}")
                logger.info(f"pattern: {pattern}, old_key: {old_key}, new_key: {new_key}")
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
    return state_dict


_C2_STAGE_NAMES = {
    "R-50": ["1.2", "2.3", "3.5", "4.2"],
    "R-101": ["1.2", "2.3", "3.22", "4.2"],
    "R-152": ["1.2", "2.7", "3.35", "4.2"],
}

C2_FORMAT_LOADER = Registry()


@C2_FORMAT_LOADER.register("R-50-C4")
@C2_FORMAT_LOADER.register("R-50-C5")
@C2_FORMAT_LOADER.register("R-101-C4")
@C2_FORMAT_LOADER.register("R-101-C5")
@C2_FORMAT_LOADER.register("R-50-FPN")
@C2_FORMAT_LOADER.register("R-50-FPN-RETINANET")
@C2_FORMAT_LOADER.register("R-101-FPN")
@C2_FORMAT_LOADER.register("R-101-FPN-RETINANET")
@C2_FORMAT_LOADER.register("R-152-FPN")
def load_resnet_c2_format(cfg: CfgNode, file_path: str) -> dict[str, Any]:
    state_dict = _load_c2_pickled_weights(file_path)
    conv_body = cfg.MODEL.BACKBONE.CONV_BODY
    arch = conv_body.replace("-C4", "").replace("-C5", "").replace("-FPN", "")
    arch = arch.replace("-RETINANET", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    # ***********************************
    # for deformable convolutional layer
    state_dict = _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg)
    # ***********************************
    return dict(model=state_dict)


def load_c2_format(cfg: CfgNode, file_path: str) -> dict[str, Any]:
    return C2_FORMAT_LOADER[cfg.MODEL.BACKBONE.CONV_BODY](cfg, file_path)
