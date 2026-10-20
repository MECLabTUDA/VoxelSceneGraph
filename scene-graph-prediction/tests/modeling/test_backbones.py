# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch

from scene_graph_prediction.config import cfg as g_cfg
# import modules to register backbones
from scene_graph_prediction.modeling.backbone import build_backbone  # NoQA
from scene_graph_prediction.modeling.registries import BACKBONES
from tests._utils import load_config

# overwrite configs if specified, otherwise default config is used
# TODO need cfgs for other Retina variants
BACKBONE_CFGS = {
    "RetinaUNetHybrid": "MICCAI2024/object_detector.yaml"
}


class TestBackbones(unittest.TestCase):
    def test_build_backbones(self):
        """ Make sure backbones run """

        self.assertGreater(len(BACKBONES), 0)

        for name, backbone_builder in BACKBONES.items():
            if name == "UNet":
                # Skip UNet3D
                continue

            with self.subTest(name, name=name, backbone_builder=backbone_builder):
                if name in BACKBONE_CFGS:
                    cfg = load_config(BACKBONE_CFGS[name])
                else:
                    # Use default config if config file is not specified
                    cfg = copy.deepcopy(g_cfg)

                backbone = backbone_builder(cfg)

                # Make sures the backbone has `out_channels`
                self.assertIsNotNone(
                    getattr(backbone, 'out_channels', None),
                    f'Need to provide out_channels for backbone {name}'
                )

                # h, w need to be divisible by 64 for the FPN
                n, c_in, h, w = 2, 3, 256, 256
                input_tensor = torch.rand([n, c_in, h, w], dtype=torch.float32)
                out = backbone(input_tensor)
                for cur_out in out:
                    self.assertEqual(cur_out.shape[:2], torch.Size([n, backbone.out_channels]))
