# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch

from scene_graph_prediction.config import cfg as g_cfg
# import modules to register rpn heads
from scene_graph_prediction.modeling.backbone import build_backbone  # NoQA
from scene_graph_prediction.modeling.region_proposal import build_rpn  # NoQA
from scene_graph_prediction.modeling.registries import *
from tests._utils import load_config

# overwrite configs if specified, otherwise default config is used
RPN_CFGS = {
}


class TestRPNHeads(unittest.TestCase):
    def test_build_rpn_heads(self):
        """Make sure rpn heads run."""

        self.assertGreater(len(RPN_HEADS), 0)

        in_channels = 64
        num_anchors = 10

        for name, builder in RPN_HEADS.items():
            with self.subTest(name, name=name, builder=builder):
                if name in RPN_CFGS:
                    cfg = load_config(RPN_CFGS[name])
                else:
                    # Use default config if config file is not specified
                    cfg = copy.deepcopy(g_cfg)

                rpn = builder(cfg, in_channels, in_channels, num_anchors)

                n, c_in, h, w = 2, in_channels, 24, 32
                input_tensor = torch.rand([n, c_in, h, w], dtype=torch.float32)
                layers = 3
                out = rpn([input_tensor] * layers)
                self.assertEqual(len(out), 2)
                logits, bbox_reg = out
                for idx in range(layers):
                    self.assertEqual(
                        logits[idx].shape,
                        torch.Size([
                            input_tensor.shape[0], num_anchors,
                            input_tensor.shape[2], input_tensor.shape[3],
                        ])
                    )
                    self.assertEqual(
                        bbox_reg[idx].shape,
                        torch.Size([
                            logits[idx].shape[0], num_anchors * 4,
                            logits[idx].shape[2], logits[idx].shape[3],
                        ]),
                    )
