# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch

from scene_graph_prediction.config import cfg as g_cfg
# import modules to register predictors
from scene_graph_prediction.modeling.backbone import build_backbone  # NoQA
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.roi_heads.roi_heads import build_roi_heads  # NoQA
from tests._utils import load_config

# overwrite configs if specified, otherwise default config is used
PREDICTOR_CFGS = {
}

# overwrite configs if specified, otherwise default config is used
PREDICTOR_INPUT_CHANNELS = {
}


class TestPredictors(unittest.TestCase):
    def _test_predictors(self, predictors: dict, overwrite_cfgs: dict, overwrite_in_channels: dict, hw_size: int):
        """Make sure predictors run."""

        self.assertGreater(len(predictors), 0)

        in_channels_default = 64

        for name, builder in predictors.items():
            if name in overwrite_cfgs:
                cfg = load_config(overwrite_cfgs[name])
            else:
                # Use default config if config file is not specified
                cfg = copy.deepcopy(g_cfg)

            in_channels = overwrite_in_channels.get(
                name, in_channels_default)

            fe = builder(cfg, in_channels)

            n, c_in, h, w = 2, in_channels, hw_size, hw_size
            input_tensor = torch.rand([n, c_in, h, w], dtype=torch.float32).squeeze()
            out = fe(input_tensor)
            yield input_tensor, out, cfg

    def test_roi_box_predictors(self):
        """ Make sure roi box predictors run """
        for cur_in, cur_out, cur_cfg in self._test_predictors(
                ROI_BOX_PREDICTOR,
                PREDICTOR_CFGS,
                PREDICTOR_INPUT_CHANNELS,
                hw_size=1,
        ):
            self.assertEqual(len(cur_out), 2)
            scores, bbox_deltas = cur_out[0], cur_out[1]
            self.assertEqual(scores.shape[1], cur_cfg.INPUT.N_OBJ_CLASSES)
            self.assertEqual(scores.shape[0], cur_in.shape[0])
            self.assertEqual(scores.shape[0], bbox_deltas.shape[0])
            self.assertEqual(scores.shape[1] * 4, bbox_deltas.shape[1])

    def test_roi_keypoints_predictors(self):
        """ Make sure roi keypoint predictors run """
        for cur_in, cur_out, cur_cfg in self._test_predictors(
                ROI_KEYPOINT_PREDICTOR,
                PREDICTOR_CFGS,
                PREDICTOR_INPUT_CHANNELS,
                hw_size=14,
        ):
            self.assertEqual(cur_out.shape[0], cur_in.shape[0])
            self.assertEqual(cur_out.shape[1], cur_cfg.INPUT.N_KP_CLASSES)

    def test_roi_mask_predictors(self):
        """ Make sure roi mask predictors run """
        for cur_in, cur_out, cur_cfg in self._test_predictors(
                ROI_MASK_PREDICTOR,
                PREDICTOR_CFGS,
                PREDICTOR_INPUT_CHANNELS,
                hw_size=14,
        ):
            self.assertEqual(cur_out.shape[0], cur_in.shape[0])
