# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch

from scene_graph_prediction.config import cfg as g_cfg
from scene_graph_prediction.modeling.backbone import build_backbone  # NoQA
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.roi_heads.roi_heads import build_roi_heads  # NoQA
from scene_graph_prediction.structures import BoxList
from tests._utils import load_config

# overwrite configs if specified, otherwise default config is used
FEATURE_EXTRACTORS_CFGS = {
}

# overwrite configs if specified, otherwise default config is used
FEATURE_EXTRACTORS_INPUT_CHANNELS = {
    # in_channels was not used, load through config
    "ResNet50Conv5ROIFeatureExtractor": 1024,
}


class TestFeatureExtractors(unittest.TestCase):
    def _test_feature_extractors(self, extractors, overwrite_cfgs, overwrite_in_channels, required_attr: str):
        """ Make sure roi box feature extractors run """

        self.assertGreater(len(extractors), 0)

        in_channels_default = 64

        for name, builder in extractors.items():
            with self.subTest(name=name):
                if name in overwrite_cfgs:
                    cfg = load_config(overwrite_cfgs[name])
                else:
                    # Use default config if config file is not specified
                    cfg = copy.deepcopy(g_cfg)

                in_channels = overwrite_in_channels.get(name, in_channels_default)

                # Dummy strides
                fe = builder(cfg, in_channels, anchor_strides=((32, 32),))
                self.assertIsNotNone(
                    getattr(fe, required_attr, None),
                    f"Need to provide {required_attr} for feature extractor {name}"
                )
                out_chan = getattr(fe, required_attr)

                n, c_in, h, w = 2, in_channels, 24, 32
                input_tensor = torch.rand([n, c_in, h, w], dtype=torch.float32)
                bboxes = [[1, 1, 10, 10], [5, 5, 8, 8], [2, 2, 3, 4]]
                img_size = 512, 384
                box_list = BoxList(torch.tensor(bboxes), img_size, BoxList.Mode.zyxzyx)
                out = fe([input_tensor], [box_list] * n)
                self.assertEqual(out.shape[:2], torch.Size([n * len(bboxes), out_chan]))

    def test_roi_box_feature_extractors(self):
        """ Make sure roi box feature extractors run """
        self._test_feature_extractors(
            ROI_BOX_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
            "representation_size"
        )

    def test_roi_keypoints_feature_extractors(self):
        """ Make sure roi keypoints feature extractors run """
        self._test_feature_extractors(
            ROI_KEYPOINT_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
            "representation_size"
        )

    def test_roi_mask_feature_extractors(self):
        """ Make sure roi mask feature extractors run """
        self._test_feature_extractors(
            ROI_MASK_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
            "out_channels"
        )
