# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import glob
import os
import unittest

import torch
from yacs.config import CfgNode

import tests._utils as utils
from scene_graph_prediction.modeling.abstractions import AbstractDetector
from scene_graph_prediction.structures import ImageList

CONFIG_FILES = [
    # bbox
    "ISBI2024/bleed_v1_custom_anc.yaml",
    "ISBI2024/bleed_custom_retinaunet_v3.yaml",
    "MICCAI2024/object_detector.yaml",
    "MICCAI2024/object_detector.yaml",

    # relation
    "MICCAI2024/sgg_imp_full_pred.yaml",
    "MICCAI2024/sgg_imp_full_pred_with_mask.yaml",
    "MICCAI2024/sgg_imp_use_gt.yaml",
    "MICCAI2024/sgg_imp_use_gt_with_mask.yaml",
    "MICCAI2024/sgg_motifs_full_pred.yaml",
    "MICCAI2024/sgg_motifs_full_pred_with_mask.yaml",
    "MICCAI2024/sgg_motifs_use_gt.yaml",
    "MICCAI2024/sgg_motifs_use_gt_with_mask.yaml",
]


EXCLUDED_FOLDERS = [
    "test/cfg_recursive_load"
]

class TestDetectors(unittest.TestCase):
    @staticmethod
    def get_config_files(file_list: list[str], exclude_folders: list[str] | None) -> list[str]:
        cfg_root_path = utils.get_config_root_path()
        if file_list is not None:
            files = [os.path.join(cfg_root_path, x) for x in file_list]
        else:
            files = glob.glob(os.path.join(cfg_root_path, "./**/*.yaml"), recursive=True)

        if exclude_folders is not None:
            files = [x for x in files if not any(fld in x for fld in exclude_folders)]

        return files

    @staticmethod
    def create_model(cfg: CfgNode, device: torch.device) -> torch.nn.Module:
        cfg = copy.deepcopy(cfg)
        cfg.freeze()
        model = AbstractDetector.build(cfg)
        model = model.to(device)
        return model

    @staticmethod
    def create_random_input(cfg: CfgNode, device: torch.device) -> ImageList:
        shape = (1,) + cfg.DATALOADER.SIZE_DIVISIBILITY
        ret = [torch.rand(shape)]
        ret = ImageList.to_image_list(ret, len(shape) - 1)
        ret = ret.to(device)
        return ret

    def _test_run_selected_detectors(self, cfg_files, device):
        """ Make sure models build and run """
        self.assertGreater(len(cfg_files), 0)

        with torch.no_grad():
            for cfg_file in cfg_files:
                with self.subTest(cfg_file=cfg_file):
                    cfg = utils.load_config_from_file(cfg_file)
                    cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 10
                    model = self.create_model(cfg, device)
                    inputs = self.create_random_input(cfg, device)
                    model.eval()
                    output, _ = model(inputs)
                    self.assertEqual(len(output), len(inputs.image_sizes))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_run_selected_detectors_cuda(self):
        """Make sure models build and run on cuda."""
        # Run on selected models
        cfg_files = self.get_config_files(CONFIG_FILES, EXCLUDED_FOLDERS)
        self._test_run_selected_detectors(cfg_files, "cuda")
