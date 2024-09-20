# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
import unittest

from tests._utils import get_config_root_path, load_config_from_file, load_config


class TestConfigs(unittest.TestCase):
    def test_configs_load(self):
        """ Make sure configs are loadable."""

        cfg_root_path = get_config_root_path()
        files = glob.glob(os.path.join(cfg_root_path, "./**/*.yaml"), recursive=True)
        self.assertGreater(len(files), 0)

        for fn in files:
            with self.subTest(fn):
                load_config_from_file(fn)

    def test_load_base_cfg(self):
        """A <- B"""
        cfg = load_config("test/cfg_recursive_load/B.yaml")
        self.assertEqual(cfg.OUTPUT_DIR, "B")
        self.assertEqual(cfg.INPUT.N_DIM, -2)
        self.assertEqual(cfg.GLOVE_DIR, "A")

    def test_load_base_cfg_chain(self):
        """A <- B <- C"""
        cfg = load_config("test/cfg_recursive_load/C.yaml")
        self.assertEqual(cfg.OUTPUT_DIR, "C")
        self.assertEqual(cfg.INPUT.N_DIM, -2)
        self.assertEqual(cfg.GLOVE_DIR, "A")

    def test_load_base_cfg_loop(self):
        """D <- E <- D"""
        cfg = load_config("test/cfg_recursive_load/D.yaml")
        self.assertEqual(cfg.OUTPUT_DIR, "D")
        self.assertEqual(cfg.INPUT.N_DIM, -1)

    def test_load_base_cfg_self(self):
        """F <- F"""
        cfg = load_config("test/cfg_recursive_load/F.yaml")
        self.assertEqual(cfg.OUTPUT_DIR, "F")
