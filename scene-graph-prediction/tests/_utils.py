import copy
import os
from argparse import Namespace

from yacs.config import CfgNode

from scene_graph_prediction.config import cfg as g_cfg
from scene_graph_prediction.engine.training_script_blobs import build_config

# Since tests can modify the main global config, we need to make a copy of it
_g_cfg = copy.deepcopy(g_cfg)


def get_config_root_path() -> str:
    """ Path to configs for unit tests """
    # cur_file_dir is root/tests/env_tests
    cur_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    ret = os.path.dirname(cur_file_dir)
    ret = os.path.join(ret, "configs")
    return ret


def load_config(rel_path: str) -> CfgNode:
    """ Load config from file path specified as path relative to config_root """
    args = Namespace()
    args.config_file = os.path.join(get_config_root_path(), rel_path)
    args.opts = []
    # Then we need to revert the main global config to its original state before loading the new config
    g_cfg.defrost()
    g_cfg.merge_from_other_cfg(_g_cfg)
    build_config(args)
    g_cfg.defrost()
    return g_cfg


def load_config_from_file(file_path: str) -> CfgNode:
    """ Load config from file path specified as absolute path """
    args = Namespace()
    args.config_file = file_path
    args.opts = []
    # Then we need to revert the main global config to its original state before loading the new config
    g_cfg.defrost()
    g_cfg.merge_from_other_cfg(_g_cfg)
    build_config(args)
    g_cfg.defrost()
    return g_cfg
