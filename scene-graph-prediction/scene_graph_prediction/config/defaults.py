# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from scene_graph_prediction.utils.config import AccessTrackingCfgNode

from .dataloader import DATALOADER
from .datasets import DATASETS
from .input import INPUT
from .model import MODEL
from .solver import SOLVER
from .test import TEST

# -------------------------------------------------------------------------------------------------------------------- #
# Convention about Training / Test specific parameters
# -------------------------------------------------------------------------------------------------------------------- #
# Whenever an argument can be either used for training or for testing, the corresponding name will be post-fixed by 
# a _TRAIN for a training parameter, or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be INPUT.MAX_SIZE_TRAIN, 
# while for testing it will be INPUT.MAX_SIZE_TEST

# -------------------------------------------------------------------------------------------------------------------- #
# Config definition
# -------------------------------------------------------------------------------------------------------------------- #
# Checkout the other files in this module to learn more about keys for each specific application

_c = AccessTrackingCfgNode()

_c.MODEL = MODEL
_c.INPUT = INPUT
_c.DATASETS = DATASETS
_c.DATALOADER = DATALOADER
_c.SOLVER = SOLVER
_c.TEST = TEST

# -------------------------------------------------------------------------------------------------------------------- #
# Misc options
# -------------------------------------------------------------------------------------------------------------------- #
# Path to the directory where to save the model, logs, checkpoints, predictions...
_c.OUTPUT_DIR = "."
# Path to a base config file and allows for config referencing. If not "", the base config is loaded recursively.
_c.BASE_CFG = ""
# Path to the directory containing the GloVe embeddings.
# Note: if "", then defaults to the .cache directory in the data folder.
_c.GLOVE_DIR = ""
# Whether to enable metrics tracking with aim
_c.AIM_TRACKING = False
