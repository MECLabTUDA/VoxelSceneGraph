# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import PIL

from torch.utils.collect_env import get_pretty_env_info


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += f"\n        Pillow ({PIL.__version__})"
    return env_str
