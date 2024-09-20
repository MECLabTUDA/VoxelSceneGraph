# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from collections import OrderedDict
from typing import Any

import torch

_STATE_DICT_T = dict[str, Any]


def _align_and_update_state_dicts(
        model_state_dict: _STATE_DICT_T,
        loaded_state_dict: _STATE_DICT_T,
        load_mapping: _STATE_DICT_T
):
    """
    Strategy: suppose that the models that we will create will have prefixes appended to each of its keys,
    for example due to an extra level of nesting that the original pre-trained weights from ImageNet won't contain.
    For example, model.state_dict() might return backbone[0].body.res2.conv1.weight,
    while the pre-trained model contains res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with the longest size of the corresponding name.
    For example, for the same model as before, the pre-trained weight file can contain both res2.conv1.weight,
    and conv1.weight. In this case, we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    logger = logging.getLogger(__name__)
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # Get a matrix of string matches,
    # where each (i, j) entry correspond to the size of the loaded_key string,
    # if it matches
    # Note: Kaihua Tang, since some modules of current model will be initialized from assigned layer of
    # loaded model, we use load_mapping to do such operation
    mapped_current_keys = current_keys.copy()
    for i, key in enumerate(mapped_current_keys):
        for source_key, target_key in load_mapping.items():
            if source_key in key:
                mapped_current_keys[i] = key.replace(source_key, target_key)
                logger.debug(f"MAPPING {key} in current model to {mapped_current_keys[i]} in loaded model.")

    match_matrix = [len(j) if i.endswith(j) else 0 for i in mapped_current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, indexes = match_matrix.max(1)
    # Remove indices that correspond to no-match
    indexes[max_match_size == 0] = -1

    # Used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    for idx_new, idx_old in enumerate(indexes.tolist()):
        if idx_old == -1:
            key = current_keys[idx_new]
            logger.debug(
                f"NO-MATCHING of current module: {key:<{max_size}} of shape "
                f"{str(tuple(model_state_dict[key].shape)):<{max_size_loaded}}"
            )
            continue

        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]

        # add a control gate for this logger (it's too large)
        if not key.startswith('module.') and key != key_old or \
                key.startswith('module.') and key[7:] != key_old:
            logger.debug(f"REMATCHING! {key} loaded from {key_old} of shape {tuple(loaded_state_dict[key_old].shape)}")


def _strip_prefix_if_present(state_dict: _STATE_DICT_T, prefix: str) -> _STATE_DICT_T:
    """If all keys in state_dict start with prefix, then add new keys without the prefix."""
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model: torch.nn.Module, loaded_state_dict: _STATE_DICT_T, load_mapping: dict):
    model_state_dict = model.state_dict()
    # If the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = _strip_prefix_if_present(loaded_state_dict, prefix="module.")
    _align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping)

    # Use strict loading because of the extra unused/unaligned keys
    model.load_state_dict(model_state_dict)
