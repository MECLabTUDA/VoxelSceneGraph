from typing import Hashable

import torch

from scene_graph_prediction.structures import BoxList


def split_logits_for_each_image(proposals: list[BoxList], field: Hashable, head_output: torch.Tensor):
    """
    Heads in two-stage methods, take a batch of boxes (stacked from multiple images) and produce a single tensor output.
    This function splits this tensor back into multiple tensors (corresponding to each image).
    The split tensors are added as the specified field in the proposals.
    :param proposals: proposals per image.
    :param field: field used to add the produced tensors.
    :param head_output: tensor containing the staked predictions.
    """
    outputs = head_output.split([len(p) for p in proposals])
    for p, o in zip(proposals, outputs):
        p.add_field(field, o)
