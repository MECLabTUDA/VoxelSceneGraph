# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod

import torch

from scene_graph_prediction.structures import BoxList


class Matcher(ABC):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth element.
    Each predicted element will have exactly zero or one matches;
    each ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix,
    that characterizes how well each (ground-truth, predicted)-pair match.
    For example, if the elements are boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N,
    containing the index of the ground-truth element m that matches to prediction n.
    If there is no match, a negative value is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __call__(self, target: BoxList, proposal: BoxList) -> torch.LongTensor:
        """
        Matches elements between groundtruth (M boxes) and proposals (N boxes). The tensor needs at least one element.
        :return: matches tensor: An N tensor where N[i] is a matched gt in [0, M - 1] or
                                  a negative value indicating that prediction could not be matched.
        """

        # Empty targets or proposals not supported during training
        if len(target) == 0:
            raise ValueError("No ground-truth boxes available for one of the images during training")
        elif len(proposal) == 0:
            raise ValueError("No proposal boxes available for one of the images during training")

        return self._match(target, proposal)

    @abstractmethod
    def _match(self, target: BoxList, proposal: BoxList) -> torch.LongTensor:
        """
        Matches elements between groundtruth (M boxes) and proposals (N boxes). The tensor needs at least one element.
        :return: matches tensor: An N tensor where N[i] is a matched gt in [0, M - 1] or
                                  a negative value indicating that prediction could not be matched.
        """
        raise NotImplementedError
