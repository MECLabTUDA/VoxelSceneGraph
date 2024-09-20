# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import torch

from scene_graph_prediction.structures import BoxList, ImageList
from .backbone import FeatureMaps
from .loss import RPNLossDict

# Anchor Generation
# Note to convert a RawAnchorGrid to a BoxList given an image size as a specific scale:
#  BoxList(anchors_per_feature_map, (image_width, image_height), mode=BoxList.Mode.zyxzyx)

RawAnchorGrid = torch.Tensor
# List for each image of list for each feature level of anchor boxes
# zyxzyx BoxList, field: visibility -> torch.Tensor
ImageAnchors = list[BoxList]


class AnchorGenerator(torch.nn.Module, ABC):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    Uses ImageAnchors i.e. BoxLists with a "visibility" field.
    Note: As I currently understand this, an anchor is simply a point on a grid over a feature map
    """
    n_dim: int

    @abstractmethod
    def num_anchors_per_level(self) -> int:
        """Returns the number of anchors to be predicted at each level."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, image_list: ImageList, feature_maps: FeatureMaps) -> list[ImageAnchors]:
        """Given an ImageList, returns the anchors for each image."""
        raise NotImplementedError


# RPN
# Tensor of shape batch_size, nb_anchors, D, H, W for a specific feature map
Objectness = torch.FloatTensor
# List over all selected feature maps
FeatureMapsObjectness = list[Objectness]
RPNProposals = list[BoxList]  # field: objectness -> Objectness
# Tensor of shape batch_size, nb_anchors * 2 * n_dim, D, H, W
# Use a Box Coder's decode method to convert to BoxList
BoundingBoxRegression = torch.Tensor
FeatureMapsBoundingBoxRegression = list[BoundingBoxRegression]


class RPNHead(torch.nn.Module, ABC):
    """
    Interface for RPN Heads.
    These are used on the FeatureMaps by the FPN modul
    to produce bounding box regressions with their respective objectness.
    """
    n_dim: int

    @abstractmethod
    def forward(self, features: FeatureMaps) -> tuple[FeatureMapsObjectness, FeatureMapsBoundingBoxRegression]:
        raise NotImplementedError


RPNRawPredictions = TypeVar("RPNRawPredictions", bound=tuple)


class RPN(torch.nn.Module, ABC, Generic[RPNRawPredictions]):
    """
    Interface for the RPN module.
    Uses RPNProposals i.e. BoxLists with an "objectness" field.

    Note: usually contains an AnchorGenerator, a RPNHead and an RPN post-processor.
    Note: RPNProposals are obtained from ImageAnchors and also contain the "visibility" field.
    """
    n_dim: int

    @abstractmethod
    def forward(
            self,
            images: ImageList,
            features: FeatureMaps
    ) -> RPNRawPredictions:
        """
        :param images: Images for which we want to compute the predictions.
        :param features: Features computed from the images that are used for computing the predictions.
                          Each tensor in the list correspond to different feature levels

        :returns: Typically, but can also return some other stuff:
            anchors: a per-level list of anchors for the batch
            objectness: a per-level list of objectness for the batch
            box_regression: a per-level list of regressions for the batch
        """
        raise NotImplementedError

    @abstractmethod
    def post_process_predictions(
            self,
            args: RPNRawPredictions,
            targets: list[BoxList] | None = None
    ) -> RPNProposals:
        """
        If cfg.MODEL.RELATION_ON, this method should also do
        :param args: see self.forward().
        :param targets: Ground truth boxes present in the image (optional). Used to add GT objects during training.
        :returns: The predicted boxes from the RPN, one BoxList per image.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(
            self,
            args: RPNRawPredictions,
            targets: list[BoxList]
    ) -> RPNLossDict:
        """
        :param args: see self.forward().
        :param targets: Ground truth boxes present in the image.
        :returns: The losses for the model.
        """
        raise NotImplementedError


# RetinaNet
# Tensor of shape batch_size, nb_anchors * C, D, H, W
ClassWiseObjectness = torch.FloatTensor
# Tensor of shape batch_size, num_classes, D, H, W
SegLogits = torch.FloatTensor
