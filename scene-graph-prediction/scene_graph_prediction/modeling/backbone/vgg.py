# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.models as models
from yacs.config import CfgNode

from ..abstractions.backbone import Backbone, FeatureMaps, Images, AnchorStrides


class VGG16(Backbone):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        assert cfg.INPUT.N_DIM == 2
        self.n_dim = 2
        self.out_channels = cfg.MODEL.VGG.VGG16_OUT_CHANNELS
        vgg = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
        # noinspection PyProtectedMember
        self.conv_body = torch.nn.Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x: Images) -> FeatureMaps:
        return [self.conv_body(x)]

    @property
    def feature_strides(self) -> AnchorStrides:
        # TODO checkout feature maps sizes...
        raise NotImplementedError("TODO...")
