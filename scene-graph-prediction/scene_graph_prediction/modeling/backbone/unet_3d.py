# Based on implementation: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py


import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.backbone import Backbone, Images, FeatureMaps, AnchorStrides


# FIXME for 3D we need to adjust the strides

def _double_conv(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
        torch.nn.ReLU(inplace=True)
    )


class UNet3D(Backbone):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        assert cfg.INPUT.N_DIM == 3
        self.n_dim = 3

        self.double_conv_down1 = _double_conv(3, 16)
        self.double_conv_down2 = _double_conv(16, 32)
        self.double_conv_down3 = _double_conv(32, 64)
        self.double_conv_down4 = _double_conv(64, 128)

        self.max_pool = torch.nn.MaxPool3d(2)

        self.double_conv_up3 = _double_conv(64 + 128, 64)
        self.double_conv_up2 = _double_conv(32 + 64, 32)
        self.double_conv_up1 = _double_conv(32 + 16, 16)

        self.out_channels = 16

    def forward(self, x: Images) -> FeatureMaps:
        conv1 = self.double_conv_down1(x)
        x = self.max_pool(conv1)

        conv2 = self.double_conv_down2(x)
        x = self.max_pool(conv2)

        conv3 = self.double_conv_down3(x)
        x = self.max_pool(conv3)

        x = self.double_conv_down4(x)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.double_conv_up3(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.double_conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.double_conv_up1(x)

        return [x]

    @property
    def feature_strides(self) -> AnchorStrides:
        return (1, 1, 1),
