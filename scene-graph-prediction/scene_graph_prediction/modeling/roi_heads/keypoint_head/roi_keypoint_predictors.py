import torch

from scene_graph_prediction.modeling.abstractions.keypoint_head import KeypointHeadFeatures, KeypointLogits
from scene_graph_prediction.modeling.registries import *


@ROI_KEYPOINT_PREDICTOR.register("KeypointRCNNPredictor")
class KeypointRCNNPredictor(ROIKeypointPredictor):
    """Note: supports 2D and 3D."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.n_dim = cfg.INPUT.N_DIM
        assert self.n_dim in [2, 3]
        input_features = in_channels
        num_keypoints = cfg.INPUT.N_KP_CLASSES

        conv_module = torch.nn.ConvTranspose2d if self.n_dim == 2 else torch.nn.ConvTranspose3d
        deconv_kernel = 4
        self.kps_score_lowres = conv_module(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        torch.nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        torch.nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x: KeypointHeadFeatures) -> KeypointLogits:
        x = self.kps_score_lowres(x)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.up_scale,
            mode="bilinear" if self.n_dim == 2 else "trilinear",
            align_corners=False
        )

        return x


def build_roi_keypoint_predictor(cfg: CfgNode, in_channels: int) -> ROIKeypointPredictor:
    predictor = ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
