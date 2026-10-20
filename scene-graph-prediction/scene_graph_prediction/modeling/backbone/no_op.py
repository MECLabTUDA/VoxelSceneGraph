from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.backbone import Backbone, Images, FeatureMaps, AnchorStrides


class NoOpBackbone(Backbone):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.n_dim = cfg.INPUT.N_DIM
        self.out_channels = cfg.INPUT.N_CHANNELS

    def forward(self, x: Images) -> FeatureMaps:
        return [x]

    @property
    def feature_strides(self) -> AnchorStrides:
        return (1,) * self.n_dim,
