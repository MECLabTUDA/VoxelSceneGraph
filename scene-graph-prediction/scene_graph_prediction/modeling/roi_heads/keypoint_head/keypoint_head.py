import torch
from yacs.config import CfgNode

from .inference import build_roi_keypoint_post_processor
from .loss import build_roi_keypoint_loss_evaluator
from .roi_keypoint_feature_extractors import build_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import build_roi_keypoint_predictor
from .sampling import build_roi_keypoint_samp_processor
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadFeatures
from ...abstractions.keypoint_head import KeypointHeadTargets, KeypointHeadFeatures, KeypointLogits, \
    ROIKeypointHead as AbstractROIKeypointHead, KeypointHeadProposals
from ...abstractions.loss import KeypointHeadLossDict
from ...abstractions.region_proposal import RPNProposals


class ROIKeypointHead(AbstractROIKeypointHead):
    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__(cfg, in_channels)
        self.cfg = cfg.clone()
        self.n_dim = cfg.INPUT.N_DIM
        self.feature_extractor = build_roi_keypoint_feature_extractor(cfg, in_channels, anchor_strides)
        self.predictor = build_roi_keypoint_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = build_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = build_roi_keypoint_loss_evaluator(cfg)
        self.samp_processor = build_roi_keypoint_samp_processor(cfg)
        assert self.feature_extractor.n_dim == self.n_dim
        assert self.predictor.n_dim == self.n_dim
        assert self.post_processor.n_dim == self.n_dim
        assert self.samp_processor.n_dim == self.n_dim

    def subsample(
            self,
            proposals: RPNProposals,
            targets: KeypointHeadTargets
    ) -> list[torch.BoolTensor]:
        """Sample training examples from proposals."""
        if len(proposals) == 0:
            return []

        with torch.no_grad():
            # Add fields "labels"
            sampled_masks = self.samp_processor.subsample(proposals, targets)
        return sampled_masks

    def forward(
            self,
            features: FeatureMaps | BoxHeadFeatures,
            proposals: RPNProposals
    ) -> KeypointLogits:
        """
        Note: regarding BoxList fields:
        - proposals need the "matched_idxs" field
        - targets need the "labels", and "keypoints" fields
        - during training: "keypoints" is added
        - during testing: "keypoint_logits", and "keypoints" are added
        """
        if self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # In this case, features is a tensor as the pooler in the BoxHead already reduces the feature maps
            x: torch.Tensor = features
        else:
            # In this case, features is a list of tensors and will be reduced by the pooler in the extractor
            x: KeypointHeadFeatures = self.feature_extractor(features, proposals)

        kp_logits = self.predictor(x)
        return kp_logits

    def post_process_predictions(
            self,
            kp_logits: KeypointLogits,
            proposals: RPNProposals
    ) -> KeypointHeadProposals:
        """Convert attribute logits to actual predictions and store them in the proposals."""
        result = self.post_processor(kp_logits, proposals)
        return result

    def loss(self, kp_logits: KeypointLogits, proposals: KeypointHeadProposals) -> KeypointHeadLossDict:
        """Compute a loss given attribute logits and proposals with relevant ground truth fields."""
        loss_kp = self.loss_evaluator(proposals, kp_logits)
        return {"loss_kp": loss_kp}
