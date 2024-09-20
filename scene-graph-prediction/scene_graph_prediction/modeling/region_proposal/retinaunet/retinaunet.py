from yacs.config import CfgNode

from scene_graph_prediction.modeling.region_proposal.retinanet.retinanet import RetinaNetModule
from scene_graph_prediction.structures import ImageList, BoxList
from .loss import build_retinaunet_seg_loss_evaluator
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import RPNProposals
from ...utils.build_layers import build_conv


class RetinaUNetModule(RetinaNetModule):
    """
    Module for RetinaUNet computation. Takes feature maps from the backbone and RetinaNet outputs and losses.
    Note: Require the FPN defined in ..backbone.fpn.py or at least a pyramid with as many levels.
    Note: nnDetection only does binary segmentation (FG vs BG)...
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            # Does not affect segmentation
            override_detec_num_fg_classes: int | None = None
    ):
        super().__init__(cfg, in_channels, anchor_strides, override_detec_num_fg_classes)

        # Note: we're not using self.num_classes for segmentation because we want to have the background class
        self.final_seg_conv = build_conv(self.n_dim, in_channels, cfg.INPUT.N_OBJ_CLASSES,
                                         kernel_size=1, stride=1, padding=0, norm=None, activation=False)
        self.seg_loss_eval = build_retinaunet_seg_loss_evaluator(self.n_dim, cfg.INPUT.N_OBJ_CLASSES)

        # Some processing is not required for training the object detector
        #  e.g. produce the predicted semantic segmentation
        # But this can be required for training later parts of the model, e.g. relation head...
        self.training_requires_full_processing = self.cfg.MODEL.RELATION_ON or self.cfg.MODEL.ROI_HEADS_ONLY

    def _forward_box_detection(self, images: ImageList, features: FeatureMaps):
        anchors, class_logits, box_regression = super()._forward_box_detection(images, features)
        seg_logits = self.final_seg_conv(features[0])
        return anchors, class_logits, box_regression, seg_logits

    def post_process_predictions(
            self, args, targets: list[BoxList] | None = None, keep_seg_logits: bool = False
    ) -> RPNProposals | BoxHeadTestProposals:
        anchors, class_logits, box_regression, seg_logits = args
        boxes = super().post_process_predictions((anchors, class_logits, box_regression), targets)

        if not self.training or self.training_requires_full_processing:
            # Add labelmap to proposal; required even for training relations or evaluation
            # Binary masks are only computed and assigned at evaluation to save memory
            for proposal, seg_logit in zip(boxes, seg_logits):
                seg = seg_logit.argmax(dim=0).int()  # DxHxW (No batch or channel dim)
                proposal.PRED_SEGMENTATION = seg
                if keep_seg_logits:
                    proposal.PRED_SEGMENTATION_LOGITS = seg_logit

        return boxes

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        anchors, class_logits, box_regression, seg_logits = args
        losses = super().loss((anchors, class_logits, box_regression), targets)
        losses["loss_rpn_seg"] = self.seg_loss_eval(seg_logits, targets)
        return losses
