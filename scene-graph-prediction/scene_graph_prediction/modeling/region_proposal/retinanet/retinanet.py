import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.utils.label_assignment import assign_label_to_proposals
from scene_graph_prediction.structures import ImageList, BoxList, BoxListOps
from .inference import build_retinanet_postprocessor
from .loss import build_retinanet_loss_evaluator
from .._common import build_anchor_generator
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import RPNHead, RPN, FeatureMapsBoundingBoxRegression, ClassWiseObjectness, \
    ImageAnchors, RPNProposals
from ...utils import BoxCoder
from ...utils.build_layers import build_conv, NormType


class RetinaNetHead(RPNHead):
    """
    Adds a RetinaNet head with classification and regression heads.
    Note: supports 2D and 3D.
    """

    def __init__(self, n_dim: int, in_channels: int, num_fg_classes: int, num_anchors: int):
        """
        :param n_dim:
        :param in_channels:
        :param num_fg_classes: number of classes including the background.
        """
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim

        n_features = 256 if self.n_dim == 2 else 128

        # Classification
        self.cls_tower = torch.nn.Sequential(
            # Internal
            build_conv(self.n_dim, in_channels, n_features,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True),
            build_conv(self.n_dim, n_features, n_features,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True),
            # Out
            build_conv(self.n_dim, n_features, num_anchors * num_fg_classes,
                       kernel_size=3, stride=1, padding=1, norm=None, activation=False)
        )

        # Bbox regression
        self.bbox_tower = torch.nn.Sequential(
            # Internal
            build_conv(self.n_dim, in_channels, n_features,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True),
            build_conv(self.n_dim, n_features, n_features,
                       kernel_size=3, stride=1, padding=1, norm=NormType.Group, activation=True),
            # Out
            build_conv(self.n_dim, n_features, num_anchors * self.n_dim * 2,
                       kernel_size=3, stride=1, padding=1, norm=None, activation=False)
        )

    def forward(self, features: FeatureMaps) -> tuple[list[ClassWiseObjectness], FeatureMapsBoundingBoxRegression]:
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_tower(feature))
            bbox_reg.append(self.bbox_tower(feature))
        return logits, bbox_reg


class RetinaNetModule(RPN[tuple[list[ImageAnchors], list[ClassWiseObjectness], FeatureMapsBoundingBoxRegression]]):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RetinaNet outputs and losses.
    Note: Require the FPN defined in ..backbone.fpn.py or at least a pyramid with as many levels.
    """

    def __init__(
            self,
            cfg: CfgNode,
            in_channels: int,
            anchor_strides: AnchorStrides,
            # If not None and not binary_classification, use this number instead of cfg.INPUT.N_OBJ_CLASSES - 1
            override_num_fg_classes: int | None = None
    ):
        super().__init__()
        self.cfg = cfg

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            raise NotImplementedError("Predicate classification is not supported for one-stage methods.")

        nb_selected_maps = len(cfg.MODEL.RETINANET.SELECTED_FEATURE_MAPS)
        assert nb_selected_maps > 0, \
            f"At least one feature map level needs to be selected (currently {nb_selected_maps})"
        self.selected_features_maps = cfg.MODEL.RETINANET.SELECTED_FEATURE_MAPS

        self.n_dim = cfg.INPUT.N_DIM

        box_coder = BoxCoder(weights=(1.,) * self.n_dim + (1.,) * self.n_dim, n_dim=self.n_dim)

        # We only do region proposal (binary classification) if RPN only or this is the RPN for a two-stage method
        self.is_binary_classification = cfg.MODEL.RETINANET.TWO_STAGE
        if cfg.MODEL.RPN_ONLY:
            assert self.is_binary_classification, "Only TWO_STAGE RetinaNets can be trained in RPN_ONLY mode."

        # Select anchor strides corresponding to selected feature maps
        selected_anchor_strides = tuple(anchor_strides[lvl] for lvl in self.selected_features_maps)
        self.anchor_generator = build_anchor_generator(cfg, selected_anchor_strides)

        # Number of foreground classes
        if self.is_binary_classification:
            self.num_fg_classes = 1
        else:
            self.num_fg_classes = (
                cfg.INPUT.N_OBJ_CLASSES - 1
                if override_num_fg_classes is None
                else override_num_fg_classes
            )

        self.head = RetinaNetHead(
            self.n_dim,
            in_channels,
            self.num_fg_classes,
            self.anchor_generator.num_anchors_per_level()
        )

        self.box_selector = build_retinanet_postprocessor(
            cfg,
            box_coder,
            self.is_binary_classification,
            self.num_fg_classes
        )

        self.loss_evaluator = build_retinanet_loss_evaluator(
            cfg,
            box_coder,
            self.is_binary_classification,
            self.num_fg_classes,
            self.anchor_generator.num_anchors_per_level()
        )

        # Some processing is not required for training the object detector
        #  e.g. produce the predicted semantic segmentation
        # But this can be required for training later parts of the model, e.g. relation head...
        self.training_requires_full_processing = (self.cfg.MODEL.BOX_ON or
                                                  self.cfg.MODEL.ATTRIBUTE_ON or
                                                  self.cfg.MODEL.MASK_ON or
                                                  self.cfg.MODEL.KEYPOINT_ON or
                                                  self.cfg.MODEL.RELATION_ON)

    def forward(self, images: ImageList, features: FeatureMaps):
        """
        Support both normal detection and relation detection.

        :param images: Images for which we want to compute the predictions.
        :param features: Features computed from the images that are used for computing the predictions.
                         Each tensor in the list correspond to a different feature level.
                         Note: usually expects the 6 levels produces by the FPN backbone.

        :returns: The predicted boxes from the RPN, one BoxList per image
        """
        # If we're predicting relations from GT boxes, then skip the prediction pipeline
        if self.cfg.MODEL.RELATION_ON and self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            # Actual GT boxes will be prepared by the detector
            return [], [], []
        # Note: do not unpack as RetinaUNet has more stuff
        return self._forward_box_detection(images, features)

    def _forward_box_detection(
            self,
            images: ImageList,
            features: FeatureMaps
    ) -> tuple[list[ImageAnchors], list[ClassWiseObjectness], FeatureMapsBoundingBoxRegression]:
        # Object detection pipeline, we're either:
        # - training for detection
        # - a two-stage method, and we shouldn't handle relation stuff
        assert max(self.selected_features_maps) < len(features)
        # Select which feature level should be used based on the config
        selected_features = [features[lvl] for lvl in self.selected_features_maps]
        class_logits, box_regression = self.head(selected_features)
        anchors = self.anchor_generator(images, selected_features)
        return anchors, class_logits, box_regression

    def post_process_predictions(
            self, args, targets: list[BoxList] | None = None
    ) -> RPNProposals | BoxHeadTestProposals:
        # Note: we know that self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX is False if we're entering this method.

        anchors, class_logits, box_regression = args

        # Only do box selection (NMS) during:
        # - testing
        # - if not a one stage method
        # Otherwise just return anchors
        if not self.training or \
                self.cfg.MODEL.RETINANET.TWO_STAGE or \
                self.training_requires_full_processing:
            with torch.no_grad():
                boxes = self.box_selector(anchors, class_logits, box_regression, targets)
        else:
            boxes = [BoxListOps.cat(ancs_per_image_per_lvl) for ancs_per_image_per_lvl in anchors]

        return boxes

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        anchors, class_logits, box_regression = args
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, class_logits, box_regression, targets)
        return {"loss_objectness": loss_box_cls, "loss_rpn_box_reg": loss_box_reg}

    def assign_label_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]):
        return assign_label_to_proposals(proposals, targets, self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD)

    def is_one_stage_detector(self) -> bool:
        return not self.is_binary_classification
