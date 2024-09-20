import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import ImageList, BoxList, BoxListOps
from .inference import build_retinanet_postprocessor
from .loss import build_retinanet_loss_evaluator
from .._common import build_anchor_generator
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import RPNHead, RPN, FeatureMapsBoundingBoxRegression, ClassWiseObjectness, \
    ImageAnchors, RPNProposals
from ...utils import BoxCoder, IoUMatcher
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
        self.is_binary_classification = cfg.MODEL.RPN_ONLY or cfg.MODEL.RETINANET.TWO_STAGE

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

        # For relation detection, we need to assign GT labels, using an IoUMatcher with the relation head threshold
        self.rel_object_matcher = IoUMatcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, 0.)

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

        # Only do box selection (NMS) during training if not a one stage method, otherwise just return anchors
        if not self.training or \
                self.cfg.MODEL.RETINANET.TWO_STAGE or \
                self.cfg.MODEL.ROI_HEADS_ONLY or \
                self.cfg.MODEL.RELATION_ON:
            with torch.no_grad():
                boxes = self.box_selector(anchors, class_logits, box_regression, targets)
        else:
            boxes = [BoxListOps.cat(ancs_per_image_per_lvl) for ancs_per_image_per_lvl in anchors]

        if not self.cfg.MODEL.RELATION_ON:
            # Normal box prediction pipeline
            return boxes

        # Relation prediction stuff:
        # During training, we however need to match predictions to groundtruth and
        # add the LABELS (and ATTRIBUTES) fields to the prediction
        if targets is not None:
            # This is needed during validation loss computation of relations
            # We need to assign GT labels, using an IoUMatcher with the relation head threshold
            self._assign_label_to_proposals(boxes, targets)

        return boxes

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        anchors, class_logits, box_regression = args
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, class_logits, box_regression, targets)
        return {"loss_objectness": loss_box_cls, "loss_rpn_box_reg": loss_box_reg}

    def _assign_label_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]):
        """
        Add the LABELS (and ATTRIBUTES) fields of the proposals after matching with groundtruth.
        I.e. converts RPNProposals to BoxHeadTargets.
        A 0 in these fields' tensors means that there was no match.
        Note: only used for relation prediction.
        """
        if len(proposals) == 0:
            return proposals

        fields_to_copy = [BoxList.AnnotationField.LABELS]
        if self.cfg.MODEL.ATTRIBUTE_ON:
            fields_to_copy.append(BoxList.AnnotationField.ATTRIBUTES)

        # For each image in the batch, match proposals to groundtruth and add LABELS field
        # Note: we're doing part of the sampling for relation detection
        #       I.e. we rely on the detector to find out the groundtruth label of the detection,
        #       But we need to keep in mind that the classification may not be final yet
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            if len(proposal) == 0:
                # Still need to add empty attributes
                proposal.LABELS = torch.zeros(0, device=proposal.boxes.device, dtype=torch.int64)
                if self.cfg.MODEL.ATTRIBUTE_ON:
                    proposal.ATTRIBUTES = torch.zeros(
                        (0, target.ATTRIBUTES.shape[1]),
                        device=proposal.boxes.device, dtype=torch.int64
                    )
                continue

            matched_indexes = self.rel_object_matcher(target, proposal)
            proposal.LABELS = target.LABELS.long()[matched_indexes.clamp(min=0)]
            proposal.LABELS[matched_indexes < 0] = 0

            # Attributes
            if self.cfg.MODEL.ATTRIBUTE_ON:
                proposal.ATTRIBUTES = target.ATTRIBUTES.long()[matched_indexes.clamp(min=0)]
                proposal.ATTRIBUTES[matched_indexes < 0] = 0
