import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.utils.label_assignment import assign_label_to_proposals
from scene_graph_prediction.structures import ImageList, BoxList
from ...abstractions.backbone import FeatureMaps, AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import RPN, RPNProposals


class ProposalsFromPrecomputations(RPN[list[BoxList]]):
    """
    Simply uses precomputations as region/object proposals.
    Can be used as an RPN or one-stage detector.
    Ensures that the required fields are available, though it would be best that they are pre-computed as well.
    """
    INF = 10

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        super().__init__()
        self.is_one_stage = cfg.MODEL.RPN.FROM_PRECOMPUTATIONS_ONE_STAGE
        self.compute_boxes_per_cls = not cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION
        self.num_classes = cfg.INPUT.N_OBJ_CLASSES

    def forward(self, images: ImageList, features: FeatureMaps, precomputations: list[BoxList]):
        """
        Support both normal detection and relation detection.

        :param images: (not used)
        :param features: (not used)
        :param precomputations: precomputations that will serve as detections

        :returns: The boxes from the RPN, one BoxList per image
        """
        return precomputations

    def post_process_predictions(
            self, args, targets: list[BoxList] | None = None
    ) -> RPNProposals | BoxHeadTestProposals:
        precomputations: list[BoxList] = args
        # Note: we assume that there is no reasonable setup, where we do not need to do the pos-processing

        # Make sure that all usual fields are there
        # Otherwise, do your best to fill the gaps
        if self.is_one_stage:
            # Add the PRED_SCORES, PRED_LOGITS, and BOXES_PER_CLS fields
            for precomputation in precomputations:
                # If the PRED_LABELS field is missing, we're cooked
                if not precomputation.has_field(BoxList.PredictionField.PRED_LABELS):
                    raise RuntimeError("Pre-computations NEED to have the PRED_LABELS field!")
                if precomputation.has_fields(
                        BoxList.PredictionField.PRED_SCORES,
                        BoxList.PredictionField.PRED_LOGITS,
                        BoxList.PredictionField.BOXES_PER_CLS,
                ):
                    # Pre-computations have these fields already
                    break
                # Replace as little as possible
                if not precomputation.has_field(BoxList.PredictionField.PRED_SCORES):
                    precomputation.PRED_SCORES = torch.ones(len(precomputation), device=precomputation.boxes.device)
                if not precomputation.has_field(BoxList.PredictionField.PRED_LOGITS):
                    precomputation.PRED_LOGITS = torch.full(
                        (len(precomputation), self.num_classes),
                        -self.INF,
                        device=precomputation.boxes.device
                    )
                    precomputation.PRED_LOGITS[
                        torch.tensor(list(range(len(precomputation)))), precomputation.PRED_LABELS] = self.INF
                if self.compute_boxes_per_cls and not precomputation.has_field(BoxList.PredictionField.BOXES_PER_CLS):
                    precomputation.BOXES_PER_CLS = torch.tile(precomputation.boxes, (1, self.num_classes))

        else:
            # Add the OBJECTNESS field
            for precomputation in precomputations:
                if precomputation.has_field(BoxList.PredictionField.OBJECTNESS):
                    # Pre-computations have this field already
                    break
                precomputation.OBJECTNESS = torch.ones(len(precomputation), device=precomputation.boxes.device)

        return precomputations

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        # noinspection PyTypeChecker
        return {}

    def assign_label_to_proposals(self, proposals: list[BoxList], targets: list[BoxList]):
        """
        Optimally, this is already pre-computed as well (i.e. precomputations have the LABELS and MATCHED_IDXS fields).
        """
        if not self.is_one_stage_detector():
            raise RuntimeError("An RPN cannot assign labels to proposals.")
        else:
            if not proposals:
                return proposals
            if proposals[0].has_fields(BoxList.AnnotationField.LABELS, BoxList.PredictionField.MATCHED_IDXS):
                # We assume that pre-computations have all the same fields
                return proposals
            return assign_label_to_proposals(proposals, targets, self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD)

    def is_one_stage_detector(self) -> bool:
        return not self.is_one_stage
