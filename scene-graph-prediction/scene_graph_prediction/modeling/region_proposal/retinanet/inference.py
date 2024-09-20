import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.region_proposal.two_stage.inference import RPNPostProcessor
from scene_graph_prediction.structures import BoxList, BoxListOps
from .._common.utils import permute_and_flatten
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.region_proposal import BoundingBoxRegression, ClassWiseObjectness
from ...utils import BoxCoder


class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes (only applied during testing).
    Note: supports both one-stage and two-stage methods.
    """

    def __init__(
            self,
            pre_nms_score_thresh: float,

            # FIXME the pre_nms_top_n are not used for RetinaNet
            pre_nms_top_n_train: int,
            post_nms_top_n_train: int,
            pre_nms_top_n_test: int,
            post_nms_top_n_test: int,

            nms_thresh: float,
            n_dim: int,
            box_coder: BoxCoder,
            num_fg_classes: int,  # Excluding the background
            is_binary_classification: bool,
            roi_heads_only: bool = False
    ):
        super().__init__(
            pre_nms_top_n_train,
            post_nms_top_n_train,
            pre_nms_top_n_test,
            post_nms_top_n_test,
            nms_thresh,
            n_dim,
            box_coder,
        )
        # Uses confidence thresholding instead of a fixed number of candidates
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.num_fg_classes = num_fg_classes
        if is_binary_classification:
            assert num_fg_classes == 1, num_fg_classes
        self.is_binary_classification = is_binary_classification
        self.roi_heads_only = roi_heads_only

    def _forward_for_single_feature_map(
            self,
            anchors: list[BoxList],
            cls_logits: ClassWiseObjectness,
            box_regression: BoundingBoxRegression
    ) -> BoxHeadTestProposals:
        """
        :param anchors: List of BoxLists (for an image batch) at a specific feature level
        :param cls_logits: tensor of size N, A * C, *zyx_lengths (H, W in 2D)
        :param box_regression: tensor of size N, A * 2 * n_dim, *zyx_lengths
        """
        n, _, *zyx_lengths = cls_logits.shape
        a = box_regression.size(1) // (2 * self.n_dim)

        # Put in the same format as anchors
        flat_cls_logits = permute_and_flatten(cls_logits, n, a, self.num_fg_classes, *zyx_lengths)
        flat_cls_score = flat_cls_logits.sigmoid()
        flat_box_regression = permute_and_flatten(box_regression, n, a, 2 * self.n_dim, *zyx_lengths)

        # Look for boxes where at least one class has prob above thr
        candidate_indexes = flat_cls_score > self.pre_nms_score_thresh
        # Contains the number of positive matches (all classes together) for each image (count used for topk)
        pre_nms_top_n = candidate_indexes.view(n, -1).sum(1)

        results = []
        # Iterate over images in batch and keep only relevant boxes
        for cls_score, cls_logits, box_regression, per_pre_nms_top_n, per_candidate_indexes, img_anchors in \
                zip(flat_cls_score, flat_cls_logits, flat_box_regression, pre_nms_top_n, candidate_indexes, anchors):
            # Select candidates only and compute topk (we only keep classes with score above thr)
            fg_class_score = cls_score[per_candidate_indexes]
            topk_fg_cls_score, topk_indices = fg_class_score.topk(min(per_pre_nms_top_n, pre_nms_top_n), sorted=False)

            # Note: it's important to do nonzero first,
            # such that the produced indexes can be used on the pre-topk tensors
            # Note: non-zero elements in per_candidate_indexes match the elements considered in the topk
            candidate_non_zeros = per_candidate_indexes.nonzero()[topk_indices]

            per_box_loc = candidate_non_zeros[:, 0]
            per_class = candidate_non_zeros[:, 1]

            proposals = self.box_coder.decode(
                box_regression[per_box_loc].view(-1, 2 * self.n_dim),
                img_anchors.boxes[per_box_loc].view(-1, 2 * self.n_dim)
            )

            boxlist = BoxList(proposals, img_anchors.size, mode=BoxList.Mode.zyxzyx)

            if self.is_binary_classification:
                # We add fields as in an RPN
                boxlist.OBJECTNESS = cls_score[per_box_loc, 0]
            else:
                # We add fields as if we were in a BoxHead
                # Note: +1 to offset for the background class
                boxlist.PRED_LABELS = per_class + 1
                # Note: we use the field PRED_SCORES rather than objectness, because this is a one-stage method
                boxlist.PRED_SCORES = cls_score[per_box_loc, per_class]
                # We need to add dummy logits and boxes for background (-10 is -INF)
                fg_pred_logits = cls_logits[per_box_loc]
                boxlist.PRED_LOGITS = torch.cat(
                    [torch.full((fg_pred_logits.shape[0], 1), -10).to(fg_pred_logits), fg_pred_logits],
                    1
                )
                # Box regression is class agnostic, so we just have to repeat the boxes multiple times
                boxlist.BOXES_PER_CLS = torch.tile(boxlist.boxes, (1, self.num_fg_classes + 1))

            boxlist = BoxListOps.clip_to_image(boxlist, remove_empty=True)
            results.append(boxlist)

        return results

    def _select_over_all_levels(self, boxlists: BoxHeadTestProposals) -> BoxHeadTestProposals:
        score_field = BoxList.PredictionField.OBJECTNESS if self.is_binary_classification else \
            BoxList.PredictionField.PRED_SCORES

        post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test

        results = []
        # Iter over all images in batch and combine all predictions
        for boxlist in boxlists:
            scores = boxlist.get_field(score_field)
            if self.is_binary_classification:
                # We need to fake the labels
                labels = torch.ones(scores.shape[0], dtype=torch.long, device=scores.device)
                boxlist.PRED_LABELS = labels

            # Perform class-wise nms
            post_cw_nms, _ = BoxListOps.nms_classwise(
                boxlist,
                self.nms_thresh,
                max_proposals=post_nms_top_n,
                score_field=score_field,
                label_field=BoxList.PredictionField.PRED_LABELS
            )
            results.append(post_cw_nms)

        return results


def build_retinanet_postprocessor(
        cfg: CfgNode,
        box_coder: BoxCoder,
        is_binary_classification: bool,
        num_fg_classes: int
) -> RetinaNetPostProcessor:
    """
    :param cfg:
    :param box_coder:
    :param is_train:
    :param is_binary_classification: whether RetinaNet is used as an RPN of a two-stage method
                                     (i.e. we only do region proposal)
    :param num_fg_classes: number of classes excluding the background.
    :return:
    """
    return RetinaNetPostProcessor(
        pre_nms_score_thresh=cfg.MODEL.RETINANET.INFERENCE_TH,
        pre_nms_top_n_train=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN,
        post_nms_top_n_train=cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN,
        pre_nms_top_n_test=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST,
        post_nms_top_n_test=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST,
        nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        n_dim=cfg.INPUT.N_DIM,
        box_coder=box_coder,
        num_fg_classes=num_fg_classes,
        is_binary_classification=is_binary_classification,
        roi_heads_only=cfg.MODEL.ROI_HEADS_ONLY
    )
