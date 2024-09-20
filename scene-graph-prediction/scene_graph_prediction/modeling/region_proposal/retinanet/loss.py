from typing import Callable

import torch
from yacs.config import CfgNode

# from scene_graph_prediction.layers import SigmoidFocalLoss
from scene_graph_prediction.modeling.region_proposal.two_stage.loss import RPNLossComputationBase
from scene_graph_prediction.structures import BoxList, BoxListOps
from .._common.utils import concat_box_prediction_layers
from ...abstractions.matcher import Matcher
from ...abstractions.region_proposal import ImageAnchors, ClassWiseObjectness
from ...abstractions.sampler import Sampler
from ...utils import BoxCoder, IoUMatcher, ATSSMatcher
from ...utils.box_regression_losses import BoxRegressionLoss
from ...utils.sampling import HardNegativeSampler


class RetinaNetLossComputation(RPNLossComputationBase):
    """
    This class computes the RetinaNet loss.
    Note: BoxLists proposals need the "labels" field, see RetinaNetProposals
    """

    def __init__(
            self,
            proposal_matcher: Matcher,
            fg_bg_sampler: Sampler,
            box_coder: BoxCoder,
            generate_labels_func: Callable[[BoxList], torch.Tensor],
            weighted_training: bool,
            regression_loss: BoxRegressionLoss
    ):
        super().__init__(proposal_matcher, box_coder, generate_labels_func, weighted_training)
        self.regression_loss = regression_loss
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(
            self,
            anchors: list[ImageAnchors],
            box_cls: list[ClassWiseObjectness],
            box_regression: list[torch.Tensor],
            targets: list[BoxList]  # Also needs the "labels" field in one-stage mode
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat_anchors: list[BoxList] = [BoxListOps.cat(anchors_per_image) for anchors_per_image in anchors]
        labels, matched_targets, weights = self._prepare_targets(
            cat_anchors, targets,
            False, True,
            # Actually only required in one-stage mode
            [BoxList.AnnotationField.LABELS],
            encode_targets=self.regression_loss.require_box_coding
        )

        cat_box_cls, cat_box_regression = concat_box_prediction_layers(box_cls, box_regression)

        with torch.no_grad():
            cat_labels = torch.cat(labels, dim=0)
            cat_targets = torch.cat(matched_targets, dim=0)
            # noinspection PyTypeChecker
            pos_indexes = torch.nonzero(cat_labels > 0).squeeze(1)
            cat_weights = torch.cat(weights, dim=0)

        # Sample pos/neg locations
        sampled_pos_masks, sampled_neg_masks = self.fg_bg_sampler(labels, cat_box_cls.max(1)[0])
        # Note: need binary masks so that we can cat them between images
        sampled_pos_masks = torch.cat(sampled_pos_masks, dim=0)
        sampled_neg_masks = torch.cat(sampled_neg_masks, dim=0)

        # To get bincount of positives: torch.bincount(cat_labels.argmax(1))
        # Assert to avoid indexing errors caused by a misconfiguration
        assert cat_labels.shape[0] == cat_box_regression.shape[0], "Shape mismatch. Anchors are misconfigured..."

        # FIXME maybe do some class-wise weighting?
        # Note: we always train the regression on all positive cases and ignore the sampling
        if pos_indexes.numel() > 0:
            cat_box_regression = cat_box_regression[pos_indexes]
            cat_targets = cat_targets[pos_indexes]

            if not self.regression_loss.require_box_coding:
                # Create BoxList from decoded regressions
                n_dim = cat_box_regression.shape[-1] // 2
                cat_cat_anchors = torch.cat([anc.boxes for anc in cat_anchors])[pos_indexes]
                cat_box_regression = BoxList(
                    self.box_coder.decode(cat_box_regression, cat_cat_anchors),
                    (1,) * n_dim,
                    BoxList.Mode.zyxzyx
                )
                cat_targets = BoxList(cat_targets, (1,) * n_dim, BoxList.Mode.zyxzyx)

            weights = cat_weights[pos_indexes]
            loss_vector = self.regression_loss(cat_box_regression, cat_targets)
            box_loss = torch.sum(loss_vector * weights) / (torch.mean(weights) * loss_vector.numel())
        else:
            box_loss = torch.tensor(0., device=pos_indexes.device, requires_grad=True)

        # BCE loss for classification
        all_masks = torch.logical_or(sampled_pos_masks, sampled_neg_masks)
        if torch.any(all_masks):
            selected_labels = cat_labels[all_masks]
            one_hot = torch.zeros(
                selected_labels.shape[0], cat_box_cls.shape[1] + 1,
                dtype=torch.float32,
                device=selected_labels.device
            )
            one_hot[range(selected_labels.shape[0]), selected_labels.long()] = 1
            # Remove the background class afterward, directly encoding labels without it is annoying
            one_hot = one_hot[:, 1:]
            cls_loss = torch.sum(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    cat_box_cls[all_masks],
                    one_hot,
                    reduction="none"
                ) * cat_weights[all_masks, None]) / (torch.mean(cat_weights[all_masks]) * selected_labels.numel())
        else:
            cls_loss = torch.tensor(0., device=sampled_pos_masks.device, requires_grad=True)

        return cls_loss, box_loss


def build_retinanet_loss_evaluator(
        cfg: CfgNode,
        box_coder: BoxCoder,
        is_binary_classification: bool,
        n_classes: int,
        num_anchors_per_lvl: int
) -> RetinaNetLossComputation:
    # In binary classification, we function as an RPN
    def generate_rpn_labels(matched_targets: BoxList) -> torch.Tensor:
        matched_indexes = matched_targets.MATCHED_IDXS
        # In this case, n_classes should always be 1 (bg class is excluded)
        assert n_classes == 1, n_classes
        return matched_indexes >= 0

    def generate_retinanet_labels(matched_targets: BoxList) -> torch.Tensor:
        matched_indexes = matched_targets.MATCHED_IDXS
        labels = matched_targets.LABELS.long()
        # Note: the LABELS field is copied from Loss._match_targets_to_anchors and can be modified in place
        labels[matched_indexes < 0] = 0  # Set all non-matched boxes to bg
        return labels

    match cfg.MODEL.RPN.MATCHER:
        case "IoUMatcher":
            matcher = IoUMatcher(
                cfg.MODEL.RPN.FG_IOU_THRESHOLD,
                cfg.MODEL.RPN.BG_IOU_THRESHOLD,
                always_keep_best_match=True,
            )
        case "ATSSMatcher":
            matcher = ATSSMatcher(
                num_anchors_per_lvl=num_anchors_per_lvl,
                num_candidates=cfg.MODEL.RPN.ATSS_NUM_CANDIDATES
            )
        case _:
            raise ValueError(f"Unknown matcher class (\"{cfg.MODEL.RPN.MATCHER}\")")

    fg_bg_sampler = HardNegativeSampler(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)

    return RetinaNetLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels if is_binary_classification else generate_retinanet_labels,
        cfg.MODEL.WEIGHTED_BOX_TRAINING,
        regression_loss=BoxRegressionLoss.build(cfg)
    )
