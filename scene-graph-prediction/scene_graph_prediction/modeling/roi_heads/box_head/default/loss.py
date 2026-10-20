# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets
from scene_graph_prediction.modeling.utils import cat, BoxCoder
from scene_graph_prediction.modeling.utils.box_regression_losses import BoxRegressionLoss
from scene_graph_prediction.structures import BoxList


class FastRCNNLossComputation(torch.nn.Module):
    """
    Computes the loss for Faster R-CNN. Also supports FPN.
    Note: supports ND.
    """

    def __init__(
            self,
            n_dim: int,
            box_coder: BoxCoder,
            regression_loss: BoxRegressionLoss,
            cls_agnostic_bbox_reg: bool = False
    ):
        super().__init__()
        self.n_dim = n_dim
        self.box_coder = box_coder
        self.regression_loss = regression_loss
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(
            self,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,  # These are class-wise regressions
            proposals: BoxHeadTargets
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for Faster R-CNN. This requires that the subsample method has been called beforehand.
        :returns: classification_loss, box_loss
        """

        cat_labels = cat([proposal.LABELS for proposal in proposals], dim=0).long()

        # Compute the weight off all boxes
        if self.weighted_training:
            cat_weights = cat([proposal.IMPORTANCE for proposal in proposals], dim=0)
        else:
            cat_weights = torch.ones_like(cat_labels).float()

        # Only compute class loss if we have proposals
        if cat_labels.numel() > 0:
            classification_loss = torch.sum(
                torch.nn.functional.cross_entropy(class_logits, cat_labels.long(), reduction="none") *
                cat_weights
            ) / torch.sum(cat_weights)
        else:
            classification_loss = torch.tensor(0., device=cat_labels.device, requires_grad=True)

        # Get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with advanced indexing
        # noinspection PyTypeChecker
        sampled_pos_indexes = torch.nonzero(cat_labels > 0).squeeze(1)
        labels_pos = cat_labels[sampled_pos_indexes]

        # Only compute box loss if we have positive matches
        if labels_pos.numel() > 0:
            # The regression_targets field is set by the FastRCNNSampling class
            # It contains the relative delta between the centers and lengths of the proposals to the GT
            # Note: use MODEL.ROI_HEADS.BBOX_REG_WEIGHTS to scale the delta for loss computation
            regression_targets = cat([proposal.REGRESSION_TARGETS for proposal in proposals], dim=0)

            # Compute the indices to select the correct class-wise regression
            if self.cls_agnostic_bbox_reg:
                map_indexes = torch.tensor(list(range(2 * self.n_dim, 4 * self.n_dim)), device=class_logits.device)
            else:
                map_indexes = self.n_dim * 2 * labels_pos[:, None] + \
                              torch.tensor(list(range(self.n_dim * 2)), device=class_logits.device)

            box_regression = box_regression[sampled_pos_indexes[:, None], map_indexes]
            regression_targets = regression_targets[sampled_pos_indexes]

            if not self.regression_loss.require_box_coding:
                # Create BoxList from decoded regressions
                box_regression = BoxList(
                    self.box_coder.decode(box_regression, cat([proposal.boxes for proposal in proposals])),
                    (1,) * self.n_dim,
                    BoxList.Mode.zyxzyx
                )
                regression_targets = BoxList(regression_targets, (1,) * self.n_dim, BoxList.Mode.zyxzyx)

            reg_weights = cat_weights[sampled_pos_indexes]
            loss_vector = self.regression_loss(box_regression, regression_targets)
            box_loss = torch.sum(loss_vector * reg_weights) / (torch.mean(reg_weights) * loss_vector.numel())
        else:
            box_loss = torch.tensor(0., device=cat_labels.device, requires_grad=True)

        return classification_loss, box_loss


def build_roi_box_loss_evaluator(cfg: CfgNode, box_coder: BoxCoder) -> FastRCNNLossComputation:
    return FastRCNNLossComputation(
        cfg.INPUT.N_DIM,
        box_coder=box_coder,
        regression_loss=BoxRegressionLoss.build(cfg),
        cls_agnostic_bbox_reg=cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    )
