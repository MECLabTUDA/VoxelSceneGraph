# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.modeling.utils.box_regression_losses import BoxRegressionLoss
from scene_graph_prediction.structures import BoxList
from ..default.loss import FastRCNNLossComputation


class LossComputationHybrid(FastRCNNLossComputation):
    """
    Computes the loss for Faster R-CNN. Also supports FPN.
    Note: supports ND.
    """

    def __init__(self, n_dim: int, regression_loss: BoxRegressionLoss, num_normal_fg_classes: int):
        super().__init__(n_dim, regression_loss, cls_agnostic_bbox_reg=False)
        self.num_normal_fg_classes = num_normal_fg_classes

    def forward(
            self,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,  # These are class-wise regressions
            proposals: BoxHeadTargets
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compared to the original version, we exclude unique objects from classification learning.
        We also reuse the logits produced by the RetinaNet as a base and only compute an update (see head.forward()).
        Box regression is learned for all foreground boxes.
        """

        labels = cat([proposal.LABELS for proposal in proposals], dim=0).long()
        # For classification, we need to exclude unique objects
        if labels.numel() > 0:
            keep = labels <= self.num_normal_fg_classes
            classification_loss = torch.nn.functional.cross_entropy(class_logits[keep], labels[keep])
        else:
            classification_loss = torch.tensor(0., device=labels.device, requires_grad=True)

        # Get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with advanced indexing
        # noinspection PyTypeChecker
        pos_mask = labels > 0
        labels_pos = labels[pos_mask]

        # The regression_targets field is set by the FastRCNNSampling class
        # It contains the relative delta between the centers and lengths of the proposals to the GT
        # Note: use MODEL.ROI_HEADS.BBOX_REG_WEIGHTS to scale the delta for loss computation
        regression_targets = cat([proposal.REGRESSION_TARGETS for proposal in proposals], dim=0)

        # Only compute box loss if we have positive matches
        if labels_pos.numel() > 0:
            # Compute the indices to select the correct class-wise regression
            map_indexes = self.n_dim * 2 * labels_pos[:, None] + \
                          torch.tensor(list(range(self.n_dim * 2)), device=class_logits.device)

            box_regression = box_regression[pos_mask[:, None], map_indexes]
            regression_targets = regression_targets[pos_mask]

            if not self.regression_loss.require_box_coding:
                # Create BoxList from decoded regressions
                box_regression = BoxList(
                    self.box_coder.decode(box_regression, cat([proposal.boxes for proposal in proposals])),
                    (1,) * self.n_dim,
                    BoxList.Mode.zyxzyx
                )
                regression_targets = BoxList(regression_targets, (1,) * self.n_dim, BoxList.Mode.zyxzyx)

            box_loss = torch.mean(self.regression_loss(box_regression, regression_targets))
        else:
            box_loss = torch.tensor(0., device=labels.device, requires_grad=True)

        return classification_loss, box_loss


def build_roi_box_loss_evaluator_hybrid(cfg: CfgNode) -> LossComputationHybrid:
    return LossComputationHybrid(
        cfg.INPUT.N_DIM,
        BoxRegressionLoss.build(cfg),
        cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1,
    )
