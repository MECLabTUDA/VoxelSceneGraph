# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTargets
from scene_graph_prediction.modeling.utils import cat, BoxCoder, HardNegativeSampler
from scene_graph_prediction.modeling.utils.box_regression_losses import BoxRegressionLoss
from scene_graph_prediction.structures import BoxList


class LossComputationHybrid:
    """
    Computes the loss for Faster R-CNN. Also supports FPN.
    Note: supports ND.
    """

    def __init__(
            self,
            n_dim: int,
            box_coder: BoxCoder,
            regression_loss: BoxRegressionLoss,
            fg_bg_sampler: HardNegativeSampler,
            num_normal_fg_classes: int,
            weighted_training: bool
    ):
        super().__init__()
        self.n_dim = n_dim
        self.box_coder = box_coder
        self.regression_loss = regression_loss
        self.fg_bg_sampler = fg_bg_sampler
        self.num_normal_fg_classes = num_normal_fg_classes
        self.weighted_training = weighted_training

    def __call__(
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

        cat_labels = cat([proposal.LABELS for proposal in proposals], dim=0).long()
        cat_pred_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0).long()

        # Compute the weight off all boxes
        if self.weighted_training:
            cat_weights = cat([proposal.IMPORTANCE for proposal in proposals], dim=0)
        else:
            cat_weights = torch.ones_like(cat_labels).float()

        # For classification, we need to exclude unique objects
        keep = cat_labels <= self.num_normal_fg_classes

        if keep.any():
            cls_cat_labels = cat_labels[keep]
            cls_cat_pred_labels = cat_pred_labels[keep]
            class_logits = class_logits[keep]

            # First select the logit for the GT class for each object
            index = torch.tensor(list(range(cls_cat_labels.shape[0])))
            relevant_logit = class_logits[index, cls_cat_pred_labels]

            # Sample pos/neg locations
            sampled_pos_masks, sampled_neg_masks = self.fg_bg_sampler(
                # Here we need to provide the labels per image
                [proposal.LABELS[proposal.LABELS <= self.num_normal_fg_classes] for proposal in proposals],
                relevant_logit
            )
            # Note: need binary masks so that we can cat them between images
            sampled_pos_masks = torch.cat(sampled_pos_masks, dim=0)
            sampled_neg_masks = torch.cat(sampled_neg_masks, dim=0)
            all_masks = torch.logical_or(sampled_pos_masks, sampled_neg_masks)

            if torch.any(all_masks):
                cls_weights = cat_weights[keep][all_masks]
                classification_loss = torch.sum(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        relevant_logit[all_masks],
                        # Need to binarize targets, i.e. positive case or not
                        (cls_cat_labels[all_masks] > 0).float(),
                        reduction="none"
                    ) * cls_weights / torch.sum(cls_weights))
            else:
                classification_loss = torch.tensor(0., device=cat_labels.device, requires_grad=True)
        else:
            classification_loss = torch.tensor(0., device=cat_labels.device, requires_grad=True)

        # Get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with advanced indexing
        # noinspection PyTypeChecker
        pos_mask = cat_labels > 0
        labels_pos = cat_labels[pos_mask]

        # Only compute box loss if we have positive matches
        if labels_pos.numel() > 0:
            # It contains the relative delta between the centers and lengths of the proposals to the GT
            regression_targets = cat([proposal.REGRESSION_TARGETS for proposal in proposals], dim=0)

            # Compute the indices to select the correct class-wise regression
            map_indexes = self.n_dim * 2 * labels_pos[:, None] + \
                          torch.tensor(list(range(self.n_dim * 2)), device=class_logits.device)
            sampled_pos_indexes = torch.nonzero(pos_mask).squeeze(1)
            box_regression = box_regression[sampled_pos_indexes[:, None], map_indexes]
            regression_targets = regression_targets[pos_mask]

            if not self.regression_loss.require_box_coding:
                # Create BoxList from decoded regressions
                box_regression = BoxList(
                    self.box_coder.decode(box_regression, cat([proposal.boxes for proposal in proposals])[pos_mask]),
                    (1,) * self.n_dim,
                    BoxList.Mode.zyxzyx
                )
                regression_targets = BoxList(regression_targets, (1,) * self.n_dim, BoxList.Mode.zyxzyx)

            reg_weights = cat_weights[pos_mask]
            loss_vector = self.regression_loss(box_regression, regression_targets)
            box_loss = torch.sum(loss_vector * reg_weights) / (torch.mean(reg_weights) * loss_vector.numel())
        else:
            box_loss = torch.tensor(0., device=cat_labels.device, requires_grad=True)

        return classification_loss, box_loss


def build_roi_box_loss_evaluator_hybrid(cfg: CfgNode, box_coder: BoxCoder) -> LossComputationHybrid:
    fg_bg_sampler = HardNegativeSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)

    return LossComputationHybrid(
        n_dim=cfg.INPUT.N_DIM,
        box_coder=box_coder,
        regression_loss=BoxRegressionLoss.build(cfg),
        fg_bg_sampler=fg_bg_sampler,
        num_normal_fg_classes=cfg.INPUT.N_OBJ_CLASSES - cfg.INPUT.N_UNIQUE_OBJ_CLASSES - 1,
        weighted_training=cfg.MODEL.WEIGHTED_BOX_TRAINING
    )
