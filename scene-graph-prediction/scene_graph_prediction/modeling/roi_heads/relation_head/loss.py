# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from yacs.config import CfgNode

from scene_graph_prediction.layers import LabelSmoothingRegression
from scene_graph_prediction.modeling.abstractions.attribute_head import BoxHeadTargets
from scene_graph_prediction.modeling.abstractions.relation_head import RelationLogits
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList
from ._utils.motifs import generate_attributes_target


class RelationLossComputation(torch.nn.Module):
    """Computes the loss for relation triplet."""

    def __init__(
            self,
            batch_size: int,
            num_fg_classes: int,
            positive_weight: float,
            use_label_smoothing: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_fg_classes = num_fg_classes
        self.positive_weight = positive_weight
        self.use_label_smoothing = use_label_smoothing

        if self.use_label_smoothing:
            self.criterion_loss = LabelSmoothingRegression(eps=0.1)
        else:
            self.criterion_loss = torch.nn.CrossEntropyLoss()

    def forward(
            self,
            proposals: BoxHeadTargets,
            rel_labels: list[torch.Tensor],
            relation_logits: list[RelationLogits],
            refined_obj_logits: list[torch.Tensor],
            _: list[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.
        Note: the proposals need the ATTRIBUTES field if attributes are on.
        :returns:
            predicate_loss:
            refined_obj_loss:
            refined_attr_loss: optional
        """
        # Object classification refinement loss
        refined_obj_logits = cat(refined_obj_logits, dim=0)
        obj_labels = cat([proposal.LABELS for proposal in proposals], dim=0)

        losses_fg = []
        for label in torch.unique(obj_labels):
            keep = obj_labels == label
            # noinspection PyTypeChecker
            if not torch.any(keep):
                continue
            filtered_logits = refined_obj_logits[keep]
            filtered_labels = obj_labels[keep]
            cls_loss = self.criterion_loss(filtered_logits, filtered_labels.long())
            losses_fg.append(cls_loss)

        loss_refine_obj = torch.mean(torch.stack(losses_fg))

        # Relation classification refinement loss
        relation_logits = cat(relation_logits, dim=0)
        rel_labels = cat(rel_labels, dim=0)

        pred_labels = relation_logits.argmax(1)
        # noinspection PyTypeChecker
        keep = torch.logical_and(pred_labels > 0, rel_labels == 0)
        # noinspection PyTypeChecker
        if not torch.any(keep):
            loss_rel_fp = torch.tensor(0.).to(loss_refine_obj)
        else:
            filtered_logits = relation_logits[keep]
            filtered_labels = rel_labels[keep]
            loss_rel_fp = self.criterion_loss(filtered_logits, filtered_labels.long())

        losses_rel_fg = []
        for label in range(1, self.num_fg_classes + 1):
            keep = rel_labels == label
            # noinspection PyTypeChecker
            if not torch.any(keep):
                continue
            filtered_logits = relation_logits[keep]
            filtered_labels = rel_labels[keep]
            cls_loss = self.criterion_loss(filtered_logits, filtered_labels.long())
            losses_rel_fg.append(cls_loss)

        # If we only sample images with no relations, then we cannot stack
        if losses_rel_fg:
            # Note: we do not use torch.mean as we do not always have examples from each foreground class
            losses_rel_fg = torch.sum(torch.stack(losses_rel_fg)) * self.positive_weight / self.num_fg_classes
        else:
            losses_rel_fg = 0.

        loss_relation = loss_rel_fp * (1 - self.positive_weight) + losses_rel_fg


        return loss_relation, loss_refine_obj, None


class RelationWithAttributesLossComputation(RelationLossComputation):
    """Computes the loss for relation triplet."""

    def __init__(
            self,
            batch_size: int,
            num_fg_classes: int,
            positive_weight: float,
            use_label_smoothing: bool,
            attribute_refinement: bool,
            num_attributes_cat: int,
            attribute_sampling: bool,
            attribute_bgfg_ratio: int,
    ):
        super().__init__(batch_size, num_fg_classes, positive_weight, use_label_smoothing)
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.attribute_refinement = attribute_refinement
        self.num_attributes_cat = num_attributes_cat

    def forward(
            self,
            proposals: BoxHeadTargets,
            rel_labels: list[torch.Tensor],
            relation_logits: list[RelationLogits],
            refined_obj_logits: list[torch.Tensor],
            refined_att_logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.
        Note: the proposals need the ATTRIBUTES field if attributes are on.
        :returns:
            predicate_loss:
            refined_obj_loss:
            refined_attr_loss: optional
        """
        loss_relation, loss_refine_obj, _ = super().forward(
            proposals, rel_labels, relation_logits, refined_obj_logits, refined_att_logits
        )

        # Handle attributes if necessary
        if self.attribute_refinement:
            assert refined_att_logits is not None
            loss_refine_att = self.forward_attributes(proposals, refined_att_logits)
        else:
            loss_refine_att = None

        return loss_relation, loss_refine_obj, loss_refine_att

    def forward_attributes(
            self,
            proposals: BoxHeadTargets,
            refined_att_logits: list[torch.Tensor]
    ) -> torch.Tensor:
        refined_att_logits = cat(refined_att_logits, dim=0)
        fg_attributes = cat([proposal.get_field(BoxList.AnnotationField.ATTRIBUTES) for proposal in proposals], dim=0)

        attribute_targets, fg_attributes_idx = generate_attributes_target(fg_attributes, self.num_attributes_cat)
        if fg_attributes_idx.sum() > 0:
            # Have at least one bbox got fg attributes
            refined_att_logits = refined_att_logits[fg_attributes_idx > 0]
            attribute_targets = attribute_targets[fg_attributes_idx > 0]
        else:
            refined_att_logits = refined_att_logits[0].view(1, -1)
            attribute_targets = attribute_targets[0].view(1, -1)

        loss_refine_att = self._attribute_loss(
            refined_att_logits,
            attribute_targets,
            fg_bg_sample=self.attribute_sampling,
            bg_fg_ratio=self.attribute_bgfg_ratio
        )
        return loss_refine_att

    def _attribute_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            fg_bg_sample: bool = True,
            bg_fg_ratio: int = 3
    ) -> torch.Tensor:
        if not fg_bg_sample:
            attributes_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            attributes_loss = attributes_loss * self.num_attributes_cat / 20.0
            # noinspection PyTypeChecker
            return attributes_loss

        loss_matrix = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
        fg_loss = loss_matrix[labels.view(-1) > 0]
        bg_loss = loss_matrix[labels.view(-1) <= 0]

        num_fg = fg_loss.shape[0]
        # if there is no fg, add at least one bg
        num_bg = max(int(num_fg * bg_fg_ratio), 1)
        perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
        bg_loss = bg_loss[perm]

        return torch.cat([fg_loss, bg_loss], dim=0).mean()


def build_roi_relation_loss_evaluator(cfg: CfgNode, predictor_supports_attr_refine: bool) -> RelationLossComputation:
    if not cfg.MODEL.ATTRIBUTE_ON or not predictor_supports_attr_refine:
        return RelationLossComputation(
            cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
            cfg.INPUT.N_REL_CLASSES - 1,
            cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_WEIGHT,
            cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS
        )
    return RelationWithAttributesLossComputation(
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.INPUT.N_REL_CLASSES - 1,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_WEIGHT,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.INPUT.N_ATT_CLASSES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
    )
