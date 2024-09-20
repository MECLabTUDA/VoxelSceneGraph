# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.attribute_head import AttributeHeadProposals, AttributeLogits
from scene_graph_prediction.modeling.utils import cat


class AttributeHeadLossComputation(torch.nn.Module):
    """Compute the loss for attribute head."""

    def __init__(
            self,
            loss_weight: float = 0.1,
            num_attribute_cat: int = 201,
            attribute_sampling: bool = True,
            attribute_bgfg_ratio: int = 5,
            use_binary_loss: bool = True,
            pos_weight: int = 1
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_attribute_cat = num_attribute_cat
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_binary_loss = use_binary_loss
        self.pos_weight = pos_weight

    def forward(self, attribute_logits: AttributeLogits, proposals: AttributeHeadProposals) -> torch.Tensor:
        """Calculate attribute loss."""
        attributes = cat([proposal.ATTRIBUTES for proposal in proposals], dim=0)
        assert attributes.shape[0] == attribute_logits.shape[0]

        # Generate attribute targets
        # noinspection PyTypeChecker
        attribute_targets, selected_idxs = self._generate_attributes_target(attributes)

        attribute_logits = attribute_logits[selected_idxs]
        attribute_targets = attribute_targets[selected_idxs]

        attribute_loss = self._attribute_loss(attribute_logits, attribute_targets)

        # noinspection PyTypeChecker
        return attribute_loss * self.loss_weight

    # noinspection DuplicatedCode
    def _generate_attributes_target(self, attributes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From list of (int) attribute indexes to one-hot encoding.
        :returns:
            one-hot attributes
            selected indexes with pos/neg cases
        """
        max_num_attribute = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attribute_idx = (attributes.sum(-1) > 0).long()
        without_attribute_idx = 1 - with_attribute_idx
        num_pos = int(with_attribute_idx.sum())
        num_neg = int(without_attribute_idx.sum())
        assert num_pos + num_neg == num_obj

        if self.attribute_sampling:
            num_neg = min(num_neg, num_pos * self.attribute_bgfg_ratio) if num_pos > 0 else 1

        attribute_targets = torch.zeros((num_obj, self.num_attribute_cat), device=attributes.device).float()
        if not self.use_binary_loss:
            attribute_targets[without_attribute_idx > 0, 0] = 1.0

        pos_idxs = torch.nonzero(with_attribute_idx).squeeze(1)
        perm = torch.randperm(num_obj - num_pos, device=attributes.device)[:num_neg]
        neg_idxs = torch.nonzero(without_attribute_idx).squeeze(1)[perm]
        selected_idxs = torch.cat((pos_idxs, neg_idxs), dim=0)
        assert selected_idxs.shape[0] == num_neg + num_pos

        for idx in torch.nonzero(with_attribute_idx).squeeze(1).tolist():
            for k in range(max_num_attribute):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets, selected_idxs

    def _attribute_loss(self, logits: AttributeLogits, labels: torch.LongTensor) -> torch.Tensor:
        if self.use_binary_loss:
            pos_weight = torch.FloatTensor([self.pos_weight] * self.num_attribute_cat).cuda()
            all_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            return all_loss

        # Soft cross entropy attribute deteriorate the box head,
        # even with 0.1 weight (although bottom-up top-down use cross entropy attribute)
        all_loss = -torch.nn.functional.softmax(logits, dim=-1).log()
        all_loss = (all_loss * labels).sum(-1) / labels.sum(-1)
        return all_loss.mean()


def build_roi_attribute_loss_evaluator(cfg: CfgNode) -> AttributeHeadLossComputation:
    return AttributeHeadLossComputation(
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT,
        cfg.MODEL.INPUT.N_ATT_CLASSES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.POS_WEIGHT,
    )
