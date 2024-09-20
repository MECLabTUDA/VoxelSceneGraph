import torch

# from scene_graph_prediction.layers import SigmoidFocalLoss
from scene_graph_prediction.structures import BoxList
from ...abstractions.region_proposal import SegLogits


class RetinaUNetSegLossComputation(torch.nn.Module):
    """
    This class only computes the segmentation loss (CE + squared Dice loss).
    Note: BoxLists proposals need the "labelmap" field.
    """

    def __init__(self, n_dim: int, num_classes: int):
        super().__init__()
        self.n_dim = n_dim
        self.num_classes = num_classes  # Including the background

    def forward(self, seg_logits: SegLogits, targets: list[BoxList]) -> torch.Tensor:
        flat_gt_seg = [target.SEGMENTATION.view(-1) for target in targets]
        # Since segmentations are not padded, they can have different sizes
        # As such we flatten them to 1D and concatenate together
        cat_flat_gt_seg = torch.cat(flat_gt_seg)

        # We also need to remove the padding from the seg_logits,
        # so that its shape matches the one of the (unpadded) segmentation
        unpadded_seg_logits = [
            logits[(slice(None),) + tuple(slice(0, target.size[dim]) for dim in range(self.n_dim))]
            for logits, target in zip(seg_logits, targets)
        ]
        # Then we need to permute axes before we can flatten (put channels last)
        # Note: seg_logits.dim() - 1 because we don't have any batch dim anymore
        axes = tuple(range(1, seg_logits.dim() - 1)) + (0,)
        permuted_seg_logits = [logits.permute(axes).view(-1, self.num_classes) for logits in unpadded_seg_logits]
        cat_seg_logits = torch.cat(permuted_seg_logits)

        # Compute the Dice loss
        seg_loss_dice = self._squared_dice_loss(cat_seg_logits, cat_flat_gt_seg)

        # Compute the cross-entropy loss on the groundtruth foreground and predicted false positives
        # Note: nnDetection does not apply any non-linearity before loss call
        pred_seg = torch.argmax(cat_seg_logits, dim=1)
        gt_seg = cat_flat_gt_seg.long()
        fg_union_mask = (pred_seg + gt_seg) > 0
        seg_loss_ce = torch.nn.functional.cross_entropy(cat_seg_logits[fg_union_mask], gt_seg[fg_union_mask])

        return (seg_loss_ce + seg_loss_dice) / 2

    def _squared_dice_loss(self, cat_seg_logits: torch.Tensor, cat_flat_gt_seg: torch.Tensor) -> torch.FloatTensor:
        """Compute the class-wise squared dice loss for foreground classes."""
        one_hot_seg = self._one_hot_encode_seg(cat_flat_gt_seg)
        flat_softmax_logits = torch.nn.functional.softmax(cat_seg_logits, dim=1)[:, 1:]  # Fg only
        intersection = (flat_softmax_logits * one_hot_seg).sum(0)
        denominator = (flat_softmax_logits ** 2 + one_hot_seg ** 2).sum(0)
        eps = 1e-6
        # The background is already excluded from the calculation
        return 1. - torch.mean(((2 * intersection + eps) / (denominator + eps)))

    def _one_hot_encode_seg(self, cat_flat_gt_seg: torch.Tensor) -> torch.Tensor:
        """
        One-hot encode all foreground classes for a GT segmentation (channel 0 for background is already removed).
        :param cat_flat_gt_seg: 1D tensor with GT segmentation.
        :return: the segmentations as one-hot encoded (fg only).
        """
        shape = (cat_flat_gt_seg.shape[0], self.num_classes - 1)
        one_hot = torch.zeros(shape, dtype=torch.long, device=cat_flat_gt_seg.device)
        for cls_idx in range(1, self.num_classes):
            one_hot[:, cls_idx - 1][cat_flat_gt_seg == cls_idx] = 1
        return one_hot


def build_retinaunet_seg_loss_evaluator(n_dim: int, num_classes: int) -> RetinaUNetSegLossComputation:
    return RetinaUNetSegLossComputation(n_dim, num_classes)
