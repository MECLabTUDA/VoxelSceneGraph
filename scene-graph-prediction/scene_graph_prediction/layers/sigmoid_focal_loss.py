from typing import Literal

import torch
from torchvision.ops import sigmoid_focal_loss


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: Literal["sum", "mean", "none"]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(logits, targets, self.gamma, self.alpha, reduction=self.reduction)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"
