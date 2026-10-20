from typing import Literal

import torch


class LabelSmoothingRegression(torch.nn.Module):
    def __init__(self, eps: float = 0.01, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unknown reduction {self.reduction}")

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _smooth_label(target: torch.Tensor, length: int, smooth_factor: float):
        """
        Converts targets to one-hot format, and smooths them.

        :param target: target in form with [label1, label2, label batch size]
        :param length: length of one-hot format(number of classes)
        :param smooth_factor: smooth factor for label smooth
        
        :return: smoothed labels in one hot format
        """

        one_hot = torch.zeros((target.size(0), length), device=target.device)
        one_hot[range(target.size(0)), target] = 1 - smooth_factor
        one_hot += smooth_factor / length

        return one_hot

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.size(0) != target.size(0):
            raise ValueError(f"Expected input batch size ({x.size(0)}) to match target batch_size({target.size(0)})")

        if x.dim() < 2:
            raise ValueError(f"Expected input tensor to have least 2 dimensions(got {x.size(0)})")

        if x.dim() != 2:
            raise ValueError(f"Only 2 dimension tensor are implemented, (got {x.size()})")

        smoothed_target = self._smooth_label(target, x.size(1), self.eps)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == "mean":
            return torch.mean(loss)

        if self.reduction == "sum":
            return torch.sum(loss)

        return loss
