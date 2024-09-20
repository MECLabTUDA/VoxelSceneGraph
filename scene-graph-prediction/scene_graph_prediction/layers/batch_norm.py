# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class FrozenBatchNorm(torch.nn.Module):
    """BatchNorm (ND) where the batch statistics and the affine parameters are fixed."""

    def __init__(self, n: int):
        """:param n: number of channels."""
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # noinspection PyAttributeOutsideInit
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale

        shape = [1] * x.dim()
        shape[1] = -1
        scale = scale.reshape(*shape)
        bias = bias.reshape(*shape)
        # scale = scale.reshape(1, -1, 1, 1)
        # bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
