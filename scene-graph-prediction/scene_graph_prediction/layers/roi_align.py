# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Any

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair

from .c_layers import roi_align_forward, roi_align_backward, roi_align_forward_3d, roi_align_backward_3d

_SIZE_T = tuple[int, ...]  # dhw ordered


class _ROIAlign(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
            ctx: Any,
                input_tensor: torch.Tensor,
                roi: torch.Tensor,
                output_size: _SIZE_T,  # h, w
                spatial_scale: float,
            sampling_ratio: int
    ) -> torch.Tensor:
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input_tensor.size()

        return roi_align_forward(
            input_tensor,
            roi,
            spatial_scale,
            output_size[0],
            output_size[1],
            sampling_ratio
        )

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = roi_align_backward(grad_output, rois, spatial_scale, output_size[0], output_size[1],
                                        bs, ch, h, w, sampling_ratio)

        # Number of returned elements needs to be the same as the number of arguments in forward() (except context)
        return grad_input, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass


class _ROIAlign3D(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
            ctx: Any,
                input_tensor: torch.Tensor,
                roi: torch.Tensor,
                output_size: _SIZE_T,  # d, h, w
                spatial_scale: float,
                spatial_scale_depth: float,
            sampling_ratio: int
    ) -> torch.Tensor:
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.spatial_scale_depth = spatial_scale_depth
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input_tensor.size()

        return roi_align_forward_3d(
            input_tensor,
            roi,
            spatial_scale,
            spatial_scale_depth,
            output_size[0],
            output_size[1],
            output_size[2],
            sampling_ratio
        )

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        spatial_scale_depth = ctx.spatial_scale_depth
        sampling_ratio = ctx.sampling_ratio
        bs, ch, d, h, w = ctx.input_shape

        grad_input = roi_align_backward_3d(
            grad_output,
            rois,
            spatial_scale,
            spatial_scale_depth,
            output_size[0],
            output_size[1],
            output_size[2],
            bs,
            ch,
            d,
            h,
            w,
            sampling_ratio
        )

        # Number of returned elements needs to be the same as the number of arguments in forward() (except context)
        return grad_input, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass


class ROIAlign(nn.Module):
    def __init__(self, output_size: _SIZE_T, spatial_scale: float, sampling_ratio: int):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return _ROIAlign.apply(
            input_tensor,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size}, " \
               f"spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio})"


class ROIAlign3D(nn.Module):
    def __init__(self, output_size: _SIZE_T, spatial_scale: float, spatial_scale_depth: float,
                 sampling_ratio: int):
        super().__init__()
        self.output_size = output_size  # d, h, w
        self.spatial_scale = spatial_scale
        self.spatial_scale_depth = spatial_scale_depth
        self.sampling_ratio = sampling_ratio

    def forward(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return _ROIAlign3D.apply(
            input_tensor,
            rois,
            self.output_size,
            self.spatial_scale,
            self.spatial_scale_depth,
            self.sampling_ratio
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size}, " \
               f"spatial_scale={self.spatial_scale}, " \
               f"spatial_scale_depth={self.spatial_scale_depth}, " \
               f"sampling_ratio={self.sampling_ratio})"
