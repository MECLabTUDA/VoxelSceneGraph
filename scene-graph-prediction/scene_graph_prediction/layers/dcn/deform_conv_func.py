from typing import Any

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair

from ..c_layers import *

_PAIR_T = int | tuple[int, int]


class _DeformConvFunction(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
            ctx: Any,
            input_tensor: torch.Tensor,
            offset: torch.Tensor,
            weight: torch.Tensor,
            stride: _PAIR_T = 1,
            padding: _PAIR_T = 0,
            dilation: _PAIR_T = 1,
            groups: int = 1,
            deformable_groups: int = 1,
            im2col_step: int = 64) -> torch.Tensor:
        if input_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor as input, got {input_tensor.dim()}D tensor instead.")

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input_tensor, offset, weight)

        output = input_tensor.new_empty(
            _DeformConvFunction._output_size(input_tensor, weight, ctx.padding, ctx.dilation, ctx.stride))

        ctx.bufs_ = [input_tensor.new_empty(0), input_tensor.new_empty(0)]  # columns, ones

        if not input_tensor.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input_tensor.shape[0])
            assert input_tensor.shape[0] % cur_im2col_step == 0, 'im2col step must divide batch size'
            deform_conv_forward(
                input_tensor,
                weight,
                offset,
                output,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                cur_im2col_step
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        input_tensor, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError

        cur_im2col_step = min(ctx.im2col_step, input_tensor.shape[0])
        assert input_tensor.shape[0] % cur_im2col_step == 0, 'im2col step must divide batch size'

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input_tensor)
            grad_offset = torch.zeros_like(offset)
            deform_conv_backward_input(
                input_tensor,
                offset,
                grad_output,
                grad_input,
                grad_offset,
                weight,
                ctx.bufs_[0],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                cur_im2col_step
            )

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            deform_conv_backward_parameters(
                input_tensor,
                offset,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                1,
                cur_im2col_step
            )

        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(
            input_tensor: torch.Tensor,
                     weight: torch.Tensor,
                     padding: tuple[int, int],
                     dilation: tuple[int, int],
            stride: tuple[int, int]
    ) -> tuple[int, int]:
        channels = weight.size(0)
        output_size = input_tensor.size(0), channels
        for d in range(input_tensor.dim() - 2):
            in_size = input_tensor.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(f"Convolution input is too small (output would be {'x'.join(map(str, output_size))})")
        return output_size

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError


class _ModulatedDeformConvFunction(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
            ctx: Any,
            input_tensor: torch.Tensor,
            offset: torch.Tensor,
            mask: torch.Tensor,
            weight: torch.Tensor,
            bias: float | None = None,
            stride: _PAIR_T = 1,
            padding: _PAIR_T = 0,
            dilation: _PAIR_T = 1,
            groups: int = 1,
            deformable_groups: int = 1
    ) -> torch.Tensor:

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if bias is None:
            bias = input_tensor.new_empty(1)  # fake tensor

        if not input_tensor.is_cuda:
            raise NotImplementedError

        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input_tensor.requires_grad:
            ctx.save_for_backward(input_tensor, offset, mask, weight, bias)

        output = input_tensor.new_empty(_ModulatedDeformConvFunction._infer_shape(ctx, input_tensor, weight))
        ctx.bufs_ = input_tensor.new_empty(0), input_tensor.new_empty(0)

        modulated_deform_conv_forward(
            input_tensor,
            weight,
            bias,
            ctx.bufs_[0],
            offset,
            mask,
            output,
            ctx.bufs_[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        if not grad_output.is_cuda:
            raise NotImplementedError

        input_tensor, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input_tensor)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        modulated_deform_conv_backward(
            input_tensor,
            weight,
            bias,
            ctx.bufs_[0],
            offset,
            mask,
            ctx.bufs_[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )

        if not ctx.with_bias:
            grad_bias = None

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx: Any, input_tensor: torch.Tensor, weight: torch.Tensor) -> tuple[int, int, int, int]:
        n = input_tensor.size(0)
        channels_out = weight.size(0)
        height, width = input_tensor.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError


deform_conv = _DeformConvFunction.apply
modulated_deform_conv = _ModulatedDeformConvFunction.apply
