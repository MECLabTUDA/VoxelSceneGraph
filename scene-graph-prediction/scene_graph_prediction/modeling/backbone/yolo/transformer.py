# Ultralytics YOLO, AGPL-3.0 license
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid
from .utils import multi_scale_deformable_attn3d, multi_scale_deformable_attn2d


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            cm: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.0,
            act: nn.Module = nn.GELU(),
            normalize_before: bool = False
    ):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim

        self.ma = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(in_channels, cm)
        self.fc2 = nn.Linear(cm, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def _forward_post(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            pos: torch.Tensor | None = None
    ):
        """Performs forward pass with post-normalization."""
        q = k = self._with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def _forward_pre(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            pos: torch.Tensor | None = None
    ):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self._with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            pos: torch.Tensor | None = None
    ):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self._forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self._forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    # noinspection PyMethodOverriding
    def forward(self, x: torch.Tensor):
        """Forward pass for the AIFI transformer layer."""
        if self.n_dim == 2:
            # 2D
            c, h, w = x.shape[1:]
            pos_embed = self._build_2d_sincos_position_embedding(w, h, c)
            # Flatten [B, C, H, W] to [B, HxW, C]
            x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
            return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        # 3D
        c, d, h, w = x.shape[1:]
        pos_embed = self.build_3d_sincos_position_embedding(w, h, d, c)
        # Flatten [B, C, D, H, W] to [B, DxHxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, d, h, w]).contiguous()

    @staticmethod
    def _build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], 1)[None]

    @staticmethod
    def _build_3d_sincos_position_embedding(
            w: int,
            h: int,
            d: int,
            embed_dim: int = 256,
            temperature: float = 10000.0
    ):
        """Builds 3D sine-cosine position embedding."""
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for 3D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)
        grid_d, grid_h, grid_w = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
        pos_dim = embed_dim // 8
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        out_d = grid_d.flatten()[..., None] @ omega[None]

        return torch.cat(
            [
                torch.sin(out_d), torch.cos(out_d),
                torch.sin(out_h), torch.cos(out_h),
                torch.sin(out_w), torch.cos(out_w),
            ], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, channels: int, num_heads: int):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc2 = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            n_dim: int,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            num_layers: int
    ):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.n_dim = n_dim
        self.conv = None
        if in_channels != out_channels:
            self.conv = Conv(n_dim, in_channels, out_channels)
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(out_channels, num_heads) for _ in range(num_layers)))

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, c, *dhw = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, c, *dhw)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: nn.Module = nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            act: nn.Module = nn.ReLU(),
            sigmoid: bool = False
    ):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if self.sigmoid else x


class LayerNorm(nn.Module):
    """
    Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, n_dim: int, num_channels: int, eps: float = 1e-6):
        """Initialize LayerNorm2d/3d with the given parameters."""
        assert n_dim in [2, 3]
        super().__init__()
        self.n_dim = n_dim
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.n_dim == 2:
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        # TODO test
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(
            self,
            n_dim: int,
            d_model: int = 256,
            n_levels: int = 4,
            n_heads: int = 8,
            n_points: int = 4,
            im2col_step: int = 64  # TODO may need to change this value for 3D
    ):
        """Initialize MSDeformAttn with the given parameters."""
        assert n_dim in [2, 3]
        assert n_points in [2, 4]
        super().__init__()
        self.n_dim = n_dim
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = im2col_step

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * n_dim)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2/3], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(D_0, H_0, W_0), (H_1, W_1), ..., (D_{L-1}, H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        # TODO test this for 3D
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, self.n_dim
        )
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, n_dim
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, self.n_dim:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :self.n_dim] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        if self.n_dim == 2:
            output = multi_scale_deformable_attn2d(value, value_shapes, sampling_locations, attention_weights)
        else:
            output = multi_scale_deformable_attn3d(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
            self,
            n_dim: int,
            d_model: int = 256,
            n_heads: int = 8,
            d_ffn: int = 1024,
            dropout: float = 0.0,
            act: nn.Module = nn.ReLU(),
            n_levels: int = 4,
            n_points: int = 4
    ):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()
        self.n_dim = n_dim

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(n_dim, d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def _with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def _forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        # Self attention
        q = k = self._with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self._with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self._forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
