"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import ObjectClasses, RelationClasses
from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadFeatures, ClassLogits
from scene_graph_prediction.modeling.abstractions.relation_head import RelationContext, RelationHeadProposals
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList
from .._utils.motifs import obj_edge_vectors, to_onehot, encode_box_info
from .._utils.relation import classwise_boxes_iou


class _ScaledDotProductAttention(torch.nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Note: len_k==len_v, and dim_q==dim_k

        :param q: (bsz, len_q, dim_q)
        :param k: (bsz, len_k, dim_k)
        :param v: (bsz, len_v, dim_v)
        :param mask:
        :returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        if mask is not None:
            # noinspection PyUnresolvedReferences
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class _MultiHeadAttention(torch.nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = _ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Note: len_k==len_v, and dim_q==dim_k

        :param q: (bsz, len_q, dim_q)
        :param k: (bsz, len_k, dim_k)
        :param v: (bsz, len_v, dim_v)
        :param mask:
        :returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class _PositionWiseFeedForward(torch.nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = torch.nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = torch.nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge adjacent information. Equal to linear layer if kernel size is 1
        :param x: (bsz, len, dim)
        :returns: output (bsz, len, dim)
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.nn.functional.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class _EncoderLayer(torch.nn.Module):
    """Compose with two layers."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.slf_attn = _MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = _PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self,
            enc_input: torch.Tensor,
            non_pad_mask: torch.Tensor,
            slf_attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn


class _TransformerEncoder(torch.nn.Module):
    """An encoder model with self attention mechanism."""

    def __init__(
            self,
            n_layers: int,
            n_head: int,
            d_k: int,
            d_v: int,
            d_model: int,
            d_inner: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.layer_stack = torch.nn.ModuleList(
            [_EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
             for _ in range(n_layers)]
        )

    def forward(self, input_feats: torch.Tensor, num_objs: list[int]) -> torch.Tensor:
        """
        :param input_feats: bounding box features of a batch (#total_box, d_model)
        :param num_objs: number of bounding box of each image (bsz, )
        :returns: enc_output (#total_box, d_model)
        """
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = torch.nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1) \
            .ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1)  # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1) \
            .lt(num_objs_).unsqueeze(-1)  # (bsz, pad_len, 1)

        # Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class TransformerContext(RelationContext):
    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: ObjectClasses,
            rel_classes: RelationClasses,
            in_channels: int
    ):
        super().__init__()
        self.cfg = cfg
        self.n_dim = cfg.INPUT.N_DIM
        # Setting parameters
        self.mode = SGGEvaluationMode.build(cfg)

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # Rhe following word embedding layer should be initialized by glove.6B before using
        embed_vectors = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = torch.nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = torch.nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vectors, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vectors, non_blocking=True)

        # Position embedding
        self.bbox_embed = torch.nn.Sequential(
            torch.nn.Linear(4 * self.n_dim + 1, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1)
        )
        self.lin_obj = torch.nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = torch.nn.Linear(self.embed_dim + self.hidden_dim + self.in_channels, self.hidden_dim)
        self.out_obj = torch.nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = _TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                               self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_edge = _TransformerEncoder(self.edge_layer, self.num_head, self.k_dim,
                                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(
            self,
            roi_features: BoxHeadFeatures,
            proposals: RelationHeadProposals,
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Labels will be used in DecoderRNN during training
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            # noinspection PyTypeChecker
            obj_labels = cat([proposal.LABELS for proposal in proposals], dim=0)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
        else:
            obj_labels = None

        # Label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field(BoxList.PredictionField.PRED_LOGITS)
                              for proposal in proposals], dim=0).detach()
            obj_embed = torch.nn.functional.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # Bbox embedding will be used as input
        assert proposals[0].mode == BoxList.Mode.zyxzyx
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # Encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)
        obj_feats = self.context_obj(obj_pre_rep, num_objs)

        # Predict obj_dists and obj_predictions
        if self.mode == SGGEvaluationMode.PredicateClassification:
            obj_predictions = obj_labels
            obj_dists = to_onehot(obj_predictions, self.num_obj_cls)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            # NMS when evaluating SceneGraphGeneration (otherwise GT boxes are used, so no need)
            if self.mode == SGGEvaluationMode.SceneGraphGeneration and not self.training and \
                    not self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
                boxes_per_cls = [proposal.get_field(BoxList.PredictionField.BOXES_PER_CLS) for proposal in proposals]
                obj_predictions = self._nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_predictions = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_predictions)), dim=-1)

        # Edge context
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        edge_ctx = self.context_edge(edge_pre_rep, num_objs)

        return obj_dists, obj_predictions, edge_ctx, None

    def _nms_per_cls(
            self,
            obj_dists: torch.Tensor,
            boxes_per_cls: list[torch.Tensor],
            num_objs: list[int]
    ) -> torch.Tensor:
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_predictions = []
        for i in range(len(num_objs)):
            is_overlap = classwise_boxes_iou(boxes_per_cls[i], self.n_dim).cpu().numpy() >= self.nms_thresh  # (n, n, c)

            out_dists_sampled = torch.nn.functional.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for _ in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_predictions.append(out_label.long())
        obj_predictions = torch.cat(obj_predictions, dim=0)
        return obj_predictions
