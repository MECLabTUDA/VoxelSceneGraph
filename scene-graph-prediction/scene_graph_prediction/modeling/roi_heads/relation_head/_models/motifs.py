# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.nn.utils.rnn import PackedSequence
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import DatasetStatistics
from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadFeatures, ClassLogits
from scene_graph_prediction.modeling.abstractions.relation_head import RelationLSTMContext, RelationHeadProposals
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList, BoxListOps
from .._utils.motifs import obj_edge_vectors, sort_by_score, to_onehot, get_dropout_mask, encode_box_info
from .._utils.relation import classwise_boxes_iou


class FrequencyBias(torch.nn.Module):
    """The goal of this is to provide a simplified way of computing P(predicate | obj1, obj2)."""

    def __init__(self, statistics: DatasetStatistics):
        super().__init__()
        pred_dist = statistics["pred_dist"].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = torch.clamp(pred_dist.view(-1, self.num_rels), min=-10)

        self.obj_baseline = torch.nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """:param labels: [batch_size, 2]"""
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob: torch.Tensor) -> torch.Tensor:
        """:param pair_prob: [batch_size, num_obj, 2]"""
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = (
                pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) *
                pair_prob[:, :, 1].contiguous().view(batch_size, 1, num_obj)
        )

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels: torch.Tensor):
        """Alias for self.index_with_labels"""
        return self.index_with_labels(labels)


class DecoderRNN(torch.nn.Module):
    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: list[str],
            embed_dim: int,
            inputs_dim: int,
            hidden_dim: int,
            rnn_drop: float
    ):
        super().__init__()
        self.cfg = cfg
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim
        self.n_dim = cfg.INPUT.N_DIM

        obj_embed_vectors = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = torch.nn.Embedding(len(self.obj_classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vectors, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.rnn_drop = rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = torch.nn.Linear(self.hidden_size, len(self.obj_classes))

        # Use sensible default initializations for parameters.
        with torch.no_grad():
            torch.nn.init.constant_(self.state_linearity.bias, 0.0)
            torch.nn.init.constant_(self.input_linearity.bias, 0.0)

    def _lstm_equations(
            self,
            timestep_input: torch.Tensor,
            previous_state: torch.Tensor,
            previous_memory: torch.Tensor,
            dropout_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                     projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0, and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output *= dropout_mask
        return timestep_output, memory

    # noinspection DuplicatedCode
    def forward(
            self,
            inputs: PackedSequence,
            initial_state: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            boxes_for_nms: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, PackedSequence):
            raise ValueError(f'inputs must be PackedSequence but got {type(inputs)}')

        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self._lstm_equations(
                timestep_input, previous_state, previous_memory, dropout_mask=dropout_mask
            )

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone().long()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed + 1)
            else:
                assert l_batch == 1
                out_dist_sample = softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind + 1)

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = classwise_boxes_iou(
                boxes_for_nms.view(boxes_for_nms.shape[0], len(self.obj_classes), 2 * self.n_dim),
                self.n_dim).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = softmax(torch.cat(out_dists, 0), 1).cpu().numpy()
            out_dists_sampled[:, 0] = 0

            out_commitments = out_commitments[0].new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            out_commitments = out_commitments
        else:
            out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments


class LSTMContext(RelationLSTMContext):
    """Modified from neural-motifs to encode contexts for each object."""

    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: list[str],
            rel_classes: list[str],
            in_channels: int
    ):
        super().__init__()
        self.cfg = cfg
        self.n_dim = cfg.INPUT.N_DIM
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # Mode
        self.mode = SGGEvaluationMode.build(cfg)

        # Word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vectors = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = torch.nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = torch.nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vectors, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vectors, non_blocking=True)

        # Position embedding
        self.pos_embed = torch.nn.Sequential(
            torch.nn.Linear(4 * self.n_dim + 1, 32),
            torch.nn.BatchNorm1d(32, momentum=0.001),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(inplace=True)
        )

        # Object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.obj_dim + self.embed_dim + 128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True
        )
        self.decoder_rnn = DecoderRNN(
            self.cfg, self.obj_classes,
            embed_dim=self.embed_dim,
            inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim + 128,
            hidden_dim=self.hidden_dim,
            rnn_drop=self.dropout_rate
        )
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim + self.hidden_dim + self.obj_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True
        )
        # Map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.sorting_strategy = self.cfg.MODEL.ROI_RELATION_HEAD.MOTIFS.SORTING_STRATEGY
        assert self.sorting_strategy in ["width", "height", "depth", "volume"]
        if self.sorting_strategy == "depth":
            assert self.n_dim > 2, "Sorting by depth is only available for >2D."

        # Untreated average features
        self.average_ratio = 0.0005
        # TODO effect analysis
        # self.effect_analysis = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        # if self.effect_analysis:
        #     self.register_buffer("untreated_dcd_feat",
        #                          torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim + 128))
        #     self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
        #     self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.obj_dim))

    def _sort_rois(self, proposals: list[BoxList]) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Rois are score from left to right."""

        def center_dim(dim: int) -> torch.Tensor:
            c_x = 0.5 * (boxes[:, dim] + boxes[:, 2 * proposals[0].n_dim + dim])
            return c_x.view(-1)

        assert proposals and proposals[0].mode == BoxList.Mode.zyxzyx
        boxes = cat([p.boxes for p in proposals], dim=0)

        match self.sorting_strategy:
            case "width":
                scores = center_dim(-1)
            case "height":
                scores = center_dim(-2)
            case "depth":
                scores = center_dim(-3)
            case "volume":
                scores = BoxListOps.volume(proposals)
            case _:
                raise NotImplementedError(f"Sorting strategy {self.sorting_strategy} is not implemented.")

        scores = scores / scores.max()
        return sort_by_score(proposals, scores)

    def _obj_ctx(
            self,
            obj_feats: torch.Tensor,
            proposals: list[BoxList],
            obj_labels: torch.LongTensor | None = None,
            boxes_per_cls: torch.Tensor | None = None,
            ctx_average: bool = False
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param proposals: BoxLists
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes_per_cls:
        :param ctx_average:
        :returns:
            obj_dists: [num_obj, #classes] new probability distribution.
            obj_predictions: argmax of that distribution.
            obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self._sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed.cpu())
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # TODO effect analysis
        # Untreated decoder input
        # if not self.training and self.effect_analysis and ctx_average:
        #     batch_size = encoder_rep.shape[0]
        #     decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        # else:
        decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        # if self.training and self.effect_analysis:
        #     self.untreated_dcd_feat = moving_average(self.untreated_dcd_feat, decoder_inp, self.average_ratio)

        # Decode in order (either if we're learning to classify objects)
        if self.mode != SGGEvaluationMode.PredicateClassification and \
                not self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            decoder_inp = PackedSequence(decoder_inp, ls_transposed.cpu())
            obj_dists, obj_predictions = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            )
            obj_predictions = obj_predictions[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_predictions = obj_labels
            obj_dists = to_onehot(obj_predictions, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_predictions, encoder_rep, perm, inv_perm, ls_transposed

    def _edge_ctx(
            self,
            inp_feats: torch.Tensor,
            perm: torch.Tensor,
            inv_perm: torch.Tensor,
            ls_transposed: torch.Tensor
    ) -> torch.Tensor:
        """
        Object context and object classification.
        :param inp_feats: [num_obj, img_dim + object embedding0 dim]
        :returns: _edge_ctx [num_obj, #feats] for later
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed.cpu())
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # Map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    # noinspection DuplicatedCode
    def forward(
            self,
            x: BoxHeadFeatures,
            proposals: RelationHeadProposals,
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, None]:
        # Labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.LABELS for proposal in proposals], dim=0)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.PRED_LOGITS
                              for proposal in proposals], dim=0).detach()
            obj_embed = softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == BoxList.Mode.zyxzyx
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        # TODO effect analysis
        # if all_average and self.effect_analysis and (not self.training):
        #     obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        # else:
        obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        boxes_per_cls = None
        # NMS when evaluating SceneGraphGeneration (otherwise GT boxes are used, so no need)
        if self.mode == SGGEvaluationMode.SceneGraphGeneration and not self.training:
            # PredictionField comes from post process of box_head
            boxes_per_cls = cat([proposal.BOXES_PER_CLS for proposal in proposals], dim=0)

        # Object level contextual feature
        obj_dists, obj_predictions, obj_ctx, perm, inv_perm, ls_transposed = self._obj_ctx(
            obj_pre_rep, proposals, obj_labels, boxes_per_cls, ctx_average=ctx_average
        )
        # Edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_predictions.long())

        # TODO effect analysis
        # if (all_average or ctx_average) and self.effect_analysis and not self.training:
        #     obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        # else:
        obj_rel_rep = cat((obj_embed2, x, obj_ctx), -1)

        edge_ctx = self._edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # TODO effect analysis
        # memorize average feature
        # if self.training and self.effect_analysis:
        #     # Registered buffers
        #     # noinspection PyAttributeOutsideInit
        #     self.untreated_obj_feat = moving_average(self.untreated_obj_feat, obj_pre_rep, self.average_ratio)
        #     # noinspection PyAttributeOutsideInit
        #     self.untreated_edg_feat = moving_average(
        #         self.untreated_edg_feat, cat((obj_embed2, x), -1), self.average_ratio
        #     )

        return obj_dists, obj_predictions, edge_ctx, None
