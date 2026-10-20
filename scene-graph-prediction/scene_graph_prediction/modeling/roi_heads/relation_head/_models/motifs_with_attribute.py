# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.nn.utils.rnn import PackedSequence
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import ObjectClasses, AttributeClasses, RelationClasses
from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.attribute_head import AttributeLogits
from scene_graph_prediction.modeling.abstractions.box_head import ClassLogits
from scene_graph_prediction.modeling.abstractions.relation_head import RelationHeadProposals
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList
from .motifs import DecoderRNN, LSTMContext
from .._utils.motifs import obj_edge_vectors, to_onehot, get_dropout_mask, encode_box_info, \
    generate_attributes_target, normalize_sigmoid_logits
from .._utils.relation import classwise_boxes_iou


class _AttributeDecoderRNN(DecoderRNN):
    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: ObjectClasses,
            att_classes: AttributeClasses,
            embed_dim: int,
            inputs_dim: int,
            hidden_dim: int,
            rnn_drop: float
    ):
        super().__init__(cfg, obj_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop)
        self.att_classes = att_classes
        self.embed_dim = embed_dim
        self.num_attributes_cat = cfg.INPUT.N_ATT_CLASSES
        self.n_dim = cfg.INPUT.N_DIM

        att_embed_vectors = obj_edge_vectors(self.att_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.att_embed = torch.nn.Embedding(len(self.att_classes), embed_dim)
        with torch.no_grad():
            self.att_embed.weight.copy_(att_embed_vectors, non_blocking=True)

        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_att = torch.nn.Linear(self.hidden_size, len(self.att_classes))

    # Note: this method ought to be factorized with DecoderRNN.forward, but attributes are right in the middle...
    # noinspection DuplicatedCode
    def forward(
            self,
            inputs: PackedSequence,
            initial_state: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            boxes_for_nms: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, PackedSequence):
            raise ValueError(f'inputs must be PackedSequence but got {type(inputs)}')

        assert isinstance(inputs, PackedSequence)
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
        previous_att_embed = self.att_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        att_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                previous_att_embed = previous_att_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed, previous_att_embed), 1)

            previous_state, previous_memory = self._lstm_equations(
                timestep_input, previous_state, previous_memory, dropout_mask=dropout_mask
            )

            pred_dist = self.out_obj(previous_state)
            attr_dist = self.out_att(previous_state)
            out_dists.append(pred_dist)
            att_dists.append(attr_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
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

        # previous_att_embed = normalize_sigmoid_logits(attr_dist) @ self.att_embed.weight

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = classwise_boxes_iou(boxes_for_nms, self.n_dim).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

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

        return torch.cat(out_dists, 0), out_commitments, torch.cat(att_dists, 0)


class AttributeLSTMContext(LSTMContext):
    """Modified from neural-motifs to encode contexts for each object."""

    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: ObjectClasses,
            att_classes: AttributeClasses,
            rel_classes: RelationClasses,
            in_channels: int
    ):
        super().__init__(cfg, obj_classes, rel_classes, in_channels)
        self.att_classes = att_classes
        self.num_att_classes = len(att_classes)
        self.num_attributes_cat: int = cfg.INPUT.N_ATT_CLASSES

        # Word embedding
        att_embed_vectors = obj_edge_vectors(self.att_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.att_embed1 = torch.nn.Embedding(self.num_att_classes, self.embed_dim)
        self.att_embed2 = torch.nn.Embedding(self.num_att_classes, self.embed_dim)
        with torch.no_grad():
            self.att_embed1.weight.copy_(att_embed_vectors, non_blocking=True)
            self.att_embed2.weight.copy_(att_embed_vectors, non_blocking=True)

        # Position embedding
        # Override LSTMContext.pos_embed as this one has dropout
        self.pos_embed = torch.nn.Sequential(
            torch.nn.Linear(4 * self.n_dim + 1, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
        )

        # Object & relation context
        # Override LSTMContext.obj_ctx_rnn and .decoder_rnn because of the "* 2"
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.obj_dim + self.embed_dim * 2 + 128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = _AttributeDecoderRNN(
            self.cfg, self.obj_classes, self.att_classes,
            embed_dim=self.embed_dim,
            inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim * 2 + 128,
            hidden_dim=self.hidden_dim,
            rnn_drop=self.dropout_rate
        )
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim * 2 + self.hidden_dim + self.obj_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True
        )

    def _obj_ctx(
            self,
            obj_feats: torch.Tensor,
            proposals: list[BoxList],
            obj_labels: torch.LongTensor | None = None,
            att_labels: torch.LongTensor | None = None,
            boxes_per_cls: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param proposals: BoxLists
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes_per_cls:
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

        # Decode in order
        if self.mode != SGGEvaluationMode.PredicateClassification and \
                not self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            decoder_inp = PackedSequence(torch.cat((obj_inp_rep, encoder_rep), 1),
                                         ls_transposed.cpu())
            obj_dists, obj_predictions, att_dists = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            )
            obj_predictions = obj_predictions[inv_perm]
            obj_dists = obj_dists[inv_perm]
            att_dists = att_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_predictions = obj_labels
            obj_dists = to_onehot(obj_predictions, self.num_obj_classes)
            att_dists, att_fg_ind = generate_attributes_target(att_labels, self.num_attributes_cat)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_predictions, att_dists, encoder_rep, perm, inv_perm, ls_transposed

    # noinspection PyMethodOverriding
    def _edge_ctx(
            self,
            obj_feats: torch.Tensor,
            obj_predictions: torch.Tensor,
            att_dists: torch.Tensor,
            perm: torch.Tensor,
            inv_perm: torch.Tensor,
            ls_transposed: torch.Tensor
    ) -> torch.Tensor:
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :returns: edge_ctx: [num_obj, #feats] For later!
        """
        obj_embed2 = self.obj_embed2(obj_predictions)
        att_embed2 = normalize_sigmoid_logits(att_dists) @ self.att_embed2.weight
        inp_feats = torch.cat((obj_embed2, att_embed2, obj_feats), 1)

        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed.cpu())
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def forward(
            self,
            x: torch.Tensor,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[int] | None = None,
            _: bool = False,
            __: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, AttributeLogits]:
        # Labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field(BoxList.AnnotationField.LABELS) for proposal in proposals], dim=0)
            att_labels = cat([proposal.get_field(BoxList.AnnotationField.ATTRIBUTES) for proposal in proposals], dim=0)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
            att_labels = None
        else:
            obj_labels = None
            att_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
            gt_att_labels, _ = generate_attributes_target(att_labels, self.num_attributes_cat)
            gt_att_labels = gt_att_labels / (gt_att_labels.sum(1).unsqueeze(-1) + 1e-12)
            att_embed = gt_att_labels @ self.att_embed1.weight
        else:
            obj_logits = cat([proposal.get_field(BoxList.PredictionField.PRED_LOGITS)
                              for proposal in proposals], dim=0).detach()
            att_logits = cat([proposal.get_field(BoxList.PredictionField.ATTRIBUTE_LOGITS) for proposal in proposals],
                             dim=0).detach()
            obj_embed = softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            att_embed = normalize_sigmoid_logits(att_logits) @ self.att_embed1.weight

        assert proposals[0].mode == BoxList.Mode.zyxzyx
        pos_embed = self.pos_embed(encode_box_info(proposals))
        obj_pre_rep = cat((x, obj_embed, att_embed, pos_embed), -1)

        boxes_per_cls = None
        # NMS when evaluating SceneGraphGeneration (otherwise GT boxes are used, so no need)
        if self.mode == SGGEvaluationMode.SceneGraphGeneration and not self.training and \
                not self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            # PredictionField comes from post process of box_head
            boxes_per_cls = cat([proposal.get_field(BoxList.PredictionField.BOXES_PER_CLS)
                                 for proposal in proposals], dim=0)

        # Object level contextual feature
        obj_dists, obj_predictions, att_dists, obj_ctx, perm, inv_perm, ls_transposed = self._obj_ctx(
            obj_pre_rep,
            proposals,
            obj_labels,
            att_labels,
            boxes_per_cls
        )
        # Edge level contextual feature
        obj_rel_rep = cat((x, obj_ctx), -1)
        edge_ctx = self._edge_ctx(
            obj_rel_rep,
            obj_predictions=obj_predictions,
            att_dists=att_dists,
            perm=perm,
            inv_perm=inv_perm,
            ls_transposed=ls_transposed
        )

        return obj_dists, obj_predictions, edge_ctx, att_dists
