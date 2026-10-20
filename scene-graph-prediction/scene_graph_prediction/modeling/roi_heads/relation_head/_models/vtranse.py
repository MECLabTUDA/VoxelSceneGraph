# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn.functional import softmax, relu
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import ObjectClasses, RelationClasses
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.modeling.utils.build_layers import build_fc
from scene_graph_prediction.structures import BoxList
from .._utils import moving_average
from .._utils.motifs import obj_edge_vectors, encode_box_info
from ....abstractions.box_head import BoxHeadFeatures
from ....abstractions.relation_head import RelationContext, RelationHeadProposals


class VTransEFeature(RelationContext):
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
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

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
            torch.nn.ReLU(inplace=True),
        )

        # Object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.pred_layer = build_fc(self.obj_dim + self.embed_dim + 128, self.num_obj_classes)
        self.fc_layer = build_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        # Untreated average features
        self.average_ratio = 0.0005
        # Effect analysis handles direct and indirect effect (?)
        self.effect_analysis = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.obj_dim + 128))

    # noinspection DuplicatedCode
    def forward(
            self,
            x: BoxHeadFeatures,
            proposals: RelationHeadProposals,
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        # Labels will be used in DecoderRNN during training
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.AnnotationField.LABELS for proposal in proposals], dim=0)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field(BoxList.PredictionField.PRED_LOGITS)
                              for proposal in proposals], dim=0).detach()
            obj_embed = softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == BoxList.Mode.zyxzyx
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if (all_average or ctx_average) and self.effect_analysis and not self.training:
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        # Object level contextual feature
        obj_dists = self.pred_layer(obj_pre_rep)
        obj_predictions = obj_dists.max(-1)[1]
        # Edge level contextual feature

        if (all_average or ctx_average) and self.effect_analysis and not self.training:
            obj_embed2 = softmax(obj_dists, dim=1) @ self.obj_embed2.weight
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_embed2), dim=-1)
        else:
            obj_embed2 = self.obj_embed2(obj_predictions.long())
            obj_rel_rep = cat((x, pos_embed, obj_embed2), -1)

        edge_ctx = relu(self.fc_layer(obj_rel_rep))

        # Memorize average feature
        if self.training and self.effect_analysis:
            # Members are registered buffers
            # noinspection PyAttributeOutsideInit
            self.untreated_obj_feat = moving_average(self.untreated_obj_feat, obj_pre_rep, self.average_ratio)
            # noinspection PyAttributeOutsideInit
            self.untreated_edg_feat = moving_average(self.untreated_edg_feat,
                                                     cat((x, pos_embed), -1),
                                                     self.average_ratio)

        return obj_dists, obj_predictions, edge_ctx, None
