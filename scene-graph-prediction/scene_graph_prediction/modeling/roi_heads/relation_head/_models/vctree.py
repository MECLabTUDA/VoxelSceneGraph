# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn.functional import softmax, relu
from yacs.config import CfgNode

from scene_graph_prediction.data.datasets import DatasetStatistics, ObjectClasses, RelationClasses
from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadFeatures, ClassLogits
from scene_graph_prediction.modeling.abstractions.relation_head import RelationContext, RelationHeadProposals
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList
from .._utils.motifs import obj_edge_vectors, to_onehot, get_dropout_mask, \
    encode_box_info
from .._utils.relation import layer_init
from .._utils.treelstm import TreeLSTMContainer, MultiLayerBTreeLSTM, BiTreeLSTMForward
from .._utils.vctree import generate_forest, arbitrary_forest_to_binary_forest, \
    get_overlap_info, BinaryTree, BinaryForest


class _DecoderTreeLSTM(torch.nn.Module):
    def __init__(
            self,
            cfg: CfgNode,
            object_classes: ObjectClasses,
            embed_dim: int,
            inputs_dim: int,
            hidden_dim: int,
            is_forward: bool = False,
            dropout: float = 0.2
    ):
        """
        :param embed_dim: Dimension of the embeddings
        :param hidden_dim: Hidden dim of the decoder
        """
        super().__init__()
        self.classes = object_classes
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.5
        self.dropout = dropout
        # Generate embedding layer
        embed_vectors = obj_edge_vectors(["start"] + self.classes, wv_dir=cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = torch.nn.Embedding(len(self.classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vectors, non_blocking=True)
        # Generate out layer
        self.out = torch.nn.Linear(self.hidden_size, len(self.classes))

        if not is_forward:
            self.input_size = inputs_dim + embed_dim
        else:
            self.input_size = inputs_dim + embed_dim * 2
        self.decoderLSTM = BiTreeLSTMForward(self.input_size, self.hidden_size, is_pass_embed=True,
                                             embed_layer=self.obj_embed, embed_out_layer=self.out)

    def forward(
            self,
            tree: BinaryTree,
            features: torch.Tensor,
            num_obj: int
    ) -> tuple[torch.FloatTensor, torch.Tensor]:
        # Generate dropout
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, (1, self.hidden_size), features.device)
        else:
            dropout_mask = None

        # Generate tree lstm input/output class
        # noinspection PyTypeChecker
        h_order: torch.LongTensor = torch.tensor([0] * num_obj, device=features.device)
        lstm_io = TreeLSTMContainer(None, h_order, 0, None, None, dropout_mask)

        self.decoderLSTM(tree, features, lstm_io)

        out_dists = lstm_io.dists[lstm_io.order.long()]
        out_commitments = lstm_io.commitments[lstm_io.order.long()]

        return out_dists, out_commitments


class VCTreeLSTMContext(RelationContext):
    """Modified from neural-motifs to encode contexts for each object."""

    def __init__(
            self,
            cfg: CfgNode,
            obj_classes: ObjectClasses,
            rel_classes: RelationClasses,
            statistics: DatasetStatistics,
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

        # Overlap embedding
        self.overlap_embed = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.BatchNorm1d(128, momentum=0.001),
            torch.nn.ReLU(inplace=True)
        )

        # Box embed
        self.box_embed = torch.nn.Sequential(
            torch.nn.Linear(4 * self.n_dim + 1, 128),
            torch.nn.BatchNorm1d(128, momentum=0.001),
            torch.nn.ReLU(inplace=True)
        )

        # Object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # VCTree
        co_occurrence = statistics["pred_dist"].float().sum(-1)
        assert co_occurrence.shape[0] == co_occurrence.shape[-1]
        assert len(co_occurrence.shape) == 2
        self.bi_freq_prior = torch.nn.Linear(self.num_obj_classes * self.num_obj_classes, 1, bias=False)

        with torch.no_grad():
            co_occurrence = co_occurrence + co_occurrence.transpose(0, 1)
            self.bi_freq_prior.weight.copy_(co_occurrence.view(-1).unsqueeze(0), non_blocking=True)

        self.obj_reduce = torch.nn.Linear(self.obj_dim, 128)
        self.emb_reduce = torch.nn.Linear(self.embed_dim, 128)
        self.score_pre = torch.nn.Linear(128 * 4, self.hidden_dim)
        self.score_sub = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.score_obj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.vision_prior = torch.nn.Linear(self.hidden_dim * 3 + 1, 1)

        layer_init(self.obj_reduce, normal=False)
        layer_init(self.emb_reduce, normal=False)
        layer_init(self.score_pre, normal=False)
        layer_init(self.score_sub, normal=False)
        layer_init(self.score_obj, normal=False)

        self.obj_ctx_rnn = MultiLayerBTreeLSTM(
            in_dim=self.obj_dim + self.embed_dim + 128,
            out_dim=self.hidden_dim,
            num_layer=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0
        )
        self.decoder_rnn = _DecoderTreeLSTM(
            cfg,
            self.obj_classes, embed_dim=self.embed_dim,
            inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim + 128,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_rate
        )

        self.edge_ctx_rnn = MultiLayerBTreeLSTM(
            in_dim=self.embed_dim + self.hidden_dim + self.obj_dim,
            out_dim=self.hidden_dim,
            num_layer=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0
        )

        # Untreated average features
        self.average_ratio = 0.0005

        # T O D O reimplement causal analysis
        # self.effect_analysis = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        # if self.effect_analysis:
        #     self.register_buffer("untreated_dcd_feat",
        #                          torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim + 128))
        #     self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
        #     self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.obj_dim))

    def _obj_ctx(
            self,
            num_objs: list[int],
            obj_feats: torch.Tensor,
            proposals: list[BoxList],
            obj_labels: torch.LongTensor | None = None,
            vc_forest: BinaryForest | None = None,
            ctx_average: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param proposals: BoxLists
        :param obj_labels: [num_obj] the GT labels of the image
        :returns:
            obj_dists: [num_obj, #classes] new probability distribution.
            obj_predictions: argmax of that distribution.
            obj_final_ctx: [num_obj, #feats] For later!
        """
        obj_feats = obj_feats.split(num_objs, dim=0)
        obj_labels = obj_labels.split(num_objs, dim=0) if obj_labels is not None else None

        obj_contexts = []
        obj_predictions = []
        obj_dists = []
        for i, (feat, tree, proposal) in enumerate(zip(obj_feats, vc_forest, proposals)):
            encod_rep = self.obj_ctx_rnn(tree, feat, len(proposal))
            obj_contexts.append(encod_rep)
            # Decode in order
            if self.mode != SGGEvaluationMode.PredicateClassification:
                # T O D O reimplement causal analysis
                # if not self.training and self.effect_analysis and ctx_average:
                #     decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(encod_rep.shape[0], -1)
                # else:
                decoder_inp = torch.cat((feat, encod_rep), 1)
                # if self.training and self.effect_analysis:
                #     self.untreated_dcd_feat = moving_average(self.untreated_dcd_feat, decoder_inp, self.average_ratio)
                obj_dist, obj_pred = self.decoder_rnn(tree, decoder_inp, len(proposal))
            else:
                assert obj_labels is not None
                obj_pred = obj_labels[i]
                obj_dist = to_onehot(obj_pred, self.num_obj_classes)
            obj_predictions.append(obj_pred)
            obj_dists.append(obj_dist)

        obj_contexts = cat(obj_contexts, dim=0)
        obj_predictions = cat(obj_predictions, dim=0)
        obj_dists = cat(obj_dists, dim=0)
        return obj_contexts, obj_predictions, obj_dists

    def _edge_ctx(
            self,
            num_objs: list[int],
            obj_feats: torch.Tensor,
            forest: BinaryForest
    ) -> torch.Tensor:
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        inp_feats = obj_feats.split(num_objs, dim=0)

        edge_contexts = []
        for feat, tree, num_obj in zip(inp_feats, forest, num_objs):
            edge_rep = self.edge_ctx_rnn(tree, feat, num_obj)
            edge_contexts.append(edge_rep)
        edge_contexts = cat(edge_contexts, dim=0)

        return edge_contexts

    def forward(
            self,
            x: BoxHeadFeatures,
            proposals: RelationHeadProposals,
            all_average: bool = False,
            ctx_average: bool = False
    ) -> tuple[ClassLogits, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        num_objs = [len(b) for b in proposals]
        # Labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            # noinspection PyTypeChecker
            obj_labels: torch.LongTensor = cat([proposal.LABELS for proposal in proposals], dim=0)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
        else:
            # noinspection PyTypeChecker
            obj_labels: torch.LongTensor = torch.empty(0, dtype=torch.int64)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
            obj_logits = to_onehot(obj_labels, self.num_obj_classes)
        else:
            obj_logits = cat([proposal.get_field(BoxList.PredictionField.PRED_LOGITS)
                              for proposal in proposals],
                             dim=0).detach()
            obj_embed = softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == BoxList.Mode.zyxzyx
        box_info = encode_box_info(proposals)
        pos_embed = self.pos_embed(box_info)

        batch_size = x.shape[0]
        # T O D O reimplement causal analysis
        # if all_average and self.effect_analysis and not self.training:
        #     obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        # else:
        obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        # Construct VCTree
        box_inp = self.box_embed(box_info)
        pair_inp = self.overlap_embed(get_overlap_info(proposals))
        bi_inp = cat((self.obj_reduce(x.detach()), self.emb_reduce(obj_embed.detach()), box_inp, pair_inp), -1)
        bi_predictions, vc_scores = self._vctree_score_net(num_objs, bi_inp, obj_logits)
        forest = generate_forest(vc_scores, proposals, self.mode)
        vc_forest = arbitrary_forest_to_binary_forest(forest)

        # Object level contextual feature
        obj_contexts, obj_predictions, obj_dists = self._obj_ctx(
            num_objs,
            obj_pre_rep,
            proposals,
            obj_labels,
            vc_forest,
            ctx_average=ctx_average
        )
        # Edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_predictions.long())

        # T O D O reimplement causal analysis
        # if (all_average or ctx_average) and self.effect_analysis and (not self.training):
        #     obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_contexts), dim=-1)
        # else:
        obj_rel_rep = cat((obj_embed2, x, obj_contexts), -1)

        edge_ctx = self._edge_ctx(num_objs, obj_rel_rep, vc_forest)

        # T O D O reimplement causal analysis
        # # Memorize average feature
        # if self.training and self.effect_analysis:
        #     # noinspection PyAttributeOutsideInit
        #     self.untreated_obj_feat = moving_average(self.untreated_obj_feat, obj_pre_rep, self.average_ratio)
        #     # noinspection PyAttributeOutsideInit
        #     self.untreated_edg_feat = moving_average(
        #         self.untreated_edg_feat,
        #         cat((obj_embed2, x), -1),
        #         self.average_ratio
        #     )

        return obj_dists, obj_predictions, edge_ctx, bi_predictions

    def _vctree_score_net(
            self,
            num_objs: list[int],
            roi_feat: torch.Tensor,
            roi_dist: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        roi_dist = roi_dist.detach()
        roi_dist = softmax(roi_dist, dim=-1)
        # Separate into each image
        roi_feat = relu(self.score_pre(roi_feat))
        sub_feat = relu(self.score_sub(roi_feat))
        obj_feat = relu(self.score_obj(roi_feat))

        sub_feats = sub_feat.split(num_objs, dim=0)
        obj_feats = obj_feat.split(num_objs, dim=0)
        roi_dists = roi_dist.split(num_objs, dim=0)

        bi_predictions = []
        vc_scores = []
        for sub, obj, dist in zip(sub_feats, obj_feats, roi_dists):
            # only used to calculate loss
            num_obj = sub.shape[0]
            num_dim = sub.shape[-1]
            sub = sub.view(1, num_obj, num_dim).expand(num_obj, num_obj, num_dim)
            obj = obj.view(num_obj, 1, num_dim).expand(num_obj, num_obj, num_dim)
            sub_dist = dist.view(1, num_obj, -1).expand(num_obj, num_obj, -1).unsqueeze(2)
            obj_dist = dist.view(num_obj, 1, -1).expand(num_obj, num_obj, -1).unsqueeze(3)
            joint_dist = (sub_dist * obj_dist).view(num_obj, num_obj, -1)

            co_prior = self.bi_freq_prior(joint_dist.view(num_obj * num_obj, -1)).view(num_obj, num_obj)
            vis_prior = self.vision_prior(
                cat([sub * obj, sub, obj, co_prior.unsqueeze(-1)], dim=-1).view(num_obj * num_obj, -1)) \
                .view(num_obj, num_obj)
            joint_pred = torch.sigmoid(vis_prior) * co_prior

            bi_predictions.append(joint_pred)
            vc_scores.append(torch.sigmoid(joint_pred))

        return bi_predictions, vc_scores
