# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from scene_graph_prediction.data import get_dataset_statistics
from scene_graph_prediction.data.datasets import DatasetStatistics
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import cat
from ._models.motifs import LSTMContext, FrequencyBias
from ._models.motifs_with_attribute import AttributeLSTMContext
from ._models.msg_passing import IMPContext
from ._models.transformer import TransformerContext
from ._models.vctree import VCTreeLSTMContext
from ._utils.relation import layer_init
from ...abstractions.attribute_head import AttributeLogits
from ...abstractions.box_head import BoxHeadFeatures, ClassLogits
from ...abstractions.loss import LossDict
from ...abstractions.relation_head import ROIRelationPredictor, RelationHeadFeatures, RelationLogits, \
    RelationHeadProposals


# noinspection DuplicatedCode
@ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(ROIRelationPredictor):
    """Note: this predictor was never tested with attribute_on..."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        # Load parameters
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_att_cls = cfg.INPUT.N_ATT_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES

        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # Load class dict
        statistics: DatasetStatistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes, att_classes = \
            statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']

        assert self.num_obj_cls == len(obj_classes)
        if self.attribute_on:
            assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        # Module construct
        self.context_layer = TransformerContext(cfg, obj_classes, rel_classes, in_channels)

        # Post decoding
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = torch.nn.Linear(self.hidden_dim * 2, pooling_dim)
        self.rel_compress = torch.nn.Linear(pooling_dim, self.num_rel_cls)
        self.ctx_compress = torch.nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # Initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, normal=False)
        layer_init(self.ctx_compress, normal=False)
        layer_init(self.post_cat, normal=False)

        if self.pooling_dim != cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE:
            self.union_single_not_match = True
            self.up_dim = torch.nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE, pooling_dim)
            layer_init(self.up_dim, normal=False)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # Load statistics into FrequencyBias to avoid loading again
            # Note: this embedding is also learnable
            self.freq_bias = FrequencyBias(statistics)

    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            roi_features: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:
        """
        :returns:
            obj_dists: logits of object label distribution
            rel_dists: logits of relation label distribution
            add_losses: additional loss terms
            att_dists: logits of attributes label distribution (if attribute_on)
        """
        obj_dists, obj_predictions, edge_ctx, att_dists = self.context_layer(roi_features, proposals)

        # Post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_predictions = obj_predictions.split(num_objs, dim=0)

        # From object level feature to pairwise relation level feature
        prod_reps = []
        pair_predictions = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_predictions):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_predictions.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_predictions, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # Use union box and mask convolution
        if self.union_single_not_match:
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # Use frequency bias
        if self.use_bias:
            rel_dists += self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
        else:
            att_dists = None

        return obj_dists, rel_dists, att_dists, [None] * len(proposals)

    def supports_attribute_refinement(self):
        return True


@ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(ROIRelationPredictor):
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        self.imp_context_layer = IMPContext(cfg, self.num_obj_cls, self.num_rel_cls, in_channels)

        # Post decoding
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE:
            self.union_single_not_match = True
            self.up_dim = torch.nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE, self.pooling_dim)
            layer_init(self.up_dim, normal=False)
        else:
            self.union_single_not_match = False

        # Freq
        if self.use_bias:
            statistics = get_dataset_statistics(cfg)
            self.freq_bias = FrequencyBias(statistics)

    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            roi_features: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # Encode context information
        obj_dists, rel_dists = self.imp_context_layer(roi_features, proposals, union_features, rel_pair_idxs)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_predictions = obj_dists.max(-1)[1]
            obj_predictions = obj_predictions.split(num_objs, dim=0)

            pair_predictions = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_predictions):
                pair_predictions.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_predictions, dim=0)

            rel_dists += self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # We use obj_predictions instead of pred from obj_dists,
        # because predictions have been through a nms stage in decoder_rnn
        return obj_dists, rel_dists, None, [None] * len(proposals)

    def supports_attribute_refinement(self):
        return False


@ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(ROIRelationPredictor):
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_att_cls = cfg.INPUT.N_ATT_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES

        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # Load class dict
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # Init contextual lstm encoding
        if self.attribute_on:
            assert self.num_att_cls == len(att_classes)
            self.lstm_context_layer = AttributeLSTMContext(cfg, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.lstm_context_layer = LSTMContext(cfg, obj_classes, rel_classes, in_channels)

        # Post decoding
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = torch.nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # Initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, normal=False)
        layer_init(self.rel_compress, normal=False)

        if self.pooling_dim != cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE:
            self.union_single_not_match = True
            self.up_dim = torch.nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE, self.pooling_dim)
            layer_init(self.up_dim, normal=False)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # Convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(statistics)

    # noinspection DuplicatedCode
    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            roi_features: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:

        # Encode context information; att_dists is None if not self.attribute_on
        obj_dists, obj_predictions, edge_ctx, att_dists = self.lstm_context_layer(roi_features, proposals)

        # Post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        sub_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        ob_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = sub_rep.split(num_objs, dim=0)
        tail_reps = ob_rep.split(num_objs, dim=0)
        obj_predictions = obj_predictions.split(num_objs, dim=0)

        prod_reps = []
        pair_predictions = []
        for pair_idx, sub_rep, ob_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_predictions):
            prod_reps.append(torch.cat((sub_rep[pair_idx[:, 0]], ob_rep[pair_idx[:, 1]]), dim=-1))
            pair_predictions.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_predictions, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            prod_rep *= self.up_dim(union_features)
        else:
            prod_rep *= union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists += self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # We use obj_predictions instead of pred from obj_dists,
        # because predictions have been through a nms stage in decoder_rnn
        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
        else:
            att_dists = None

        return obj_dists, rel_dists, att_dists, [None] * len(proposals)

    def supports_attribute_refinement(self):
        return True


@ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(ROIRelationPredictor[torch.Tensor]):
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_att_cls = cfg.INPUT.N_ATT_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES

        # Load class dict
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # Init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(cfg, obj_classes, rel_classes, statistics, in_channels)

        # Post decoding
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = torch.nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.ctx_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls)
        layer_init(self.ctx_compress, normal=False)

        # Initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, normal=False)

        if self.pooling_dim != cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE:
            self.union_single_not_match = True
            self.up_dim = torch.nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE, self.pooling_dim)
            layer_init(self.up_dim, normal=False)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(statistics)

    # noinspection DuplicatedCode
    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            roi_features: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:
        # Encode context information
        obj_dists, obj_predictions, edge_ctx, binary_predictions = (
            self.context_layer(roi_features, proposals, rel_pair_idxs)
        )  # type: ClassLogits, torch.Tensor, torch.Tensor, list[torch.Tensor]

        # Post decode
        edge_rep = torch.nn.functional.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_predictions = obj_predictions.split(num_objs, dim=0)

        prod_reps = []
        pair_predictions = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_predictions):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_predictions.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_predictions, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists

        # We use obj_predictions instead of pred from obj_dists
        # because predictions has been through a nms stage in decoder_rnn
        return obj_dists.split(num_objs, dim=0), rel_dists.split(num_rels, dim=0), None, binary_predictions

    def supports_attribute_refinement(self):
        return False

    def extra_losses(
            self,
            add_losses_required: list[torch.Tensor],
            rel_binaries: list[torch.LongTensor],
            rel_labels: list[torch.LongTensor]
    ) -> LossDict:
        """Compute extra loss terms for this predictor."""
        binary_loss = []
        binary_predictions = add_losses_required
        for bi_gt, bi_pred in zip(rel_binaries, binary_predictions):
            bi_gt = (bi_gt > 0).float()
            binary_loss.append(torch.nn.functional.binary_cross_entropy_with_logits(bi_pred, bi_gt))
        return {"binary_loss": sum(binary_loss) / len(binary_loss)}


# @ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
# class CausalAnalysisPredictor(ROIRelationPredictor):
#     def __init__(self, cfg: CfgNode, in_channels: int):
#         super().__init__(cfg, in_channels)
#         self.cfg = cfg
#         self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
#         self.spatial_for_vision = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
#         self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
#         self.num_rel_cls = cfg.INPUT.N_REL_CLASSES
#         self.fusion_type = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
#         self.use_vtranse = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
#         self.effect_type = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
#
#         # Load class dict
#         statistics = get_dataset_statistics(cfg)
#         obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
#         assert self.num_obj_cls == len(obj_classes)
#         assert self.num_rel_cls == len(rel_classes)
#
#         # Init contextual lstm encoding
#         if cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
#             # Note: the LSTMContext does not explicitly implement the same interface.
#             #       However, since the last element returned by .forward() is None, we don't have any issue here...
#             self.context_layer = LSTMContext(cfg, obj_classes, rel_classes, in_channels)
#         elif cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
#             self.context_layer = VCTreeLSTMContext(cfg, obj_classes, rel_classes, statistics, in_channels)
#         elif cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
#             self.context_layer = VTransEFeature(cfg, obj_classes, rel_classes, in_channels)
#         else:
#             raise ValueError(f"ERROR: Invalid Context Layer {cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER}")
#
#         # Post decoding
#         self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
#
#         if self.use_vtranse:
#             self.edge_dim = self.pooling_dim
#             self.post_emb = torch.nn.Linear(self.hidden_dim, self.pooling_dim * 2)
#             self.ctx_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
#         else:
#             self.edge_dim = self.hidden_dim
#             self.post_emb = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)
#             self.post_cat = torch.nn.Sequential(
#                 torch.nn.Linear(self.hidden_dim * 2, self.pooling_dim),
#                 torch.nn.ReLU(inplace=True)
#             )
#             self.ctx_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls)
#         self.vis_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls)
#
#         if self.fusion_type == 'gate':
#             self.ctx_gate_fc = torch.nn.Linear(self.pooling_dim, self.num_rel_cls)
#             layer_init(self.ctx_gate_fc, normal=False)
#
#         # Initialize layer parameters
#         layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
#         if not self.use_vtranse:
#             layer_init(self.post_cat[0], normal=False)
#             layer_init(self.ctx_compress, normal=False)
#         layer_init(self.vis_compress, normal=False)
#
#         assert self.pooling_dim == cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE
#
#         # Convey statistics into FrequencyBias to avoid loading again
#         self.freq_bias = FrequencyBias(statistics)
#
#         # Add spatial emb for visual feature
#         if self.spatial_for_vision:
#             self.spt_emb = torch.nn.Sequential(
#                 torch.nn.Linear(32, self.hidden_dim),
#                 torch.nn.ReLU(inplace=True),
#                 torch.nn.Linear(self.hidden_dim, self.pooling_dim),
#                 torch.nn.ReLU(inplace=True)
#             )
#             layer_init(self.spt_emb[0], normal=False)
#             layer_init(self.spt_emb[2], normal=False)
#
#         self.label_smooth_loss = LabelSmoothingRegression(eps=1.0)
#
#         # Untreated average features
#         self.effect_analysis = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
#         self.average_ratio = 0.0005
#
#         self.register_buffer("untreated_spt", torch.zeros(32))
#         self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
#         self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
#         self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))
#
#     # noinspection DuplicatedCode
#     def _pair_feature_generate(
#             self,
#             roi_features: BoxHeadFeatures,
#             proposals: list[BoxList],
#             rel_pair_idxs: list[torch.LongTensor],
#             num_objs: list[int],
#             obj_boxs: list[torch.Tensor],
#             ctx_average: bool = False
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
#     list[torch.Tensor] | None, torch.Tensor, torch.Tensor, list[ClassLogits]]:
#         # Encode context information
#         obj_dists, obj_predictions, edge_ctx, binary_predictions = self.context_layer(roi_features,
#                                                                                       proposals,
#                                                                                       ctx_average=ctx_average)
#         # Type hint for the linter
#         binary_predictions: list[torch.Tensor] | None = binary_predictions
#
#         obj_dist_prob = torch.nn.functional.softmax(obj_dists, dim=-1)
#
#         # Post decode
#         edge_rep: torch.Tensor = self.post_emb(edge_ctx)
#         edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
#         head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
#         tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
#
#         # Split
#         head_reps = head_rep.split(num_objs, dim=0)
#         tail_reps = tail_rep.split(num_objs, dim=0)
#         obj_predictions = obj_predictions.split(num_objs, dim=0)
#         obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
#         obj_dist_list: list[ClassLogits] = obj_dists.split(num_objs, dim=0)
#         ctx_reps = []
#         pair_predictions = []
#         pair_obj_probs = []
#         pair_bbox_info = []
#         for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in \
#                 zip(rel_pair_idxs, head_reps, tail_reps, obj_predictions, obj_boxs, obj_prob_list):
#             if self.use_vtranse:
#                 ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
#             else:
#                 ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
#
#             pair_predictions.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
#             pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
#             pair_bbox_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
#
#         pair_obj_probs = cat(pair_obj_probs, dim=0)
#         pair_bbox = cat(pair_bbox_info, dim=0)
#         pair_pred = cat(pair_predictions, dim=0)
#         ctx_rep = cat(ctx_reps, dim=0)
#
#         if self.use_vtranse:
#             post_ctx_rep = ctx_rep
#         else:
#             post_ctx_rep: torch.Tensor = self.post_cat(ctx_rep)
#
#         return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, \
#             binary_predictions, obj_dist_prob, edge_rep, obj_dist_list
#
#     # noinspection DuplicatedCode
#     def forward(
#             self,
#             proposals: RelationHeadProposals,
#             rel_pair_idxs: list[torch.LongTensor],
#             rel_labels: list[torch.Tensor] | None,
#             rel_binaries: list[torch.Tensor] | None,
#             roi_features: BoxHeadFeatures,
#             union_features: RelationHeadFeatures
#     ) -> tuple[list[ClassLogits], list[RelationLogits], LossDict, list[AttributeLogits] | None]:
#         num_rels = [r.shape[0] for r in rel_pair_idxs]
#         num_objs = [len(b) for b in proposals]
#         obj_boxs = [get_box_info(p.boxes, need_norm=True, proposal=p) for p in proposals]
#
#         assert len(num_rels) == len(num_objs)
#
#         post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_predictions, obj_dist_prob, edge_rep, obj_dist_list \
#             = self._pair_feature_generate(roi_features,
#                                           proposals,
#                                           rel_pair_idxs,
#                                           num_objs,
#                                           obj_boxs)
#
#         if not self.training and self.effect_analysis:
#             with torch.no_grad():
#                 avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self._pair_feature_generate(
#                     roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, ctx_average=True
#                 )
#
#         if self.spatial_for_vision:
#             post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)
#
#         rel_dists = self._calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
#         rel_dist_list = rel_dists.split(num_rels, dim=0)
#
#         add_losses = {}
#         # Additional loss
#         if self.training:
#             rel_labels = cat(rel_labels, dim=0)
#
#             # Binary loss for VCTree
#             if binary_predictions is not None:
#                 binary_loss = []
#                 for bi_gt, bi_pred in zip(rel_binaries, binary_predictions):
#                     bi_gt = (bi_gt > 0).float()
#                     binary_loss.append(torch.nn.functional.binary_cross_entropy_with_logits(bi_pred, bi_gt))
#                 add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
#
#             # Branch constraint: make sure each branch can predict independently
#             add_losses["auxiliary_ctx"] = torch.nn.functional.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
#             if self.fusion_type != "gate":
#                 add_losses["auxiliary_vis"] = torch.nn.functional.cross_entropy(self.vis_compress(union_features),
#                                                                                 rel_labels)
#                 add_losses["auxiliary_frq"] = torch.nn.functional.cross_entropy(
#                     self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)
#
#             # Untreated average feature
#             if self.spatial_for_vision:
#                 # noinspection PyAttributeOutsideInit
#                 self.untreated_spt = moving_average(self.untreated_spt, pair_bbox, self.average_ratio)
#             # noinspection PyAttributeOutsideInit
#             self.avg_post_ctx = moving_average(self.avg_post_ctx, post_ctx_rep, self.average_ratio)
#             # noinspection PyAttributeOutsideInit
#             self.untreated_feat = moving_average(self.untreated_feat, union_features, self.average_ratio)
#
#         elif self.effect_analysis:
#             with torch.no_grad():
#                 # Untreated spatial
#                 if self.spatial_for_vision:
#                     avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
#                 # Untreated context
#                 avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
#
#                 # Untreated visual
#                 # avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
#                 # Untreated category dist
#                 avg_frq_rep = avg_pair_obj_prob
#
#             # For effect types documentation: look at the corresponding key in defaults.py
#             if self.effect_type == 'TDE':  # TDE of CTX
#                 rel_dists = self._calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - \
#                             self._calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
#             elif self.effect_type == 'NIE':  # NIE of FRQ
#                 rel_dists = self._calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - \
#                             self._calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
#             elif self.effect_type == 'TE':  # Total Effect
#                 rel_dists = self._calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - \
#                             self._calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
#             else:
#                 assert self.effect_type == 'none'
#
#             rel_dist_list = rel_dists.split(num_rels, dim=0)
#
#         return obj_dist_list, rel_dist_list, add_losses, None
#
#     def _calculate_logits(
#             self,
#             vis_rep: torch.Tensor,
#             ctx_rep: torch.Tensor,
#             frq_rep: torch.Tensor,
#             use_label_dist: bool = True,
#             mean_ctx: bool = False
#     ) -> RelationLogits:
#         if use_label_dist:
#             frq_dists = self.freq_bias.index_with_probability(frq_rep)
#         else:
#             frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
#
#         if mean_ctx:
#             ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
#         vis_dists = self.vis_compress(vis_rep)
#         ctx_dists = self.ctx_compress(ctx_rep)
#
#         if self.fusion_type == 'gate':
#             ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
#             union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
#             # Alternatives:
#             # - Improve on zero-shot, but low mean recall and TDE recall
#             # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()
#
#             # - The best conventional Recall results
#             # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)
#
#             # - Good zero-shot Recall
#             # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()
#
#             # - Good zero-shot Recall
#             # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))
#
#             # - Balanced recall and mean recall
#             # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)
#
#             # - Good zero-shot Recall
#             # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0
#
#             # - good zero-shot Recall, bad for all the rest
#             # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())
#
#         elif self.fusion_type == "sum":
#             union_dists = vis_dists + ctx_dists + frq_dists
#         else:
#             raise ValueError(self.fusion_type)
#
#         return union_dists


def build_roi_relation_predictor(cfg: CfgNode, in_channels: int) -> ROIRelationPredictor:
    predictor = ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
