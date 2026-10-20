# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from scene_graph_prediction.data import get_dataset_statistics
from scene_graph_prediction.data.datasets import DatasetStatistics
from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.registries import *
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList
from ._models.motifs import LSTMContext, FrequencyBias
from ._models.motifs_with_attribute import AttributeLSTMContext
from ._models.msg_passing import IMPContext
from ._models.transformer import TransformerContext
from ._models.vctree import VCTreeLSTMContext
from ._utils.motifs import to_onehot
from ._utils.relation import layer_init
from ...abstractions.attribute_head import AttributeLogits
from ...abstractions.box_head import BoxHeadFeatures, ClassLogits
from ...abstractions.loss import LossDict
from ...abstractions.relation_head import ROIRelationPredictor, RelationHeadFeatures, RelationLogits, \
    RelationHeadProposals
from ...utils.build_layers import build_fc


@ROI_RELATION_PREDICTOR.register("NoRefinementPredictorICH")
class NoRefinementPredictorICH(ROIRelationPredictor):
    """No object or attribute refinement. Only predict relations given union features with a fully connected layer."""

    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        assert cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.mode = SGGEvaluationMode.build(cfg)

        # ICH: twice the output
        self.predictor = build_fc(cfg.MODEL.ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE, self.num_rel_cls * 2)

        # Freq
        if self.use_bias:
            statistics = get_dataset_statistics(cfg)
            self.freq_bias = FrequencyBias(statistics)

    def forward(
            self,
            proposals: RelationHeadProposals,
            rel_pair_idxs: list[torch.Tensor],
            _: BoxHeadFeatures,
            union_features: RelationHeadFeatures
    ) -> tuple[list[ClassLogits], list[RelationLogits], list[AttributeLogits] | None, list]:

        # Compute relation logits
        rel_dists = self.predictor(union_features)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        # Sanity check on the number of images
        assert len(num_rels) == len(num_objs)

        # Assemble object predictions and dists for all modes (from object detector label predictions)
        if self.mode == SGGEvaluationMode.PredicateClassification:
            # noinspection PyTypeChecker
            obj_predictions: torch.LongTensor = cat([proposal.get_field(BoxList.AnnotationField.LABELS)
                                                     for proposal in proposals], dim=0)
        else:
            # noinspection PyTypeChecker
            obj_predictions = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
        obj_dists = to_onehot(obj_predictions, self.num_obj_cls)

        # Add frequency bias if enabled
        if self.use_bias:
            # ICH: twice bias computed separately
            pair_predictions_ventr = []
            pair_predictions_midl = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_predictions):
                pair_predictions_ventr.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
                pair_predictions_midl.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 2]]), dim=1))
            rel_dists[:self.num_rel_cls] += self.freq_bias.index_with_labels(cat(pair_predictions_ventr, dim=0).long())
            rel_dists[self.num_rel_cls:] += self.freq_bias.index_with_labels(cat(pair_predictions_midl, dim=0).long())

        # Split relation logits (from batch to #rels/img)
        rel_dists = rel_dists.split(num_rels, dim=0)
        # noinspection PyTypeChecker
        return obj_dists, rel_dists, None, [None] * len(proposals)

    def supports_attribute_refinement(self):
        return False


@ROI_RELATION_PREDICTOR.register("IMPPredictorICH")
class IMPPredictorICH(ROIRelationPredictor):
    def __init__(self, cfg: CfgNode, in_channels: int):
        super().__init__(cfg, in_channels)
        self.num_obj_cls = cfg.INPUT.N_OBJ_CLASSES
        self.num_rel_cls = cfg.INPUT.N_REL_CLASSES
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # ICH: twice the number of classes
        self.imp_context_layer = IMPContext(cfg, self.num_obj_cls, self.num_rel_cls * 2, in_channels)

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

            # ICH: twice bias computed separately
            pair_predictions_ventr = []
            pair_predictions_midl = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_predictions):
                pair_predictions_ventr.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
                pair_predictions_midl.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 2]]), dim=1))
            rel_dists[:self.num_rel_cls] += self.freq_bias.index_with_labels(cat(pair_predictions_ventr, dim=0).long())
            rel_dists[self.num_rel_cls:] += self.freq_bias.index_with_labels(cat(pair_predictions_midl, dim=0).long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # We use obj_predictions instead of pred from obj_dists,
        # because predictions have been through a nms stage in decoder_rnn
        return obj_dists, rel_dists, None, [None] * len(proposals)

    def supports_attribute_refinement(self):
        return False


@ROI_RELATION_PREDICTOR.register("MotifPredictorICH")
class MotifPredictorICH(ROIRelationPredictor):
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
            # ICH: twice the number of classes
            self.lstm_context_layer = AttributeLSTMContext(cfg, obj_classes, att_classes, rel_classes * 2, in_channels)
        else:
            self.lstm_context_layer = LSTMContext(cfg, obj_classes, rel_classes * 2, in_channels)

        # Post decoding
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = torch.nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        # ICH: twice the number of classes
        self.rel_compress = torch.nn.Linear(self.pooling_dim, self.num_rel_cls * 2, bias=True)

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
        for pair_idx, sub_rep, ob_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_predictions):
            prod_reps.append(torch.cat((sub_rep[pair_idx[:, 0]], ob_rep[pair_idx[:, 1]]), dim=-1))
        prod_rep = cat(prod_reps, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            prod_rep *= self.up_dim(union_features)
        else:
            prod_rep *= union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            # ICH: twice bias computed separately
            pair_predictions_ventr = []
            pair_predictions_midl = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_predictions):
                pair_predictions_ventr.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
                pair_predictions_midl.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 2]]), dim=1))
            rel_dists[:self.num_rel_cls] += self.freq_bias.index_with_labels(cat(pair_predictions_ventr, dim=0).long())
            rel_dists[self.num_rel_cls:] += self.freq_bias.index_with_labels(cat(pair_predictions_midl, dim=0).long())

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


def build_roi_relation_predictor(cfg: CfgNode, in_channels: int) -> ROIRelationPredictor:
    predictor = ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return predictor(cfg, in_channels)
