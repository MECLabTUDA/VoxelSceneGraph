# modified from https://github.com/rowanz/neural-motifs
import torch
from yacs.config import CfgNode

from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.box_head import ClassLogits
from scene_graph_prediction.modeling.abstractions.relation_head import RelationLogits, RelationHeadProposals
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.modeling.utils.build_layers import build_fc
from scene_graph_prediction.structures import BoxList
from .._utils.motifs import to_onehot


class IMPContext(torch.nn.Module):
    def __init__(
            self,
            cfg: CfgNode,
            num_obj: int,
            num_rel: int,
            in_channels: int,
            hidden_dim: int = 512,
            num_iter: int = 3
    ):
        super().__init__()
        self.cfg = cfg
        self.num_obj = num_obj
        self.num_rel = num_rel
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter

        # Mode
        self.mode = SGGEvaluationMode.build(cfg)

        self.rel_fc = build_fc(hidden_dim, self.num_rel)
        self.obj_fc = build_fc(hidden_dim, self.num_obj)

        self.obj_unary = build_fc(in_channels, hidden_dim)
        self.edge_unary = build_fc(self.pooling_dim, hidden_dim)

        self.edge_gru = torch.nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = torch.nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.sub_vert_w_fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim * 2, 1), torch.nn.Sigmoid())
        self.obj_vert_w_fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim * 2, 1), torch.nn.Sigmoid())
        self.out_edge_w_fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim * 2, 1), torch.nn.Sigmoid())
        self.in_edge_w_fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim * 2, 1), torch.nn.Sigmoid())

    # Maybe to be fixed: this class's forward method does not follow the RelationContext interface
    def forward(
            self,
            x: torch.Tensor,
            proposals: RelationHeadProposals,
            union_features: torch.Tensor,
            rel_pair_idxs: list[torch.Tensor]
    ) -> tuple[ClassLogits, RelationLogits]:
        """:returns: obj_dists, rel_dists"""
        num_objs = [len(b) for b in proposals]

        obj_rep = self.obj_unary(x)
        rel_rep = torch.torch.nn.functional.relu(self.edge_unary(union_features))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]

        # Generate sub-rel-obj mapping
        sub2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj_offset = 0
        rel_offset = 0
        sub_global_indexes = []
        obj_global_indexes = []
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
            rel_idx = torch.arange(num_rel).to(obj_rep.device).long().view(-1) + rel_offset

            sub_global_indexes.append(sub_idx)
            obj_global_indexes.append(obj_idx)

            sub2rel[sub_idx, rel_idx] = 1.0
            obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_indexes = torch.cat(sub_global_indexes, dim=0)
        obj_global_indexes = torch.cat(obj_global_indexes, dim=0)

        # Iterative message passing
        hx_obj = torch.zeros(obj_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()
        hx_rel = torch.zeros(rel_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.num_iter):
            # Compute edge context
            sub_vert = vert_factor[i][sub_global_indexes]
            obj_vert = vert_factor[i][obj_global_indexes]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[i]
            vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        if self.mode == SGGEvaluationMode.PredicateClassification:
            # noinspection PyTypeChecker
            obj_labels: torch.LongTensor = cat([proposal.get_field(BoxList.AnnotationField.LABELS)
                                                for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.num_obj)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION:
            obj_labels = cat([proposal.PRED_LABELS for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.num_obj)
        else:
            obj_dists = self.obj_fc(vert_factor[-1])

        rel_dists = self.rel_fc(edge_factor[-1])

        return obj_dists, rel_dists
