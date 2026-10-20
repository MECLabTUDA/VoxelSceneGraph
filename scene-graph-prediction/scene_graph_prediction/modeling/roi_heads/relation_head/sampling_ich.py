# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from yacs.config import CfgNode

from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposals
from scene_graph_prediction.modeling.abstractions.relation_head import RelationHeadTargets
from scene_graph_prediction.modeling.utils import cat
from scene_graph_prediction.structures import BoxList, BoxListOps


class RelationSampling:
    """
    Note: here we do not use any hard batch size to restrict the number of (sub, ob) pairs sampled.
          We assume that it;s better to sample when computing the loss.
          However, if REQUIRE_REL_IN_TRAIN is False, this might lead to a lot of relations being processed.
    """

    def __init__(
            self,
            fg_thres: float,  # IoU thr
            use_gt_box: bool
    ):
        super().__init__()
        self.fg_thres = fg_thres
        self.use_gt_box = use_gt_box

    @staticmethod
    def _do_compute_pairs(labels: torch.LongTensor) -> torch.LongTensor:
        # Triplets instead of pairs, but who cares? It will be handled in the post-processing
        ventricle_idx = torch.where(labels == 2)[0]
        midline_idx = torch.where(labels == 3)[0]
        ich_idxs = torch.where(labels == 1)[0]
        n_ich = ich_idxs.numel()
        # noinspection PyTypeChecker
        return torch.cat([
            ich_idxs.view(-1, 1),
            torch.tile(ventricle_idx, (n_ich,)).view(-1, 1),
            torch.tile(midline_idx, (n_ich,)).view(-1, 1),
        ], dim=1)

    def prepare_test_pairs(self, proposals: list[BoxList]) -> list[torch.LongTensor]:
        """Prepare test pairs for evaluation by computing possible relation pairs."""
        # Prepare object pairs for relation prediction
        rel_pair_idxs = []
        for proposal in proposals:
            rel_pair_idxs.append(self._do_compute_pairs(proposal.PRED_LABELS))
        return rel_pair_idxs

    # noinspection DuplicatedCode
    def gtbox_relation_sample(
            self,
            proposals: BoxHeadTestProposals,
            targets: RelationHeadTargets
    ) -> tuple[list[torch.LongTensor], list[torch.LongTensor], list[torch.LongTensor]]:
        """
        Method called when sampling from groundtruth annotation.
        Here we do:
        1. Compute foreground relations pairs.
        2. Compute background relations pairs.
        3. Ensure batch siwe and positive fraction if we have too many foreground relations.
        4. Fill the rest of the batch with negative cases.
        5. Generate the list of indexes and labels used for training.

        :param proposals: only used to check that proposal boxes are target boxes.
        :param targets: contain fields: RELATIONS
        """
        assert self.use_gt_box
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binaries = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            assert proposal.boxes.shape[0] == target.boxes.shape[0]

            all_rel_pair_idxs = self._do_compute_pairs(proposal.LABELS)
            ich_idxs = all_rel_pair_idxs[:, 0].contiguous().view(-1)
            ventr_idxs = all_rel_pair_idxs[:, 1].contiguous().view(-1)
            midl_idxs = all_rel_pair_idxs[:, 2].contiguous().view(-1)

            device = proposal.boxes.device
            num_prp = proposal.boxes.shape[0]

            # Compute binary matrix mask with objects having a relation (becomes quite useless but eh)
            rel_possibility = torch.zeros((num_prp, num_prp), device=device, dtype=torch.int64)
            rel_possibility[ich_idxs, ventr_idxs] = 1
            rel_possibility[ventr_idxs, ich_idxs] = 1
            rel_possibility[ich_idxs, midl_idxs] = 1
            rel_possibility[midl_idxs, ich_idxs] = 1
            rel_sym_binaries.append(rel_possibility)

            # Here we store the relation labels for both objects (ventricles and midline)
            tgt_rel_matrix = target.RELATIONS  # [tgt, tgt]
            all_rel_labels = torch.cat([
                tgt_rel_matrix[ich_idxs, ventr_idxs],
                tgt_rel_matrix[ich_idxs, midl_idxs],
            ], dim=1)

            rel_idx_pairs.append(all_rel_pair_idxs)
            rel_labels.append(all_rel_labels)

        return rel_labels, rel_idx_pairs, rel_sym_binaries

    def detect_relation_sample(
            self,
            proposals: BoxHeadTestProposals,
            targets: RelationHeadTargets
    ) -> tuple[list[torch.Tensor], list[torch.LongTensor], list[torch.Tensor]]:
        """
        The input proposals are already processed by subsample function of the detector.
        Here we do:
        1. Match predicted objects to groundtruth ones.
        2. Remove pairs with at least an object that is not matched to a groundtruth object.
        3. Filter possible relations based on overlap or prior knowledge.
        4. Sample a batch of fore- and background relation triplets.

        Note: corresponds to rel_assignments function in neural-motifs

        :param proposals: contain fields LABELS, and PRED_LOGITS
        :param targets: contain fields: LABELS
        """
        # FIXME does it work that easily?
        # TODO No. Also set negative labels for FP ICH?
        return self.gtbox_relation_sample(proposals, targets)


def build_roi_relation_samp_processor(cfg: CfgNode) -> RelationSampling:
    return RelationSampling(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX)
