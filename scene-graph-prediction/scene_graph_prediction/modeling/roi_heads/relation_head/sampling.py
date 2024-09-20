# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy.random as npr
import torch
from yacs.config import CfgNode

from scene_graph_prediction.data import get_dataset_statistics, DatasetStatistics
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
            require_overlap: bool,
            num_sample_per_gt_rel: int,
            use_gt_box: bool,
            statistics: DatasetStatistics | None = None
    ):
        super().__init__()
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.use_gt_box = use_gt_box
        # Get the count of fg relations given a pair of (sub lbl, ob lbl)
        self.fg_matrix = statistics["fg_matrix"][1:, 1:, 1:].sum(-1) if statistics is not None else None
        self.statistics = statistics

    def prepare_test_pairs(self, proposals: list[BoxList]) -> list[torch.LongTensor]:
        """Prepare test pairs for evaluation by computing possible relation pairs."""
        # Prepare object pairs for relation prediction
        rel_pair_idxs = []
        for proposal in proposals:
            rel_possibility = self._compute_relation_possibilities(proposal)
            idxs = torch.nonzero(rel_possibility).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                # If there is no candidate pairs, give a placeholder
                rel_pair_idxs.append(torch.zeros((0, 2), dtype=torch.int64, device=proposal.boxes.device))
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

            device = proposal.boxes.device
            num_prp = proposal.boxes.shape[0]

            # Compute relation pairs from the GT matrix
            tgt_rel_matrix = target.RELATIONS  # [tgt, tgt]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)

            # Compute a list of indexes for subject, objects and relation classes
            tgt_subj_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_obj_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labels = tgt_rel_matrix[tgt_subj_idxs, tgt_obj_idxs].contiguous().view(-1)

            # Compute binary matrix mask with objects having a relation
            rel_possibility = torch.zeros((num_prp, num_prp), device=device, dtype=torch.int64)
            rel_possibility[tgt_subj_idxs, tgt_obj_idxs] = 1
            rel_possibility[tgt_obj_idxs, tgt_subj_idxs] = 1
            rel_sym_binaries.append(rel_possibility)

            # Compute binary matrix mask with objects having no relation (i.e. background relation)
            # I.e. objects that could possibly have one given their classes, but don't in this case
            rel_impossibility = self._compute_relation_possibilities(proposal)
            rel_impossibility[tgt_subj_idxs, tgt_obj_idxs] = 0
            # noinspection PyTypeChecker
            tgt_bg_idxs = torch.nonzero(rel_impossibility > 0)

            # Concatenate all pairs and labels for sampling
            all_rel_pair_idxs = torch.cat([tgt_pair_idxs, tgt_bg_idxs])
            all_rel_labels = torch.cat(
                [tgt_rel_labels.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device, dtype=torch.int64)]
            ).contiguous().view(-1)

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
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binaries = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            prp_lab = proposal.LABELS.long()
            tgt_lab = target.LABELS.long()
            tgt_rel_matrix = target.RELATIONS  # [tgt, tgt]

            # IoU matching between predictions GT based on labels and overlap
            ious = BoxListOps.iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious >= self.fg_thres)  # [tgt, prp]

            # Proposal self IoU to filter non-overlap
            rel_possibility = self._compute_relation_possibilities(proposal)

            # Remove pairs with at least an object that is not matched to a GT object
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self._motif_rel_fg_bg_sampling(
                proposal.boxes.device,
                tgt_rel_matrix,
                ious,
                is_match,
                rel_possibility
            )
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binaries.append(binary_rel)  # (n_pred_i, n_pred_i)

        return rel_labels, rel_idx_pairs, rel_sym_binaries

    def _compute_relation_possibilities(self, proposal: BoxList) -> torch.ByteTensor:
        """Compute a binary matrix mask for possible foreground relations between predicted objects."""
        # All relations considered except self
        num_prp = proposal.boxes.shape[0]
        rel_possibility = (
                torch.ones((num_prp, num_prp), device=proposal.boxes.device, dtype=torch.uint8) -
                torch.eye(num_prp, device=proposal.boxes.device, dtype=torch.uint8)
        )

        # Remove pairs with no overlap between subject and object
        if not self.use_gt_box and self.require_overlap:
            rel_possibility &= BoxListOps.iou(proposal, proposal).gt(0)

        # Note: see REQUIRE_REL_IN_TRAIN
        if self.fg_matrix is not None:
            self.fg_matrix = self.fg_matrix.to(rel_possibility.device)
            # We have some statistics on the training data,
            # so we can enforce that only (sub lbl, ob lbl) pairs found in the training data can have a relation
            labels = proposal.PRED_LABELS \
                if proposal.has_field(BoxList.PredictionField.PRED_LABELS) \
                else proposal.LABELS

            rel_pair_idxs = torch.nonzero(rel_possibility)
            # Note: to learn sub/ob ordering, also allow pairs where the reverse ordering is found in the groundtruth
            # Note: careful about order of sub and ob in idxs returned by torch.non_zero
            impossible_idxs = self.fg_matrix[
                                  labels[rel_pair_idxs[:, 0]].long() - 1,
                                  labels[rel_pair_idxs[:, 1]].long() - 1
                              ] + self.fg_matrix[
                                  labels[rel_pair_idxs[:, 1]].long() - 1,
                                  labels[rel_pair_idxs[:, 0]].long() - 1
                              ] == 0
            rel_possibility[rel_pair_idxs[impossible_idxs][:, 0], rel_pair_idxs[impossible_idxs][:, 1]] = 0

        return rel_possibility

    # noinspection DuplicatedCode
    def _motif_rel_fg_bg_sampling(
            self,
            device: torch.device,
            tgt_rel_matrix: torch.Tensor,
            ious: torch.Tensor,
            is_match: torch.Tensor,
            rel_possibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare to sample foreground relation triplets and background relation triplets.
        Here we do:
        1. Use the groundtruth to compute an array of relation indexes and labels.
        2. We want to convert that to a batch of relations (i.e. prepare a batch for classification). For each GT rel:
          2.1. Find all combinations of pairs from the predictions given is_match.
          2.2. Remove reflexive relations.
          2.3. Update the binary symmetric binary_rel matrix.
          2.4. Mark these pairs as "taken" such that we do not use them as negative examples.
          2.5. Generate all possible triplets given these pairs.
          2.6. Keep only the maximum number of triplets allowed based on the IoU of the matches.
        3. Select up to num_pos_per_img foreground relations.
        3. Fill the batch with background relations.
        4. Concatenate.

        :param device:
        :param tgt_rel_matrix:  [number_target, number_target] groundtruth relation matrix
        :param ious:            [number_target, num_proposal] iou between targets and predictions
        :param is_match:        [number_target, num_proposal] whether a target got matched to a predicted box
        :param rel_possibility: [num_proposal, num_proposal] whether a foreground relation can be considered
        :returns:
            Nx3 relation batch, (sub, ob, rel)
            binary matrix indicating whether two objects are related (there is a relation in either direction)
              Note: only used for training.
        """
        # Compute a list of indexes for subject, objects and relation classes
        # noinspection PyTypeChecker
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        tgt_subj_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_obj_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labels = tgt_rel_matrix[tgt_subj_idxs, tgt_obj_idxs].contiguous().view(-1)

        # Iterate over GT relations triplets
        num_prp = is_match.shape[-1]
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
        fg_rel_triplets = []  # subj, obj, lbl
        for subj_idx, obj_idx, lbl in zip(tgt_subj_idxs, tgt_obj_idxs, tgt_rel_labels):
            if lbl < 0:
                # Skip ignored cases
                continue

            # Find matching pair in proposals (might be more than one)
            bi_match_subj = torch.nonzero(is_match[subj_idx])
            bi_match_obj = torch.nonzero(is_match[obj_idx])

            num_bi_subj = bi_match_subj.shape[0]
            num_bi_obj = bi_match_obj.shape[0]
            if num_bi_subj == 0 or num_bi_obj == 0:
                continue

            # All combination pairs
            bi_match_subj = bi_match_subj.view(1, num_bi_subj).expand(num_bi_obj, num_bi_subj).contiguous()
            bi_match_obj = bi_match_obj.view(num_bi_obj, 1).expand(num_bi_obj, num_bi_subj).contiguous()
            # Check that we do not have a reflexive relation
            # noinspection PyTypeChecker
            valid_pair: torch.Tensor = bi_match_subj != bi_match_obj
            if valid_pair.sum().item() <= 0:
                continue
            bi_match_subj = bi_match_subj[valid_pair]
            bi_match_obj = bi_match_obj[valid_pair]

            # Binary relations only consider whether they are related or not, so it's symmetric
            binary_rel[bi_match_subj.view(-1), bi_match_obj.view(-1)] = 1
            binary_rel[bi_match_obj.view(-1), bi_match_subj.view(-1)] = 1

            # Remove selected pairs from rel_possibility
            # Note: we do this to identify sensible negative cases
            rel_possibility[bi_match_subj, bi_match_obj] = 0

            # Construct corresponding proposal triplets corresponding to this gt relation
            fg_labels = torch.full((bi_match_obj.shape[0], 1), lbl, dtype=torch.int64, device=device)
            fg_triplet_i = cat((bi_match_subj.view(-1, 1), bi_match_obj.view(-1, 1), fg_labels), dim=-1).long()

            # Select if too many corresponding proposal pairs to one pair of gt relationship triplet
            if fg_triplet_i.shape[0] > self.num_sample_per_gt_rel:
                # Note: that in original motif, the selection is based on the ious_score
                ious_score = (
                        ious[subj_idx, bi_match_subj] * ious[obj_idx, bi_match_obj]
                ).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                # Note: TIL (23.01.2024) that you can index tensors with numpy arrays
                fg_triplet_i = fg_triplet_i[perm]

            fg_rel_triplets.append(fg_triplet_i)

        # Concatenate fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).long()

        # Concatenate background relations
        # Note: these are possible relations that did not match any relation in the GT
        bg_rel_indexes = torch.nonzero(rel_possibility).view(-1, 2)
        bg_rel_labels = torch.zeros(bg_rel_indexes.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_indexes, bg_rel_labels.view(-1, 1)), dim=-1).long()

        # If both fg and bg are empty
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            return torch.zeros((0, 3), dtype=torch.int64, device=device), binary_rel

        # Concatenate all and sample
        return torch.cat([fg_rel_triplets, bg_rel_triplets]), binary_rel


def build_roi_relation_samp_processor(cfg: CfgNode) -> RelationSampling:
    if cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_REL_IN_TRAIN:
        assert cfg.MODEL.ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION, \
            f"If REQUIRE_REL_IN_TRAIN is True, then you must set DISABLE_RECLASSIFICATION to True."

    return RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        get_dataset_statistics(cfg) if cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_REL_IN_TRAIN else None
    )
