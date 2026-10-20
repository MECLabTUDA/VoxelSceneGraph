from functools import reduce
from typing import Optional

import numpy as np
import torch

from scene_graph_prediction.data.evaluation import SGGEvaluationMode
from scene_graph_prediction.modeling.abstractions.relation_head import RelationHeadProposals
from scene_graph_prediction.structures import BoxList, BoxListOps
from scene_graph_prediction.utils.logger import setup_logger

_logger = setup_logger("modeling.roi_heads.relation_head._utils.vctree", save_dir="", distributed_rank=0)


class BinaryTree:
    def __init__(
            self,
            idx: int,
            node_score: float,
            label: int,
            box: torch.Tensor | None,
            is_root: bool = False
    ):
        self.index = int(idx)
        self.is_root = is_root
        self.left_child: Optional["BinaryTree"] = None
        self.right_child: Optional["BinaryTree"] = None
        self.parent: Optional["BinaryTree"] = None
        self.num_children = 0
        self.state_c: torch.Tensor | None = None
        self.state_h: torch.Tensor | None = None
        self.state_c_backward: torch.Tensor | None = None
        self.state_h_backward: torch.Tensor | None = None
        # Used to select node
        self.node_score = float(node_score)
        self.label = label
        self.embedded_label = None
        self.box: torch.Tensor = box.view(-1)  # y1,x1,y2,x2

    def add_left_child(self, child: "BinaryTree"):
        if self.left_child is not None:
            _logger.warning('Left child already exist')
            return
        child.parent = self
        self.num_children += 1
        self.left_child = child

    def add_right_child(self, child: "BinaryTree"):
        if self.right_child is not None:
            _logger.warning('Right child already exist')
            return
        child.parent = self
        self.num_children += 1
        self.right_child = child

    def depth(self) -> int:
        if self.parent is None:
            return 1
        return self.parent.depth() + 1


class ArbitraryTree:
    def __init__(
            self,
            idx: int,
            score: float,
            label: int,
            box: torch.Tensor,
            is_root: bool = False
    ):
        self.index = int(idx)
        self.is_root = is_root
        self.score = float(score)
        self.children: list["ArbitraryTree"] = []
        self.label = label
        self.embedded_label = None
        self.box = box.view(-1)  # zyxzyx mode
        self.parent = None
        self.node_order = -1  # the n_th node added to the tree

    def generate_bi_tree(self) -> BinaryTree:
        """Converts this node ONLY to a BiTree. Children are lost."""
        return BinaryTree(self.index, self.score, self.label, self.box, self.is_root)

    def add_child(self, child: "ArbitraryTree"):
        child.parent = self
        self.children.append(child)

    def get_children_count(self) -> int:
        return len(self.children)


ArbitraryForest = list[ArbitraryTree]
BinaryForest = list[BinaryTree]


def generate_forest(
        pair_scores: list[torch.Tensor],
        proposals: RelationHeadProposals,
        mode: SGGEvaluationMode
) -> ArbitraryForest:
    """
    Generate a list of trees that covers all the objects in a batch
      proposals.bbox: [obj_num, "zyxzyx"]
      pair_scores: [obj_num, obj_num]
    output: list of trees, each present a chunk of overlapping objects
    """
    output_forest = []  # The list of trees, each one is a chunk of overlapping objects

    for pair_score, proposal in zip(pair_scores, proposals):
        num_obj = pair_score.shape[0]
        if mode == SGGEvaluationMode.PredicateClassification:
            obj_label = proposal.get_field(BoxList.AnnotationField.LABELS)
        else:
            obj_label = proposal.get_field(BoxList.PredictionField.PRED_LOGITS).max(-1)[1]

        assert pair_score.shape[0] == len(proposal)
        assert pair_score.shape[0] == pair_score.shape[1]
        node_scores = pair_score.mean(1).view(-1)
        root_idx = int(node_scores.max(-1)[1])

        root = ArbitraryTree(root_idx,
                             float(node_scores[root_idx]),
                             int(obj_label[root_idx]),
                             proposal.boxes[root_idx],
                             is_root=True)

        nodes_to_process = []
        remain_index = []
        # Put all nodes into node container
        for idx in list(range(num_obj)):
            if idx == root_idx:
                continue
            new_node = ArbitraryTree(idx, float(node_scores[idx]), int(obj_label[idx]), proposal.boxes[idx])
            nodes_to_process.append(new_node)
            remain_index.append(idx)

        # Iteratively generate tree
        _gen_tree(nodes_to_process, pair_score, root, remain_index)
        output_forest.append(root)

    return output_forest


def _gen_tree(
        nodes_to_process: ArbitraryForest,
        pair_scores: torch.Tensor,
        # node_scores: torch.Tensor,
        root: ArbitraryTree,
        remain_index: list[int]
):
    """
    From a list of tree nodes, adds parent/child relations:
    Step 1: Divide all nodes into left child container and right child container
    Step 2: From left child container and right child container, select their respective sub roots

    :param pair_scores: [obj_num, obj_num]
    # :param node_scores: [obj_num]
    """
    # Step 0
    if len(nodes_to_process) == 0:
        return

    # Step 1
    device = pair_scores.device
    select_node = []
    select_index = []
    select_node.append(root)
    select_index.append(root.index)

    while len(nodes_to_process) > 0:
        wid = len(remain_index)
        select_indexes = torch.tensor(select_index, device=device, dtype=torch.int64)
        remain_indexes = torch.tensor(remain_index, device=device, dtype=torch.int64)
        select_score_map = pair_scores[select_indexes][:, remain_indexes].view(-1)
        best_id = select_score_map.max(0)[1]

        depend_id = int(best_id) // wid
        insert_id = int(best_id) % wid
        best_depend_node = select_node[depend_id]
        best_insert_node = nodes_to_process[insert_id]
        best_depend_node.add_child(best_insert_node)

        select_node.append(best_insert_node)
        select_index.append(best_insert_node.index)
        nodes_to_process.remove(best_insert_node)
        remain_index.remove(best_insert_node.index)


def arbitrary_forest_to_binary_forest(forest: ArbitraryForest) -> BinaryForest:
    output = []
    for i in range(len(forest)):
        result_tree = _arbitrary_tree_to_binary_tree(forest[i])
        output.append(result_tree)
    return output


def _arbitrary_tree_to_binary_tree(tree: ArbitraryTree) -> BinaryTree:
    """Kick-off of the recursion."""
    root_node = tree.generate_bi_tree()
    _arbitrary_node_to_binary_node(tree, root_node)
    return root_node


def _arbitrary_node_to_binary_node(arbitrary_node: ArbitraryTree, binary_node: BinaryTree):
    """Recursive conversion."""
    if arbitrary_node.get_children_count() >= 1:
        new_bi_node = arbitrary_node.children[0].generate_bi_tree()
        binary_node.add_left_child(new_bi_node)
        _arbitrary_node_to_binary_node(arbitrary_node.children[0], binary_node.left_child)

    if arbitrary_node.get_children_count() > 1:
        current_bi_node = binary_node.left_child
        for i in range(arbitrary_node.get_children_count() - 1):
            new_bi_node = arbitrary_node.children[i + 1].generate_bi_tree()
            current_bi_node.add_right_child(new_bi_node)
            current_bi_node = current_bi_node.right_child
            _arbitrary_node_to_binary_node(arbitrary_node.children[i + 1], current_bi_node)


def bbox_intersection(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.LongTensor:
    a = box_a.size(0)
    b = box_b.size(0)
    n_dim = box_a.shape[1] // 2
    max_zyx = torch.min(box_a[:, n_dim:].unsqueeze(1).expand(a, b, n_dim),
                        box_b[:, n_dim:].unsqueeze(0).expand(a, b, n_dim))
    min_zyx = torch.max(box_a[:, :n_dim].unsqueeze(1).expand(a, b, n_dim),
                        box_b[:, :n_dim].unsqueeze(0).expand(a, b, n_dim))
    inter = torch.clamp((max_zyx - min_zyx + 1.0), min=0)
    # return inter[:, :, 0] * inter[:, :, 1]
    return reduce(lambda a, b: a * b, [inter[..., dim] for dim in range(n_dim)])


def bbox_overlap(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.FloatTensor:
    n_dim = box_a.shape[1] // 2
    inter = bbox_intersection(box_a, box_b)
    # noinspection PyUnresolvedReferences
    area_a: torch.Tensor = np.prod([box_a[:, n_dim + dim] - box_a[:, dim] + 1.0 for dim in range(n_dim)]) \
        .unsqueeze(1).expand_as(inter)  # [A,B]
    # noinspection PyUnresolvedReferences
    area_b: torch.Tensor = np.prod([box_b[:, n_dim + dim] - box_b[:, dim] + 1.0 for dim in range(n_dim)]) \
        .unsqueeze(1).expand_as(inter)  # [A,B]
    # area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
    #           (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    # area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
    #           (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / (union + 1e-9)


def get_overlap_info(proposals: list[BoxList]) -> torch.Tensor:
    assert proposals and proposals[0].mode == BoxList.Mode.zyxzyx
    overlap_info = []
    im_vol = np.prod(proposals[0].size)

    for proposal in proposals:
        boxes = proposal.boxes
        intersection = bbox_intersection(boxes, boxes).float()  # num, num
        overlap = BoxListOps.iou(proposal, proposal)

        area = proposal.volume()

        info1 = (intersection > 0.0).float().sum(1).view(-1, 1)  # Whether there is an intersection
        info2 = intersection.sum(1).view(-1, 1) / im_vol  # Ratio of intersection
        info3 = overlap.sum(1).view(-1, 1)  # Volume of overlap
        info4 = info2 / (info1 + 1e-9)  # Ratio of intersection when there is one, otherwise +inf?
        info5 = info3 / (info1 + 1e-9)  # Volume of overlap when there is an intersection, otherwise +inf?
        info6 = area / im_vol  # Ratio of area to image volume

        info = torch.cat([info1, info2, info3, info4, info5, info6[:, None]], dim=1)
        overlap_info.append(info)

    return torch.cat(overlap_info, dim=0)
