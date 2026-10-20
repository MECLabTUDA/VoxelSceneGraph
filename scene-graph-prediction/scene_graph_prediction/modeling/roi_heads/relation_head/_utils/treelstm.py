import torch

from .motifs import get_dropout_mask
from .relation import block_orthogonal
from .vctree import BinaryTree, ArbitraryTree


class TreeLSTMContainer:
    def __init__(
            self,
            hidden_tensor: torch.FloatTensor | None,
            order_tensor: torch.LongTensor,
            order_count: int,
            dists_tensor: torch.FloatTensor | None,
            commitments_tensor: torch.Tensor | None,
            dropout_mask: torch.Tensor
    ):
        self.hidden = hidden_tensor  # Float tensor [num_obj, self.out_dim]
        self.order = order_tensor  # Long tensor [num_obj]
        self.order_count = order_count  # int
        self.dists = dists_tensor  # Float tensor [num_obj, len(self.classes)]
        self.commitments = commitments_tensor
        self.dropout_mask = dropout_mask


class MultiLayerBTreeLSTM(torch.nn.Module):
    """
    Multilayer Bidirectional Tree LSTM.
    Each layer contains one forward lstm (leaves to root) and one backward lstm (root to leaves).
    """

    def __init__(self, in_dim: int, out_dim: int, num_layer: int, dropout: float = 0.0):
        super().__init__()
        self.num_layer = num_layer
        layers = [_BidirectionalTreeLSTM(in_dim, out_dim, dropout)]
        for i in range(num_layer - 1):
            layers.append(_BidirectionalTreeLSTM(out_dim, out_dim, dropout))
        self.multi_layer_lstm = torch.nn.ModuleList(layers)

    def forward(self, tree: BinaryTree, features: torch.Tensor, num_obj: int) -> torch.Tensor:
        for i in range(self.num_layer):
            features = self.multi_layer_lstm[i](tree, features, num_obj)
        return features


class _BidirectionalTreeLSTM(torch.nn.Module):
    """
    Bidirectional Tree LSTM:
    Contains one forward lstm (leaves to root) and one backward lstm (root to leaves).
    Dropout mask will be generated one time for all trees in the forest, to make sure the consistency.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        self.treeLSTM_forward = OneDirectionalTreeLSTM(in_dim, out_dim // 2, True, dropout)
        self.treeLSTM_backward = OneDirectionalTreeLSTM(in_dim, out_dim // 2, False, dropout)

    def forward(self, tree: BinaryTree, features: torch.Tensor, num_obj: int):
        forward_output = self.treeLSTM_forward(tree, features, num_obj)
        backward_output = self.treeLSTM_backward(tree, features, num_obj)

        final_output = torch.cat((forward_output, backward_output), 1)

        return final_output


class OneDirectionalTreeLSTM(torch.nn.Module):
    """One Way Tree LSTM."""

    def __init__(self, in_dim: int, out_dim: int, is_forward: bool, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        if is_forward:
            self.treeLSTM = BiTreeLSTMForward(in_dim, out_dim)
        else:
            self.treeLSTM = BiTreeLSTMBackward(in_dim, out_dim)

    def forward(self, tree: BinaryTree, features: torch.Tensor, num_obj: int):
        # Calc dropout mask, same for all
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, (1, self.out_dim), features.device)
        else:
            dropout_mask = None

        # Tree lstm input, used to resume order
        # noinspection PyTypeChecker
        h_order: torch.LongTensor = torch.tensor([0] * num_obj, device=features.device, dtype=torch.int64)
        lstm_io = TreeLSTMContainer(None, h_order, 0, None, None, dropout_mask)
        # Run tree lstm forward (leaves to root)
        self.treeLSTM(tree, features, lstm_io)
        # Resume order to the same as input
        output = lstm_io.hidden[lstm_io.order.long()]
        return output


class BiTreeLSTMForward(torch.nn.Module):
    """From leaves to root."""

    def __init__(
            self,
            feat_dim: int,
            h_dim: int,
            is_pass_embed: bool = False,
            embed_layer: torch.nn.Module | None = None,
            embed_out_layer: torch.nn.Module | None = None
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer

        self.px = torch.nn.Linear(self.feat_dim, self.h_dim)
        self.ioffu_x = torch.nn.Linear(self.feat_dim, 6 * self.h_dim)
        self.ioffu_h_left = torch.nn.Linear(self.h_dim, 6 * self.h_dim)
        self.ioffu_h_right = torch.nn.Linear(self.h_dim, 6 * self.h_dim)

        # Initialization
        with torch.no_grad():
            block_orthogonal(self.px.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffu_x.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffu_h_left.weight, [self.h_dim, self.h_dim])
            block_orthogonal(self.ioffu_h_right.weight, [self.h_dim, self.h_dim])

            self.px.bias.fill_(0.0)
            self.ioffu_x.bias.fill_(0.0)
            self.ioffu_h_left.bias.fill_(0.0)
            self.ioffu_h_right.bias.fill_(0.0)
            # Initialize forget gate biases to 1.0 as per An Empirical
            # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
            self.ioffu_h_left.bias[2 * self.h_dim:4 * self.h_dim].fill_(0.5)
            self.ioffu_h_right.bias[2 * self.h_dim:4 * self.h_dim].fill_(0.5)

    # noinspection DuplicatedCode
    def _node_forward(
            self,
            feat_inp: torch.Tensor,
            left_c: torch.Tensor,
            right_c: torch.Tensor,
            left_h: torch.Tensor,
            right_h: torch.Tensor,
            dropout_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_x = self.px(feat_inp)
        ioffu = self.ioffu_x(feat_inp) + self.ioffu_h_left(left_h) + self.ioffu_h_right(right_h)
        i, o, f_l, f_r, u, r = torch.split(ioffu, ioffu.size(1) // 6, dim=1)
        i, o, f_l, f_r, u, r = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(f_l), torch.sigmoid(f_r), torch.tanh(
            u), torch.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f_l, left_c) + torch.mul(f_r, right_c)
        h = torch.mul(o, torch.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0, and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    # noinspection DuplicatedCode
    def forward(self, tree: BinaryTree, features: torch.Tensor, tree_lstm_io: TreeLSTMContainer):
        """
        Updates tree_lstm_io with relevant information for convenience.
        :param tree: The root for a tree
        :param features: [num_obj, feature_size]
        :param tree_lstm_io: .hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
                             .order: init as 0 for all [num_obj], update for recovering original order
        """
        # Recursively search child
        if tree.left_child is not None:
            self.forward(tree.left_child, features, tree_lstm_io)
        if tree.right_child is not None:
            self.forward(tree.right_child, features, tree_lstm_io)

        # Get c,h from left child
        if tree.left_child is None:
            left_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            left_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            # Only being used in decoder network
            if self.is_pass_embed:
                left_embed = self.embed_layer.weight[0]
        else:
            left_c = tree.left_child.state_c
            left_h = tree.left_child.state_h
            # Only being used in decoder network
            if self.is_pass_embed:
                left_embed = tree.left_child.embedded_label
        # Get c,h from right child
        if tree.right_child is None:
            right_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            right_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            # Only being used in decoder network
            if self.is_pass_embed:
                right_embed = self.embed_layer.weight[0]
        else:
            right_c = tree.right_child.state_c
            right_h = tree.right_child.state_h
            # Only being used in decoder network
            if self.is_pass_embed:
                right_embed = tree.right_child.embedded_label

        # Only being used in decoder network
        if self.is_pass_embed:
            # noinspection PyUnboundLocalVariable
            # because variables will be bound if self.is_pass_embed
            next_feature = torch.cat(
                (features[tree.index].view(1, -1), left_embed.view(1, -1), right_embed.view(1, -1)), 1)
        else:
            next_feature = features[tree.index].view(1, -1)

        c, h = self._node_forward(next_feature, left_c, right_c, left_h, right_h, tree_lstm_io.dropout_mask)
        tree.state_c = c
        tree.state_h = h
        # Record label prediction
        # Only being used in decoder network
        if self.is_pass_embed:
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, tree_lstm_io, self.training)

        # Record hidden state
        if tree_lstm_io.hidden is None:
            tree_lstm_io.hidden = h.view(1, -1)
        else:
            tree_lstm_io.hidden = torch.cat((tree_lstm_io.hidden, h.view(1, -1)), 0)

        tree_lstm_io.order[tree.index] = tree_lstm_io.order_count
        tree_lstm_io.order_count += 1


class BiTreeLSTMBackward(torch.nn.Module):
    """From root to leaves."""

    def __init__(
            self,
            feat_dim: int,
            h_dim: int,
            is_pass_embed: bool = False,
            embed_layer: torch.nn.Module | None = None,
            embed_out_layer: torch.nn.Module | None = None
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.is_pass_embed = is_pass_embed
        self.embed_layer = embed_layer
        self.embed_out_layer = embed_out_layer

        self.px = torch.nn.Linear(self.feat_dim, self.h_dim)
        self.ioffu_x = torch.nn.Linear(self.feat_dim, 5 * self.h_dim)
        self.ioffu_h = torch.nn.Linear(self.h_dim, 5 * self.h_dim)

        # Initialization
        with torch.no_grad():
            block_orthogonal(self.px.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffu_x.weight, [self.h_dim, self.feat_dim])
            block_orthogonal(self.ioffu_h.weight, [self.h_dim, self.h_dim])

            self.px.bias.fill_(0.0)
            self.ioffu_x.bias.fill_(0.0)
            self.ioffu_h.bias.fill_(0.0)
            # Initialize forget gate biases to 1.0 as per An Empirical
            # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
            self.ioffu_h.bias[2 * self.h_dim:3 * self.h_dim].fill_(1.0)

    # noinspection DuplicatedCode
    def _node_backward(
            self,
            feat_inp: torch.Tensor,
            root_c: torch.Tensor,
            root_h: torch.Tensor,
            dropout_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        projected_x = self.px(feat_inp)
        ioffu = self.ioffu_x(feat_inp) + self.ioffu_h(root_h)
        i, o, f, u, r = torch.split(ioffu, ioffu.size(1) // 5, dim=1)
        i, o, f, u, r = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(f), torch.tanh(u), torch.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f, root_c)
        h = torch.mul(o, torch.tanh(c))
        h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0, and we are in training mode.
        if dropout_mask is not None and self.training:
            h_final = torch.mul(h_final, dropout_mask)
        return c, h_final

    # noinspection DuplicatedCode
    def forward(self, tree: BinaryTree, features, tree_lstm_io):
        """
        Updates tree_lstm_io with relevant information for convenience.
        :param tree: The root for a tree
        :param features: [num_obj, feature_size]
        :param tree_lstm_io: .hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
                             .order: init as 0 for all [num_obj], update for recovering original order
        """

        if tree.parent is None:
            root_c = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            root_h = torch.tensor([0.0] * self.h_dim, device=features.device).float().view(1, -1)
            if self.is_pass_embed:
                root_embed = self.embed_layer.weight[0]
        else:
            root_c = tree.parent.state_c_backward
            root_h = tree.parent.state_h_backward
            if self.is_pass_embed:
                root_embed = tree.parent.embedded_label

        if self.is_pass_embed:
            # noinspection PyUnboundLocalVariable
            # because this variable will be bound if self.is_pass_embed
            next_features = torch.cat((features[tree.index].view(1, -1), root_embed.view(1, -1)), 1)
        else:
            next_features = features[tree.index].view(1, -1)

        c, h = self._node_backward(next_features, root_c, root_h, tree_lstm_io.dropout_mask)
        tree.state_c_backward = c
        tree.state_h_backward = h
        # Record label prediction
        # Only being used in decoder network
        if self.is_pass_embed:
            pass_embed_postprocess(h, self.embed_out_layer, self.embed_layer, tree, tree_lstm_io, self.training)

        # record hidden state
        if tree_lstm_io.hidden is None:
            tree_lstm_io.hidden = h.view(1, -1)
        else:
            tree_lstm_io.hidden = torch.cat((tree_lstm_io.hidden, h.view(1, -1)), 0)

        tree_lstm_io.order[tree.index] = tree_lstm_io.order_count
        tree_lstm_io.order_count += 1

        # Recursively update from root to leaves
        if tree.left_child is not None:
            self.forward(tree.left_child, features, tree_lstm_io)
        if tree.right_child is not None:
            self.forward(tree.right_child, features, tree_lstm_io)


def pass_embed_postprocess(
        h: torch.Tensor,
        embed_out_layer: torch.nn.Module,
        embed_layer: torch.nn.Module,
        tree: BinaryTree | ArbitraryTree,
        tree_lstm_io: TreeLSTMContainer,
        is_training: bool
):
    """Calculate distribution and predict/sample labels. Updates tree_lstm_io."""
    pred_dist = embed_out_layer(h)
    label_to_embed = torch.nn.functional.softmax(pred_dist.view(-1), 0)[1:].max(0)[1] + 1
    if is_training:
        sampled_label = torch.nn.functional.softmax(pred_dist.view(-1), 0)[1:].multinomial(1).detach() + 1
        tree.embedded_label = embed_layer(sampled_label + 1)
    else:
        tree.embedded_label = embed_layer(label_to_embed + 1)

    if tree_lstm_io.dists is None:
        tree_lstm_io.dists = pred_dist.view(1, -1)
    else:
        tree_lstm_io.dists = torch.cat((tree_lstm_io.dists, pred_dist.view(1, -1)), 0)

    if tree_lstm_io.commitments is None:
        tree_lstm_io.commitments = label_to_embed.view(-1)
    else:
        tree_lstm_io.commitments = torch.cat((tree_lstm_io.commitments, label_to_embed.view(-1)), 0)
