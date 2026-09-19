"""Relation-aware heterogeneous graph neural network (simplified HGT).

This module implements a lightweight Heterogeneous Graph Transformer: for every
relation (edge) type we learn a dedicated linear projection of the source-node
features and a dedicated attention vector, then aggregate the attention-weighted
messages of the different relations separately before combining them.

The implementation is self-contained and only relies on pure PyTorch plus the
``softmax`` / ``scatter`` helpers provided by ``torch_geometric.utils`` (no
``torch_scatter``/``torch_sparse`` dependency is required).

The graph is represented with a single node type ``"node"`` and ``num_relations``
edge types whose adjacency lists live in a dict ``edge_index_dict`` mapping the
relation id (as a string) to a ``(2, E)`` LongTensor of (source, destination)
indices.  A dict of per-edge masks ``edge_mask_dict`` may be supplied so that the
same frozen backbone can be reused by the counterfactual explainer.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter, softmax


class HeteroGNNLayer(nn.Module):
    """One relation-aware message-passing layer (simplified HGT).

    For each relation ``r`` we project the source features with ``W_r`` and score
    every edge with an additive attention vector ``a_r`` over the concatenation
    ``[W_r x_src ; W_r x_dst]``.  The attention weights are normalised with a
    softmax over the neighbours of each destination node, and the messages are
    summed per destination across all relations.  A residual self-projection,
    layer norm and ReLU complete the update:

        x' = ReLU(LayerNorm(W_self x + sum_r attention_r(W_r x))).

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Output (hidden) feature dimension.
        num_relations: Number of distinct relation (edge) types.
        dropout: Dropout probability applied to messages (default ``0.0``).
        use_rel_weights: Whether to apply the learnable per-relation attention
            scalars (``False`` fixes every relation weight at 1, ablation E4e).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        dropout: float = 0.0,
        use_rel_weights: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.use_rel_weights = use_rel_weights

        # Per-relation source projection: maps source node features to hidden.
        self.rel_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim, bias=False) for r in range(num_relations)}
        )
        # Per-relation additive attention vector over [W_r x_src ; W_r x_dst].
        self.att = nn.ParameterDict(
            {str(r): nn.Parameter(torch.empty(2 * hidden_dim)) for r in range(num_relations)}
        )
        # Learnable per-relation scalar weight (relation attention): lets the
        # model up-weight informative relations and down-weight dense/noisy ones
        # instead of summing every relation with equal weight.
        self.rel_weight = nn.Parameter(torch.ones(num_relations))
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise linear weights and attention vectors."""
        gain = nn.init.calculate_gain("relu")
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.rel_proj[str(r)].weight, gain=gain)
            nn.init.xavier_uniform_(self.att[str(r)].view(1, -1), gain=gain)
        nn.init.xavier_uniform_(self.self_proj.weight, gain=gain)
        nn.init.zeros_(self.self_proj.bias)

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        edge_mask_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Run one message-passing step.

        Args:
            x: Node features of shape ``(N, in_dim)``.
            edge_index_dict: Relation id (str) -> ``(2, E_r)`` edge index tensor.
            edge_mask_dict: Optional per-edge keep masks, relation id (str) ->
                ``(E_r,)`` tensor in ``[0, 1]`` (``1`` = keep edge).  When given,
                every edge message is multiplied by its mask so that masked-out
                edges are removed from the computation graph.

        Returns:
            Updated node features of shape ``(N, hidden_dim)``.
        """
        device = x.device
        num_nodes = x.size(0)
        agg = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)

        for r in range(self.num_relations):
            key = str(r)
            edge_index = edge_index_dict[key]  # (2, E_r)
            src, dst = edge_index[0], edge_index[1]

            h_src = self.rel_proj[key](x[src])  # (E_r, hidden)
            h_dst = self.rel_proj[key](x[dst])  # (E_r, hidden)

            # Additive attention score per edge, normalised over neighbours of dst.
            e = torch.sum(torch.cat([h_src, h_dst], dim=-1) * self.att[key], dim=-1)
            e = F.leaky_relu(e, negative_slope=0.2)
            alpha = softmax(e, dst)  # (E_r,) softmax probabilities

            msg = self.dropout(h_src) * alpha.unsqueeze(-1)  # (E_r, hidden)
            if edge_mask_dict is not None:
                msg = msg * edge_mask_dict[key].unsqueeze(-1)

            weight = self.rel_weight[r] if self.use_rel_weights else torch.tensor(1.0, device=device, dtype=x.dtype)
            agg = agg + weight * scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")

        out = self.self_proj(x) + agg
        out = self.norm(out)
        return self.act(out)


class HeteroGNN(nn.Module):
    """Relation-aware heterogeneous GNN (simplified HGT) with a classifier head.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden dimension used inside every layer.
        out_dim: Number of output classes.
        num_layers: Number of stacked :class:`HeteroGNNLayer` blocks.
        num_relations: Number of distinct relation (edge) types.
        dropout: Dropout probability passed to every layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_relations: int,
        dropout: float = 0.0,
        use_rel_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                HeteroGNNLayer(layer_in, hidden_dim, num_relations, dropout=dropout, use_rel_weights=use_rel_weights)
            )
        self.classifier = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        edge_mask_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Compute class logits for every node.

        Args:
            x: Node features of shape ``(N, in_dim)``.
            edge_index_dict: Relation id (str) -> ``(2, E_r)`` edge index tensor.
            edge_mask_dict: Optional per-edge keep masks passed to every layer.

        Returns:
            Logits of shape ``(N, out_dim)``.
        """
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return self.classifier(h)

    def embed(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        edge_mask_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Return the final hidden embeddings (before the classifier head)."""
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return h
