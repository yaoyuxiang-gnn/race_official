"""Established heterogeneous backbones (HAN / HGT) with edge-mask support.

Single-node-type, multi-relation implementations of

* **HAN** — Heterogeneous Graph Attention Network (Wang et al., WWW 2019):
  node-level attention per meta-path (relation) followed by semantic-level
  attention over relations;
* **HGT** — Heterogeneous Graph Transformer (Hu et al., WWW 2020):
  per-relation Q/K/V typed attention with relation-specific ``mu``, message
  aggregation, and per-relation target (A-)projection.

Both expose the same interface as :class:`model.hetero_gnn.HeteroGNN`::

    logits = model(x, edge_index_dict, edge_mask_dict=None)

so the frozen backbone can be reused unchanged by the counterfactual
explainers (``CounterfactualExplainer`` / ``RelationTypeExplainer`` /
``optimize_edge_masks``).  Edge masks multiply the *post-softmax* attention
weights, exactly as in :class:`model.hetero_gnn.HeteroGNNLayer`, keeping the
mask semantics of the explainer consistent across backbones.  A residual
connection, layer norm and ReLU complete every layer so that all backbones
share the same training protocol (dropout, weight decay, early stopping).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter, softmax

__all__ = ["HAN", "HGT", "HANLayer", "HGTLayer"]


class HANLayer(nn.Module):
    """One HAN block: per-relation node-level attention + semantic attention.

    For every relation (meta-path) ``r`` a shared type transformation ``W_r``
    projects the node features, and per-head additive attention vectors score
    each edge; the attention-weighted messages of relation ``r`` produce the
    meta-path embedding ``z_r``.  A semantic-level attention then combines the
    meta-path embeddings per node with a learned query ``q``:

        beta_r = softmax_r( q^T tanh(W_sem z_r + b) ),
        x'     = ReLU(LayerNorm(W_self x + sum_r beta_r z_r)).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Node-level attention: per-relation (meta-path) transformation W_r and
        # per-head additive attention vector a_r over [W_r x_src ; W_r x_dst].
        self.rel_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim, bias=False) for r in range(num_relations)}
        )
        self.att = nn.ParameterDict(
            {str(r): nn.Parameter(torch.empty(num_heads, 2 * self.head_dim)) for r in range(num_relations)}
        )

        # Semantic-level attention over the K meta-path embeddings.
        self.sem_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sem_b = nn.Parameter(torch.zeros(hidden_dim))
        self.sem_q = nn.Parameter(torch.empty(1, hidden_dim))

        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.rel_proj[str(r)].weight, gain=gain)
            nn.init.xavier_uniform_(self.att[str(r)].view(1, -1), gain=gain)
        nn.init.xavier_uniform_(self.sem_W.weight, gain=gain)
        nn.init.xavier_uniform_(self.sem_q, gain=gain)
        nn.init.zeros_(self.sem_b)
        nn.init.xavier_uniform_(self.self_proj.weight, gain=gain)
        nn.init.zeros_(self.self_proj.bias)

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        edge_mask_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        device = x.device
        num_nodes = x.size(0)
        H, dh = self.num_heads, self.head_dim
        z_all: Dict[str, Tensor] = {}

        for r in range(self.num_relations):
            key = str(r)
            edge_index = edge_index_dict[key]
            src, dst = edge_index[0], edge_index[1]

            h = self.rel_proj[key](x)                 # (N, hidden)
            h_src = h[src].view(-1, H, dh)            # (E, H, dh)
            h_dst = h[dst].view(-1, H, dh)
            e = torch.sum(torch.cat([h_src, h_dst], dim=-1) * self.att[key], dim=-1)  # (E, H)
            e = F.leaky_relu(e, negative_slope=0.2)
            alpha = softmax(e, dst, num_nodes=num_nodes)  # (E, H)

            msg = h_src * alpha.unsqueeze(-1)         # (E, H, dh)
            if edge_mask_dict is not None:
                msg = msg * edge_mask_dict[key].view(-1, 1, 1)
            msg = self.dropout(msg)
            z_all[key] = scatter(msg.reshape(-1, H * dh), dst, dim=0,
                                 dim_size=num_nodes, reduce="sum")  # (N, hidden)

        # Semantic-level attention over relation embeddings, per node.
        Z = torch.stack([z_all[str(r)] for r in range(self.num_relations)], dim=0)  # (K, N, hidden)
        u = torch.tanh(self.sem_W(Z) + self.sem_b)    # (K, N, hidden)
        scores = torch.sum(u * self.sem_q, dim=-1)    # (K, N)
        beta = torch.softmax(scores, dim=0)           # (K, N)
        z = torch.sum(beta.unsqueeze(-1) * Z, dim=0)  # (N, hidden)

        out = self.self_proj(x) + z
        out = self.norm(out)
        return self.act(out)


class HGTLayer(nn.Module):
    """One HGT block: per-relation typed attention (Q/K/V + mu) and A-projection.

    For relation ``r`` the destination nodes provide queries, the source nodes
    keys and values, with a relation-specific additive bias ``mu_r``:

        attn_r = softmax( (Q_r x_dst . K_r x_src) / sqrt(dh) + mu_r ),
        x'     = ReLU(LayerNorm(W_self x + sum_r A_r(attn_r * V_r x_src))).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim) for r in range(num_relations)}
        )
        self.k_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim) for r in range(num_relations)}
        )
        self.v_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim) for r in range(num_relations)}
        )
        self.mu = nn.ParameterDict(
            {str(r): nn.Parameter(torch.zeros(1)) for r in range(num_relations)}
        )
        # Per-relation target projection (HGT's A-Linear).
        self.a_proj = nn.ModuleDict(
            {str(r): nn.Linear(hidden_dim, hidden_dim) for r in range(num_relations)}
        )

        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        for r in range(self.num_relations):
            key = str(r)
            for mod in (self.q_proj[key], self.k_proj[key], self.v_proj[key], self.a_proj[key]):
                nn.init.xavier_uniform_(mod.weight, gain=gain)
                nn.init.zeros_(mod.bias)
        nn.init.xavier_uniform_(self.self_proj.weight, gain=gain)
        nn.init.zeros_(self.self_proj.bias)

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        edge_mask_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        device = x.device
        num_nodes = x.size(0)
        H, dh = self.num_heads, self.head_dim
        agg = torch.zeros(num_nodes, H * dh, device=device, dtype=x.dtype)

        for r in range(self.num_relations):
            key = str(r)
            edge_index = edge_index_dict[key]
            src, dst = edge_index[0], edge_index[1]

            q = self.q_proj[key](x[dst]).view(-1, H, dh)   # (E, H, dh)
            k = self.k_proj[key](x[src]).view(-1, H, dh)
            v = self.v_proj[key](x[src]).view(-1, H, dh)

            attn = torch.sum(q * k, dim=-1) / math.sqrt(dh) + self.mu[key]  # (E, H)
            attn = softmax(attn, dst, num_nodes=num_nodes)
            if edge_mask_dict is not None:
                attn = attn * edge_mask_dict[key].view(-1, 1)

            msg = self.dropout(v * attn.unsqueeze(-1))     # (E, H, dh)
            z_r = scatter(msg.reshape(-1, H * dh), dst, dim=0,
                          dim_size=num_nodes, reduce="sum")  # (N, hidden)
            agg = agg + self.a_proj[key](z_r)

        out = self.self_proj(x) + agg
        out = self.norm(out)
        return self.act(out)


class HAN(nn.Module):
    """Heterogeneous Graph Attention Network (single node type, K relations)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                HANLayer(layer_in, hidden_dim, num_relations,
                         num_heads=num_heads, dropout=dropout)
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
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return h


class HGT(nn.Module):
    """Heterogeneous Graph Transformer (single node type, K relations)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                HGTLayer(layer_in, hidden_dim, num_relations,
                         num_heads=num_heads, dropout=dropout)
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
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return h


class GCNLayer(nn.Module):
    """One GCN layer (Kipf & Welling, ICLR 2017) with self-loops.

    Symmetrically normalized mean aggregation over the augmented adjacency
    ``A + I`` followed by a linear map, without residual connections, so that
    the layer stays a faithful GCN baseline (contrast with the residual +
    LayerNorm blocks of :class:`HeteroGNN` / HAN / HGT).
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index, edge_mask=None):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # self-loops (augmented adjacency A + I)
        self_loop = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, self_loop])
        dst = torch.cat([dst, self_loop])
        one = torch.ones(src.size(0), device=x.device)
        if edge_mask is not None:
            one = one * torch.cat([edge_mask, torch.ones(num_nodes, device=x.device)])
        deg_src = scatter(one, src, dim=0, dim_size=num_nodes, reduce="sum")
        deg_dst = scatter(one, dst, dim=0, dim_size=num_nodes, reduce="sum")
        norm = (deg_src.pow(-0.5)[src] * deg_dst.pow(-0.5)[dst]) * one
        agg = scatter(x[src] * norm.unsqueeze(-1), dst, dim=0, dim_size=num_nodes, reduce="sum")
        return self.act(self.lin(self.dropout(agg)))


class GCN(nn.Module):
    """Two-layer GCN (Kipf & Welling, ICLR 2017) for node classification.

    Exposes the same interface as :class:`HeteroGNN`
    (``logits = model(x, edge_index_dict, edge_mask_dict=None)``); the input
    dictionary is expected to contain a single collapsed relation.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            layer_out = hidden_dim if layer_idx < num_layers - 1 else out_dim
            is_last = layer_idx == num_layers - 1
            layer = GCNLayer(layer_in, layer_out, dropout=0.0 if is_last else dropout)
            if is_last:
                layer.act = nn.Identity()  # final layer outputs logits
            self.layers.append(layer)

    def forward(self, x, edge_index_dict, edge_mask_dict=None):
        key = list(edge_index_dict.keys())[0]
        mask = None if edge_mask_dict is None else edge_mask_dict[key]
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict[key], edge_mask=mask)
        return h

    def embed(self, x, edge_index_dict, edge_mask_dict=None):
        key = list(edge_index_dict.keys())[0]
        mask = None if edge_mask_dict is None else edge_mask_dict[key]
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index_dict[key], edge_mask=mask)
        return h


class GATLayer(nn.Module):
    """One GAT layer (Velickovic et al., ICLR 2018) with self-loops.

    Multi-head additive attention over the neighbors of each node; head
    outputs are concatenated (mean-pooled in the final layer), followed by
    ELU, without residual connections.
    """

    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.0, concat=True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.concat = concat
        self.lin = nn.Linear(in_dim, hidden_dim, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.att.view(1, -1), gain=nn.init.calculate_gain("leaky_relu", 0.2))

    def forward(self, x, edge_index, edge_mask=None):
        num_nodes, H, dh = x.size(0), self.num_heads, self.head_dim
        src, dst = edge_index[0], edge_index[1]
        self_loop = torch.arange(num_nodes, device=x.device)
        src = torch.cat([src, self_loop])
        dst = torch.cat([dst, self_loop])
        h = self.lin(x).view(-1, H, dh)
        h_src, h_dst = h[src], h[dst]
        e = F.leaky_relu(torch.sum(torch.cat([h_src, h_dst], dim=-1) * self.att, dim=-1), negative_slope=0.2)
        alpha = softmax(e, dst, num_nodes=num_nodes)
        if edge_mask is not None:
            one = torch.cat([edge_mask, torch.ones(num_nodes, device=x.device)])
            alpha = alpha * one.unsqueeze(-1)
        msg = self.dropout(h_src * alpha.unsqueeze(-1))
        agg = scatter(msg.reshape(-1, H * dh), dst, dim=0, dim_size=num_nodes, reduce="sum")
        if not self.concat:  # final layer: mean over heads
            agg = agg.view(-1, H, dh).mean(dim=1)
        return self.act(agg)


class GAT(nn.Module):
    """Two-layer GAT (Velickovic et al., ICLR 2018) for node classification.

    Same interface as :class:`HeteroGNN`; expects a single collapsed relation.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            is_last = layer_idx == num_layers - 1
            # Final layer: single head producing out_dim logits (out_dim is not
            # necessarily divisible by num_heads); hidden layers concat heads.
            layer_out = out_dim if is_last else hidden_dim
            layer_heads = 1 if is_last else num_heads
            self.layers.append(
                GATLayer(layer_in, layer_out, num_heads=layer_heads, dropout=dropout, concat=not is_last)
            )

    def forward(self, x, edge_index_dict, edge_mask_dict=None):
        key = list(edge_index_dict.keys())[0]
        mask = None if edge_mask_dict is None else edge_mask_dict[key]
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict[key], edge_mask=mask)
        return h

    def embed(self, x, edge_index_dict, edge_mask_dict=None):
        key = list(edge_index_dict.keys())[0]
        mask = None if edge_mask_dict is None else edge_mask_dict[key]
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index_dict[key], edge_mask=mask)
        return h


__all__ = ["HAN", "HGT", "HANLayer", "HGTLayer", "GCN", "GCNLayer", "GAT", "GATLayer",
           "RGCN", "RGCNLayer"]


class RGCNLayer(nn.Module):
    """One relational GCN layer (Schlichtkrull et al., ESWC 2018, basis-free).

    Per-relation linear message maps + mean/weighted-sum aggregation + a
    self-loop, with residual + LayerNorm + ReLU (mirrors the HeteroGNN block
    design minus attention, for a fair backbone comparison).  Supports the
    same ``edge_mask_dict`` interface as :class:`HeteroGNN`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.rel_proj = nn.ModuleDict(
            {str(r): nn.Linear(in_dim, hidden_dim, bias=False) for r in range(num_relations)}
        )
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_dict, edge_mask_dict=None):
        num_nodes = x.size(0)
        out = self.self_proj(x)
        for r in range(self.num_relations):
            key = str(r)
            ei = edge_index_dict[key]
            if ei.size(1) == 0:
                continue
            src, dst = ei[0], ei[1]
            h = self.rel_proj[key](x)
            msg = h[src]
            if edge_mask_dict is not None:
                msg = msg * edge_mask_dict[key].unsqueeze(-1)
            agg = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
            out = out + agg
        return self.act(self.norm(out))


class RGCN(nn.Module):
    """Relational GCN backbone with the HeteroGNN interface.

    ``logits = model(x, edge_index_dict, edge_mask_dict=None)``.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_relations=2,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            layer_out = hidden_dim if layer_idx < num_layers - 1 else out_dim
            is_last = layer_idx == num_layers - 1
            layer = RGCNLayer(layer_in, layer_out, num_relations,
                              dropout=0.0 if is_last else dropout)
            if is_last:
                layer.act = nn.Identity()
            self.layers.append(layer)

    def forward(self, x, edge_index_dict, edge_mask_dict=None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return h

    def embed(self, x, edge_index_dict, edge_mask_dict=None):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index_dict, edge_mask_dict=edge_mask_dict)
        return h
