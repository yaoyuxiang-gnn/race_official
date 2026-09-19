"""Counterfactual explainer for relation-aware heterogeneous graphs.

The explainer answers *"which relation edges, when removed, flip the model's
prediction?"*.  It learns a differentiable per-edge keep-mask (``1`` = keep,
``0`` = remove) for every relation type using a binary Gumbel-Softmax
(Gumbel-Sigmoid) relaxation, optionally hardened with a straight-through
estimator.  The objective jointly (a) pushes the prediction away from the
original class (flip) and (b) minimises the number of removed edges (minimal
intervention).

The module also provides the causal necessity/sufficiency metrics used to
evaluate the learned counterfactual, following the PNS (probability of
necessity and sufficiency) framework:

* **PN (necessity)** — removing the counterfactual edge set ``S`` flips the
  prediction: ``P(f(G \\ S) != f(G))``.
* **PS (sufficiency)** — keeping *only* the counterfactual edge set ``S`` is
  enough to reproduce the original prediction: ``P(f(S) == f(G))``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.hetero_gnn import HeteroGNN


def probability_of_necessity(y_full: Tensor, y_counterfactual: Tensor) -> Tensor:
    """PN — probability that deleting the counterfactual edges flips the label.

    Args:
        y_full: Predicted classes on the full graph, shape ``(N,)``.
        y_counterfactual: Predicted classes after removing the counterfactual
            edge set, shape ``(N,)``.

    Returns:
        Scalar tensor in ``[0, 1]``: the fraction of nodes whose prediction
        changed after the intervention.
    """
    return (y_counterfactual != y_full).float().mean()


def probability_of_sufficiency(y_full: Tensor, y_only: Tensor) -> Tensor:
    """PS — probability that keeping only the counterfactual edges preserves the label.

    Args:
        y_full: Predicted classes on the full graph, shape ``(N,)``.
        y_only: Predicted classes when only the counterfactual edge set is kept,
            shape ``(N,)``.

    Returns:
        Scalar tensor in ``[0, 1]``: the fraction of nodes whose prediction is
        unchanged when all non-counterfactual edges are removed.
    """
    return (y_only == y_full).float().mean()


def probability_of_necessity_and_sufficiency(pn: Tensor, ps: Tensor) -> Tensor:
    """PNS — joint probability of necessity and sufficiency (Pearl's PNS).

    Under the standard monotonicity assumption PNS = PN * PS.
    """
    return pn * ps


def counterfactual_loss(
    logits: Tensor,
    y_original: Tensor,
    masks: Dict[str, Tensor],
    sparsity_coef: float = 0.05,
) -> Tensor:
    """Joint flip + minimality objective for the counterfactual masks.

    Args:
        logits: Model logits computed on the masked graph, shape ``(N, C)``.
        y_original: Original predicted class on the full graph, shape ``(N,)``.
        masks: Relation id (str) -> soft keep-mask ``(E_r,)`` in ``[0, 1]``.
        sparsity_coef: Weight of the minimal-intervention (sparsity) term.

    Returns:
        Scalar loss ``mean(softmax(logits)[:, y_original]) + sparsity_coef *
        mean(1 - mask)``.
    """
    probs = F.softmax(logits, dim=-1)
    p_original = probs.gather(1, y_original.unsqueeze(-1)).squeeze(-1)
    flip_loss = p_original.mean()  # smaller -> original class less likely

    total_edges = float(sum(int(m.numel()) for m in masks.values()))
    removed_edges = sum((1.0 - m).sum() for m in masks.values()) / max(total_edges, 1.0)
    return flip_loss + sparsity_coef * removed_edges


class CounterfactualExplainer(nn.Module):
    """Learnable relation-level counterfactual edge masks with PN/PS metrics.

    The mask logits are registered lazily (once the graph is known) via
    :meth:`reset`.  During every forward call a binary Gumbel-Sigmoid converts
    the logits into a soft keep-mask; ``hard=True`` additionally applies a
    straight-through estimator so that the returned mask is binary in the
    forward pass while remaining differentiable in the backward pass.

    Args:
        backbone: A frozen :class:`~model.hetero_gnn.HeteroGNN` to be explained.
        num_relations: Number of relation (edge) types.
        gumbel_tau: Temperature of the Gumbel-Sigmoid relaxation.
        hard: Whether to use a straight-through hard mask in the forward pass.
        init_logit: Initial value of every mask logit; a large positive value
            starts with all edges kept (mask close to ``1``).
    """

    def __init__(
        self,
        backbone: HeteroGNN,
        num_relations: int,
        gumbel_tau: float = 1.0,
        hard: bool = False,
        init_logit: float = 5.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_relations = num_relations
        self.gumbel_tau = gumbel_tau
        self.hard = hard
        self.init_logit = float(init_logit)
        # relation id (str) -> mask logits (E_r,), populated by reset().
        self.mask_logits = nn.ParameterDict()

    def reset(self, edge_index_dict: Dict[str, Tensor]) -> None:
        """Register one mask-logit parameter per relation edge (idempotent).

        Args:
            edge_index_dict: Relation id (str) -> ``(2, E_r)`` edge index tensor.
        """
        for r in range(self.num_relations):
            key = str(r)
            if key not in self.mask_logits:
                num_edges = int(edge_index_dict[key].size(1))
                dev = edge_index_dict[key].device
                self.mask_logits[key] = nn.Parameter(
                    torch.full((num_edges,), self.init_logit, dtype=torch.float32, device=dev)
                )

    def _gumbel_sigmoid(self, logits: Tensor) -> Tensor:
        """Binary Gumbel-Softmax (Gumbel-Sigmoid) over the mask logits."""
        uniforms = torch.rand_like(logits).clamp(min=1e-8)
        gumbels = -torch.log(-torch.log(uniforms) + 1e-8)
        y = torch.sigmoid((logits + gumbels) / self.gumbel_tau)
        if self.hard:
            y_hard = (y > 0.5).float()
            # Straight-through estimator: binary forward, soft gradient backward.
            y = (y_hard - y).detach() + y
        return y

    def sample_masks(self, edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Sample soft keep-masks for every relation.

        Args:
            edge_index_dict: Relation id (str) -> ``(2, E_r)`` edge index tensor.

        Returns:
            Relation id (str) -> keep-mask ``(E_r,)`` in ``[0, 1]``.
        """
        self.reset(edge_index_dict)
        return {str(r): self._gumbel_sigmoid(self.mask_logits[str(r)]) for r in range(self.num_relations)}

    def hard_masks(self, edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return deterministic binary keep-masks (threshold at 0.5).

        Used for the final PN/PS evaluation; does not draw Gumbel noise.
        """
        self.reset(edge_index_dict)
        return {
            str(r): (torch.sigmoid(self.mask_logits[str(r)]) > 0.5).float()
            for r in range(self.num_relations)
        }

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        target_nodes: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the backbone logits on the counterfactual (masked) graph.

        Args:
            x: Node features of shape ``(N, in_dim)``.
            edge_index_dict: Relation id (str) -> ``(2, E_r)`` edge index tensor.
            target_nodes: Optional node indices to return logits for (defaults
                to all nodes).

        Returns:
            Logits of shape ``(N, C)`` (or ``(len(target_nodes), C)``).
        """
        masks = self.sample_masks(edge_index_dict)
        logits = self.backbone(x, edge_index_dict, edge_mask_dict=masks)
        if target_nodes is not None:
            logits = logits[target_nodes]
        return logits

    def explain(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        target_nodes: Optional[Tensor] = None,
        steps: int = 30,
        lr: float = 0.05,
        sparsity_coef: float = 0.05,
    ) -> Dict[str, Tensor]:
        """Optimise the counterfactual masks for ``steps`` gradient steps.

        Returns the final soft keep-masks (relation id -> ``(E_r,)``).
        """
        self.reset(edge_index_dict)
        with torch.no_grad():
            y_original = self.backbone(x, edge_index_dict).argmax(dim=-1)
            if target_nodes is not None:
                y_original = y_original[target_nodes]

        opt = torch.optim.Adam(self.mask_logits.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            masks = self.sample_masks(edge_index_dict)
            logits = self.backbone(x, edge_index_dict, edge_mask_dict=masks)
            if target_nodes is not None:
                logits = logits[target_nodes]
            loss = counterfactual_loss(logits, y_original, masks, sparsity_coef=sparsity_coef)
            loss.backward()
            opt.step()
        return self.sample_masks(edge_index_dict)

    def evaluate_pn_ps(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        target_nodes: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate PN / PS / PNS with hard masks (no gradient).

        Returns a dict with ``"pn"``, ``"ps"``, ``"pns"`` and ``"flip_rate"``.
        """
        self.reset(edge_index_dict)
        with torch.no_grad():
            logits_full = self.backbone(x, edge_index_dict)
            keep = self.hard_masks(edge_index_dict)          # 1 = keep, 0 = remove
            remove = {k: 1.0 - v for k, v in keep.items()}    # keep only removed set S

            logits_cf = self.backbone(x, edge_index_dict, edge_mask_dict=keep)
            logits_only = self.backbone(x, edge_index_dict, edge_mask_dict=remove)

            y_full = logits_full.argmax(dim=-1)
            y_cf = logits_cf.argmax(dim=-1)
            y_only = logits_only.argmax(dim=-1)

            if target_nodes is not None:
                y_full, y_cf, y_only = y_full[target_nodes], y_cf[target_nodes], y_only[target_nodes]

            pn = probability_of_necessity(y_full, y_cf)
            ps = probability_of_sufficiency(y_full, y_only)
            pns = probability_of_necessity_and_sufficiency(pn, ps)

        return {
            "pn": float(pn.item()),
            "ps": float(ps.item()),
            "pns": float(pns.item()),
            "flip_rate": float(pn.item()),
        }

    def masked_edges(self, edge_index_dict: Dict[str, Tensor]) -> Dict[str, List[int]]:
        """Return the removed (masked-out) edge indices per relation, as lists."""
        self.reset(edge_index_dict)
        hard = self.hard_masks(edge_index_dict)
        removed: Dict[str, List[int]] = {}
        for r in range(self.num_relations):
            key = str(r)
            removed[key] = torch.nonzero(hard[key] < 0.5).squeeze(-1).tolist()
        return removed
