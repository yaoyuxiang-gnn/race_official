"""Relation-TYPE-level counterfactual explainer.

Unlike :class:`model.counterfactual_explainer.CounterfactualExplainer` (one mask
per edge), this module learns **one scalar keep-mask per relation type**, so the
intervention is "drop relation type ``r`` entirely" rather than "drop these
specific edges".  This is the coarser, more interpretable granularity: the
explainer answers *which relation type is causal*.

It reuses the same flip + minimality objective and the PN/PS/PNS metrics, and is
device-agnostic (all tensors follow ``x.device``).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.counterfactual_explainer import (
    probability_of_necessity,
    probability_of_necessity_and_sufficiency,
    probability_of_sufficiency,
)
from model.hetero_gnn import HeteroGNN


class RelationTypeExplainer(nn.Module):
    """Learn one scalar keep-mask per relation type (relation-level intervention).

    Args:
        backbone: A frozen :class:`~model.hetero_gnn.HeteroGNN` to explain.
        num_relations: Number of relation (edge) types.
        gumbel_tau: Temperature of the Gumbel-Sigmoid relaxation.
        init_logit: Initial scalar logit (large positive = keep the relation).
    """

    def __init__(
        self,
        backbone: HeteroGNN,
        num_relations: int,
        gumbel_tau: float = 1.0,
        init_logit: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_relations = num_relations
        self.gumbel_tau = gumbel_tau
        self.mask_logits = nn.Parameter(
            torch.full((num_relations,), init_logit, dtype=torch.float32)
        )

    def _soft_masks(self, edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Broadcast the per-type scalar soft mask to every edge of the type.

        Uses a deterministic sigmoid (no Gumbel noise): with only a handful of
        relation types, Gumbel sampling makes the loss too noisy to optimise the
        scalar logits reliably.
        """
        device = next(iter(edge_index_dict.values())).device
        self.mask_logits.data = self.mask_logits.data.to(device)
        scalars = torch.sigmoid(self.mask_logits)
        return {str(r): scalars[r].expand(edge_index_dict[str(r)].size(1)) for r in range(self.num_relations)}

    def hard_masks(self, edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Deterministic binary keep-masks (one scalar per relation, broadcast)."""
        device = next(iter(edge_index_dict.values())).device
        self.mask_logits.data = self.mask_logits.data.to(device)
        scalars = (torch.sigmoid(self.mask_logits) > 0.5).float()
        return {str(r): scalars[r].expand(edge_index_dict[str(r)].size(1)) for r in range(self.num_relations)}

    def explain(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        target_nodes: Optional[Tensor] = None,
        steps: int = 100,
        lr: float = 0.1,
        sparsity: float = 0.05,
    ) -> Dict[str, Tensor]:
        """Optimise the per-type scalar masks and return the final hard masks."""
        device = x.device
        self.mask_logits.data = self.mask_logits.data.to(device)
        with torch.no_grad():
            y_orig = self.backbone(x, edge_index_dict).argmax(dim=-1)
            if target_nodes is not None:
                y_orig = y_orig[target_nodes]

        opt = torch.optim.Adam([self.mask_logits], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            masks = self._soft_masks(edge_index_dict)
            logits = self.backbone(x, edge_index_dict, edge_mask_dict=masks)
            if target_nodes is not None:
                logits = logits[target_nodes]
            probs = F.softmax(logits, dim=-1)
            p_orig = probs.gather(1, y_orig.unsqueeze(-1)).squeeze(-1).mean()
            removed = (1.0 - torch.sigmoid(self.mask_logits)).mean()  # fraction of types removed
            loss = p_orig + sparsity * removed
            loss.backward()
            opt.step()
        return self.hard_masks(edge_index_dict)

    def evaluate_pn_ps(
        self, x: Tensor, edge_index_dict: Dict[str, Tensor], target_nodes: Optional[Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate PN / PS / PNS with hard per-type masks."""
        device = x.device
        self.mask_logits.data = self.mask_logits.data.to(device)
        with torch.no_grad():
            y_full = self.backbone(x, edge_index_dict).argmax(dim=-1)
            keep = self.hard_masks(edge_index_dict)
            remove = {k: 1.0 - v for k, v in keep.items()}
            y_cf = self.backbone(x, edge_index_dict, edge_mask_dict=keep).argmax(dim=-1)
            y_only = self.backbone(x, edge_index_dict, edge_mask_dict=remove).argmax(dim=-1)
            if target_nodes is not None:
                y_full, y_cf, y_only = y_full[target_nodes], y_cf[target_nodes], y_only[target_nodes]
            pn = probability_of_necessity(y_full, y_cf)
            ps = probability_of_sufficiency(y_full, y_only)
            pns = probability_of_necessity_and_sufficiency(pn, ps)
        return {"pn": float(pn.item()), "ps": float(ps.item()), "pns": float(pns.item())}

    def explain_exhaustive(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        target_nodes: Optional[Tensor] = None,
        rel_threshold: float = 0.9,
    ) -> List[int]:
        """Exhaustively find the minimal relation-type subset that flips predictions.

        Because the number of relation types is tiny (2^K subsets), exhaustive
        search is both tractable and *optimal* — and, unlike gradient-based
        optimisation, deterministic.  It returns the smallest subset ``S`` whose
        flip rate reaches ``rel_threshold`` of the flip rate of removing **all**
        relations (the minimal-intervention causal relation set).

        Args:
            x: Node features.
            edge_index_dict: Relation id -> ``(2, E_r)``.
            target_nodes: Optional nodes to evaluate on (default: all).
            rel_threshold: Fraction of the full-removal flip rate to reach.

        Returns:
            List of relation ids forming the minimal causal subset.
        """
        device = x.device
        self.mask_logits.data = self.mask_logits.data.to(device)
        K = self.num_relations
        with torch.no_grad():
            y_full = self.backbone(x, edge_index_dict).argmax(dim=-1)
            if target_nodes is not None:
                y_full = y_full[target_nodes]

            def flip_rate(subset) -> float:
                masks = {str(r): torch.ones(edge_index_dict[str(r)].size(1), device=device) for r in range(K)}
                for r in subset:
                    masks[str(r)] = torch.zeros(edge_index_dict[str(r)].size(1), device=device)
                y_cf = self.backbone(x, edge_index_dict, edge_mask_dict=masks).argmax(dim=-1)
                if target_nodes is not None:
                    y_cf = y_cf[target_nodes]
                return float((y_cf != y_full).float().mean().item())

            full_flip = flip_rate(tuple(range(K)))
            target = rel_threshold * full_flip if full_flip > 0 else 0.0
            best: List[int] = []
            for k in range(K + 1):
                for subset in combinations(range(K), k):
                    if flip_rate(subset) >= target:
                        best = list(subset)
                        # keep the logits consistent with the found subset
                        with torch.no_grad():
                            self.mask_logits.copy_(torch.full((K,), 3.0, device=device))
                            for r in best:
                                self.mask_logits[r] = -3.0
                        return best
            return best

    def removed_relation_types(self) -> List[int]:
        """Return the ids of relation types currently masked out (kept < 0.5)."""
        keep = (torch.sigmoid(self.mask_logits) > 0.5).float()
        return [r for r in range(self.num_relations) if keep[r].item() < 0.5]
