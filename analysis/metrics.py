"""Model-analysis helpers: accuracy and counterfactual evaluation metrics."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def accuracy(
    model: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y: Tensor,
    mask: Tensor,
) -> float:
    """Classification accuracy of ``model`` over ``mask``."""
    with torch.no_grad():
        pred = model(x, edge_index_dict).argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


def compute_cf_metrics(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y: Tensor,
    test_mask: Tensor,
    keep_masks: Dict[str, Tensor],
) -> Dict[str, float]:
    """Compute counterfactual success rate (CSR), minimality, PS and PNS.

    ``keep_masks`` maps relation id (str) -> hard keep-mask ``(E_r,)`` in
    ``{0, 1}`` (``1`` = keep edge).  All metrics are computed over ``test_mask``.
    """
    with torch.no_grad():
        y_full = backbone(x, edge_index_dict).argmax(dim=-1)
        y_cf = backbone(x, edge_index_dict, edge_mask_dict=keep_masks).argmax(dim=-1)
        remove = {k: 1.0 - v for k, v in keep_masks.items()}
        y_only = backbone(x, edge_index_dict, edge_mask_dict=remove).argmax(dim=-1)

    total_edges = sum(int(m.numel()) for m in keep_masks.values())
    removed = sum(int((m < 0.5).sum().item()) for m in keep_masks.values())
    pn = float((y_cf[test_mask] != y_full[test_mask]).float().mean().item())
    ps = float((y_only[test_mask] == y_full[test_mask]).float().mean().item())
    return {
        "csr": pn,
        "minimality": removed / max(total_edges, 1),
        "ps": ps,
        "pns": pn * ps,
    }
