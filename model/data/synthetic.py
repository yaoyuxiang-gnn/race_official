"""Synthetic heterogeneous graph generator with ground-truth causal relation.

Builds a single-node-type, multi-relation graph in which *exactly one* relation
(relation 0 by default) carries the label signal, while the remaining relations
are noise.  This is the only benchmark with verifiable causal truth: the label
of a node is determined by its community, and relation 0 connects same-community
(homophilic) pairs, so removing relation 0 must flip a large fraction of
predictions, while removing any noise relation must not.

Config mirrors Step-1 / Step-4 synthetic settings (300 nodes, 3 relations,
80/20 split in the original reports; we additionally expose a 60/20/20 split
for protocol uniformity with the ACM experiments).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


def load_synthetic(
    n_nodes: int = 300,
    n_relations: int = 3,
    n_classes: int = 4,
    feat_dim: int = 16,
    feat_noise: float = 4.0,
    p_in: float = 0.6,
    p_noise: float = 0.05,
    seed: int = 0,
    split: str = "60/20/20",
    cap: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Generate a synthetic heterogeneous graph with ground-truth causal relation.

    Args:
        n_nodes: Number of nodes.
        n_relations: Number of relation types (relation 0 = causal).
        n_classes: Number of communities / classes.
        feat_dim: Node feature dimension.
        feat_noise: Noise scale on node features.  Large values make the
            features weakly informative so that the backbone must rely on the
            causal relation, which is the regime where counterfactual flips
            are observable (matches the Step-1 synthetic configuration).
        p_in: Probability of a same-community edge in the causal relation.
        p_noise: Probability of an edge in each noise relation.
        seed: Random seed.
        split: ``"60/20/20"`` or ``"80/20"`` (train/val/test; 80/20 has no val).
        cap: Optional per-relation edge cap (randomly sampled).

    Returns:
        Dict with ``x``, ``y``, ``edge_index_dict`` (relation id ``str`` ->
        ``(2, E_r)``), and ``train_mask`` / ``val_mask`` / ``test_mask``.
    """
    rng = np.random.default_rng(seed)

    # --- communities and labels ------------------------------------------
    labels = np.repeat(np.arange(n_classes), n_nodes // n_classes)
    labels = np.concatenate([labels, np.full(n_nodes - len(labels), n_classes - 1)])
    labels = labels.astype(np.int64)

    # --- features: class centroid + noise --------------------------------
    centroids = rng.standard_normal((n_classes, feat_dim))
    x = centroids[labels] + feat_noise * rng.standard_normal((n_nodes, feat_dim))
    x = x.astype(np.float32)

    # --- relation 0: causal (homophilic) edges ---------------------------
    edges: Dict[str, Tensor] = {}
    src0, dst0 = [], []
    for c in range(n_classes):
        idx = np.nonzero(labels == c)[0]
        for i, u in enumerate(idx):
            for v in idx[i + 1:]:
                if rng.random() < p_in:
                    src0.append(u)
                    dst0.append(v)
    # randomize order so edge list is not community-blocked
    perm = rng.permutation(len(src0))
    e0 = torch.tensor(np.stack([np.array(src0)[perm], np.array(dst0)[perm]]), dtype=torch.long)
    if len(src0) == 0:
        e0 = torch.zeros((2, 0), dtype=torch.long)
    edges["0"] = e0

    # --- relations 1..K-1: noise (Erdos-Renyi) ---------------------------
    for r in range(1, n_relations):
        src, dst = [], []
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if rng.random() < p_noise:
                    src.append(u)
                    dst.append(v)
        e = torch.tensor(np.stack([np.array(src), np.array(dst)]), dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edges[str(r)] = e

    # --- optional per-relation cap ----------------------------------------
    if cap is not None:
        for r, e in edges.items():
            if e.size(1) > cap:
                idx = rng.choice(e.size(1), size=cap, replace=False)
                edges[r] = e[:, idx]

    # --- splits ------------------------------------------------------------
    n_nodes_actual = n_nodes
    perm = rng.permutation(n_nodes_actual)
    train_mask = torch.zeros(n_nodes_actual, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes_actual, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes_actual, dtype=torch.bool)
    if split == "80/20":
        n_tr = int(0.8 * n_nodes_actual)
        train_mask[perm[:n_tr]] = True
        test_mask[perm[n_tr:]] = True
    else:  # 60/20/20
        n_tr, n_va = int(0.6 * n_nodes_actual), int(0.2 * n_nodes_actual)
        train_mask[perm[:n_tr]] = True
        val_mask[perm[n_tr:n_tr + n_va]] = True
        test_mask[perm[n_tr + n_va:]] = True

    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(labels),
        "edge_index_dict": edges,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
