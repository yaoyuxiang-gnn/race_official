"""Real citation-network loader (Cora).

Downloads the classic Planetoid Cora raw files from GitHub (accessible where
Google Drive / Dropbox are blocked), reconstructs the 2708-node citation graph,
and derives a **two-relation** graph:

* relation 0 : direct citation edges (undirected);
* relation 1 : common-neighbour (structural similarity) edges — node pairs
  sharing at least one neighbour, not already directly connected.

Both relations are real, derivable structure; they have distinct semantics, which
is exactly what the relation-level counterfactual explainer needs to compare.
"""

from __future__ import annotations

import os
import pickle
import urllib.request
from typing import Dict

import numpy as np
import torch
from torch import Tensor

BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


def _download(name: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return path


def load_cora(cache_dir: str = "/root/cora", cap: int = 60000) -> Dict[str, Tensor]:
    """Download + process Cora into a two-relation graph.

    Returns a dict with ``x``, ``y``, ``edge_index_dict`` (two relations),
    ``train_mask`` / ``val_mask`` / ``test_mask``.
    """
    names = [
        "ind.cora.x", "ind.cora.y", "ind.cora.tx", "ind.cora.ty",
        "ind.cora.allx", "ind.cora.ally", "ind.cora.graph",
    ]
    raw = {}
    for n in names:
        with open(_download(n, cache_dir), "rb") as f:
            raw[n] = pickle.load(f, encoding="latin1")

    x = np.vstack([raw["ind.cora.allx"].toarray(), raw["ind.cora.tx"].toarray()]).astype(np.float32)
    # Row-normalise the bag-of-words features (standard for citation networks;
    # raw counts otherwise dominate the message passing).
    x = x / (x.sum(axis=1, keepdims=True) + 1e-8)
    y = np.vstack([raw["ind.cora.ally"], raw["ind.cora.ty"]])
    y = np.argmax(y, axis=1).astype(np.int64)
    num_nodes = x.shape[0]

    # relation 0: direct citation (undirected) from the Planetoid graph dict.
    graph = raw["ind.cora.graph"]
    edges0 = set()
    for u, nbrs in graph.items():
        for v in nbrs:
            a, b = (u, v) if u < v else (v, u)
            edges0.add((a, b))
    e0 = torch.tensor(sorted(edges0), dtype=torch.long).t().contiguous()  # (2, E0)

    # relation 1: common-neighbour pairs (structural similarity), capped.
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    adj[e0[0], e0[1]] = True
    adj[e0[1], e0[0]] = True
    cn = (adj.float() @ adj.float()) > 0  # common neighbour
    cn = cn & ~adj & ~torch.eye(num_nodes, dtype=torch.bool)  # not directly connected
    rows, cols = torch.nonzero(cn, as_tuple=True)
    if rows.numel() > cap:
        perm = torch.randperm(rows.numel())[:cap]
        rows, cols = rows[perm], cols[perm]
    e1 = torch.stack([rows, cols], dim=0)  # (2, E1)

    # standard Planetoid split: 140 train / 500 val / 1000 test.
    # The 1000 test nodes are exactly the ``tx`` nodes (indices 1708..2707).
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:140] = True
    val_mask[140:640] = True
    test_mask[num_nodes - 1000:] = True

    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "edge_index_dict": {"0": e0, "1": e1},
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
