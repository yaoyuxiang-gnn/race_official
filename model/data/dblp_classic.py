"""DBLP (heterogeneous version, author classification) loader -- two relations.

Source: dblp.zip from https://data.dgl.ai/dataset/dblp.zip, whose graph.pickle is
a DGL heterograph.  The author-author relations were extracted WITHOUT dgl via
:mod:`extract_dblp_edges` (stub-based unpickling) into plain npz edge lists:

  relation 0: co-authorship          (author--paper--author meta-path)
  relation 1: shared-term co-author  (author--paper--term--paper--author,
                                      capped at ``cap`` edges)

Features are the author bag-of-words (334 terms); labels are the 4 research
areas.  Random 60/20/20 split over the labeled authors.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


def _load_csr(path: str) -> sp.csr_matrix:
    f = np.load(path, allow_pickle=True)
    return sp.csr_matrix((f["data"], f["indices"], f["indptr"]),
                         shape=tuple(f["shape"])).astype(np.float32)


def load_dblp_classic(
    root: str = "data/dblp",
    topk: int = 12,
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    labels = np.load(os.path.join(root, "labels.npy"), allow_pickle=True).astype(np.int64)
    feats = _load_csr(os.path.join(root, "features", "author.npz")).toarray()
    n_labeled = labels.shape[0]  # 4057 of 4058 authors

    def _load_edges(name: str, n_max: int):
        f = np.load(os.path.join(root, f"edges_{name}.npz"))
        src, dst = f["src"], f["dst"]
        m = (src < n_labeled) & (dst < n_labeled) & (src != dst)
        src, dst = src[m], dst[m]
        if len(src) > n_max:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(src), size=n_max, replace=False)
            src, dst = src[idx], dst[idx]
        return np.stack([src, dst])

    co = _load_edges("coauthor", 100000)          # 32,789 edges
    st = _load_edges("shared_term", cap)          # capped shared-topic relation

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_labeled)
    tr = torch.zeros(n_labeled, dtype=torch.bool)
    va = torch.zeros(n_labeled, dtype=torch.bool)
    te = torch.zeros(n_labeled, dtype=torch.bool)
    tr[perm[: int(0.6 * n_labeled)]] = True
    va[perm[int(0.6 * n_labeled): int(0.8 * n_labeled)]] = True
    te[perm[int(0.8 * n_labeled):]] = True

    return {
        "x": torch.tensor(feats[:n_labeled], dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.long),
        "edge_index_dict": {
            "0": torch.tensor(co, dtype=torch.long),
            "1": torch.tensor(st, dtype=torch.long),
        },
        "train_mask": tr,
        "val_mask": va,
        "test_mask": te,
    }
