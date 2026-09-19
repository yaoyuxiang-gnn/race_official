"""Real heterogeneous ACM dataset loader (HAN version).

Loads ``ACM.mat`` (downloaded from ``data.dgl.ai``, accessible where Google Drive
is blocked) and projects it to a single-node-type multi-relation *paper* graph
with two semantically distinct meta-path relations:

* relation 0 : PAP — paper-author-paper (co-authorship);
* relation 1 : PTP — paper-term-paper (shared keywords).

Paper features are the term bag-of-words (``PvsT``, row-normalised); labels are
the venue/conference (``PvsC``, 14 classes).  These two relations have clearly
different semantics, which is the condition under which relation-type-level
counterfactual explanation is expected to work.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch import Tensor


def _metapath_edges(A: sp.csr_matrix, cap: int, seed: int) -> Tensor:
    """``(src x src)`` meta-path edges from a bipartite ``(src, dst)`` matrix."""
    S = (A @ A.T)  # co-occurrence counts (sparse)
    S = S.tolil()
    S.setdiag(0)
    S = S.tocsr()
    rows, cols = S.nonzero()
    if len(rows) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=cap, replace=False)
        rows, cols = rows[idx], cols[idx]
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)


def load_acm_han(
    mat_path: str = "/root/ACM.mat",
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Load the HAN ACM dataset into a two-relation paper graph."""
    d = sio.loadmat(mat_path)
    PvsT = sp.csr_matrix(d["PvsT"]).astype(np.float32)  # (12499, 1903) paper-term
    PvsA = sp.csr_matrix(d["PvsA"])                      # (12499, 17431) paper-author
    PvsC = sp.csr_matrix(d["PvsC"])                      # (12499, 14) paper-conference

    # features: row-normalised term bag-of-words.
    x = PvsT.toarray().astype(np.float32)
    x = x / (x.sum(axis=1, keepdims=True) + 1e-8)

    # labels: conference id; drop papers without a label.
    label_onehot = PvsC.toarray()
    has_label = label_onehot.sum(axis=1) > 0
    y = np.argmax(label_onehot, axis=1).astype(np.int64)
    keep = has_label
    x = x[keep]
    y = y[keep]
    PvsT = PvsT[keep]
    PvsA = PvsA[keep]
    num_nodes = int(keep.sum())

    # meta-path relations (on the kept paper subset).
    pap = _metapath_edges(PvsA, cap=cap, seed=seed)
    ptp = _metapath_edges(PvsT, cap=cap, seed=seed + 1)

    # random 60/20/20 split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_tr = int(0.6 * num_nodes)
    n_va = int(0.2 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va:]] = True

    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "edge_index_dict": {"0": pap, "1": ptp},
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
