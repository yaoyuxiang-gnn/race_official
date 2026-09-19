"""ogbn-arXiv loader: paper-only graph with two derived relations.

  relation 0: direct citation (undirected, within the kept papers);
  relation 1: co-citation (papers sharing at least one reference) -- derived
              from the citation graph without label information, analogous to
              the Cora "common-neighbor" derivation.

Kept papers: a random sample of ``n_nodes`` papers whose labels belong to the
``n_classes`` most frequent subject areas.  Random 60/20/20 split.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def load_arxiv(
    root: str = "data/ogb",
    n_nodes: int = 30000,
    n_classes: int = 20,
    cap: int = 100000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    # torch >= 2.6 defaults torch.load to weights_only=True; allowlist the
    # torch_geometric globals stored in OGB's processed files.
    import torch.serialization
    import torch_geometric.data.data as pgd
    torch.serialization.add_safe_globals(
        [v for v in vars(pgd).values() if isinstance(v, type)])

    from ogb.nodeproppred import PygNodePropPredDataset

    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    g = ds[0]
    x = g.x.float()
    y = g.y.squeeze(-1).long()
    rng = np.random.default_rng(seed)

    # keep the top-n_classes labels
    label_counts = torch.bincount(y)
    top_labels = torch.topk(label_counts, n_classes).indices
    label_ok = torch.isin(y, top_labels)

    # sample n_nodes papers among label_ok
    idx_all = label_ok.nonzero().squeeze(-1).numpy()
    keep = rng.choice(idx_all, size=min(n_nodes, len(idx_all)), replace=False)
    keep_t = torch.tensor(np.sort(keep), dtype=torch.long)
    keep_set = set(keep.tolist())

    # relation 0: citation (undirected, within kept papers)
    ei = g.edge_index
    m = torch.isin(ei[0], keep_t) & torch.isin(ei[1], keep_t)
    ec = ei[:, m]
    ec_und = torch.cat([ec, ec.flip(0)], dim=1)
    if ec_und.size(1) > cap:
        idx = rng.choice(ec_und.size(1), size=cap, replace=False)
        ec_und = ec_und[:, idx]

    # relation 1: co-citation -- for each cited paper k, sample pairs among the
    # kept papers that cite k
    cited_to_src: Dict[int, list] = {}
    src_l, dst_l = ei[0].tolist(), ei[1].tolist()
    for u, k in zip(src_l, dst_l):
        if u in keep_set:
            cited_to_src.setdefault(k, []).append(u)
    rows, cols = [], []
    for k, srcs in cited_to_src.items():
        if len(srcs) < 2:
            continue
        n = len(srcs)
        # sample up to 3 pairs per cited paper to bound density
        for _ in range(3):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                rows.append(srcs[i])
                cols.append(srcs[j])
        if len(rows) > cap:
            break
    co = torch.tensor([rows[:cap], cols[:cap]], dtype=torch.long)

    # split 60/20/20 (random over kept papers)
    perm = rng.permutation(len(keep))
    n = len(keep)
    tr = torch.zeros(x.size(0), dtype=torch.bool)
    va = torch.zeros(x.size(0), dtype=torch.bool)
    te = torch.zeros(x.size(0), dtype=torch.bool)
    tr[keep_t[perm[: int(0.6 * n)]]] = True
    va[keep_t[perm[int(0.6 * n): int(0.8 * n)]]] = True
    te[keep_t[perm[int(0.8 * n):]]] = True

    # renumber to the kept nodes (0..n-1) so no isolated nodes remain
    node_map = torch.full((x.size(0),), -1, dtype=torch.long)
    node_map[keep_t] = torch.arange(n)
    ec_und = node_map[ec_und]
    co = node_map[co]
    ec_und = ec_und[:, (ec_und >= 0).all(dim=0)]
    co = co[:, (co >= 0).all(dim=0)]

    return {
        "x": x[keep_t].contiguous(),
        "y": y[keep_t],
        "edge_index_dict": {"0": ec_und, "1": co},
        "train_mask": tr[keep_t],
        "val_mask": va[keep_t],
        "test_mask": te[keep_t],
    }
