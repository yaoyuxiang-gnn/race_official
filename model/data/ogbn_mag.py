"""ogbn-mag subsample loader: second real heterogeneous dataset (extension E1).

Projects the 1.9M-node heterogeneous Microsoft Academic Graph onto a
single-node-type *paper* graph with two semantically distinct relations:

* relation 0 : PAP — paper-author-paper (co-authorship);
* relation 1 : citation — direct paper-cites-paper.

The classification target is the venue (conference/journal) of a paper,
restricted to the top-``n_venues`` venues with up to ``n_papers`` sampled
papers (default 30,000), matching the tractability regime of the ACM
experiments (full-batch message passing).  Features are the 128-dim
word-embedding features shipped with ogbn-mag.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

# ogb 1.3.6 calls torch.load without weights_only, which fails under torch>=2.6
# (weights_only=True default).  Inject weights_only=False for compatibility.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def load_mag(
    root: str = "data/ogb",
    n_papers: int = 30000,
    n_venues: int = 20,
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Load ogbn-mag and build the two-relation paper graph."""
    from ogb.nodeproppred import PygNodePropPredDataset

    ds = PygNodePropPredDataset(name="ogbn-mag", root=root)
    g = ds[0]
    x = g.x_dict["paper"]  # (num_papers, 128)
    y_raw = g.y_dict["paper"].squeeze(-1)
    labeled = ~torch.isnan(y_raw)
    x = x[labeled]
    y = y_raw[labeled].long()

    # --- restrict to top venues ------------------------------------------
    counts = torch.bincount(y)
    top_venues = torch.topk(counts, min(n_venues, len(counts))).indices
    in_top = torch.isin(y, top_venues)
    idx_all = torch.nonzero(in_top).squeeze(-1)
    rng = np.random.default_rng(seed)
    sample = []
    per_venue = max(1, n_papers // len(top_venues))
    for v in top_venues:
        vi = idx_all[y[idx_all] == int(v)]
        if len(vi) > per_venue:
            vi = vi[torch.from_numpy(rng.choice(len(vi), size=per_venue, replace=False))]
        sample.append(vi)
    sample = torch.cat(sample)
    n = sample.size(0)
    node_map = {int(old): new for new, old in enumerate(sample.tolist())}
    x_sub = x[sample].float()
    y_sub = y[sample]
    # reindex venue ids to a compact 0..(n_venues-1) range
    venue_map = {int(v): i for i, v in enumerate(sorted(top_venues.tolist()))}
    y_sub = torch.tensor([venue_map[int(v)] for v in y_sub.tolist()], dtype=torch.long)

    # --- relation 1: citation (paper cites paper) ------------------------
    cites = g.edge_index_dict[("paper", "cites", "paper")]
    keep = torch.zeros(x.size(0), dtype=torch.bool)
    keep[sample] = True
    m = keep[cites[0]] & keep[cites[1]]
    ec = cites[:, m]
    if ec.size(1) > cap:
        perm = torch.randperm(ec.size(1))[:cap]
        ec = ec[:, perm]
    ec = torch.stack([torch.tensor([node_map[int(u)] for u in ec[0].tolist()]),
                      torch.tensor([node_map[int(v)] for v in ec[1].tolist()])])
    ec = torch.cat([ec, ec.flip(0)], dim=1)  # undirected

    # --- relation 0: PAP co-authorship -----------------------------------
    writes = g.edge_index_dict[("author", "writes", "paper")]  # (2, E) author->paper
    keep_w = keep[writes[1]]
    writes = writes[:, keep_w]
    author_to_papers: Dict[int, list] = {}
    for a, p in zip(writes[0].tolist(), writes[1].tolist()):
        author_to_papers.setdefault(int(a), []).append(node_map[int(p)])
    pap_src, pap_dst = [], []
    for papers in author_to_papers.values():
        if len(papers) < 2 or len(papers) > 50:  # skip mega-authors (noise)
            continue
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                pap_src.append(papers[i])
                pap_dst.append(papers[j])
    ep = torch.tensor([pap_src, pap_dst], dtype=torch.long) if pap_src else torch.zeros((2, 0), dtype=torch.long)
    if ep.size(1) > cap:
        perm = torch.randperm(ep.size(1))[:cap]
        ep = ep[:, perm]
    ep = torch.cat([ep, ep.flip(0)], dim=1)

    # --- splits (random 60/20/20, same protocol as ACM) ------------------
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va:]] = True

    return {
        "x": x_sub,
        "y": y_sub,
        "edge_index_dict": {"0": ep, "1": ec},
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
