"""ogbn-mag 4-relation natural multi-relation loader (C2, advisor R7).

Projects the original heterogeneous MAG schema onto the SAME 30k-paper sample
as ``ogbn_mag.load_mag`` (identical sampling code and seeds, so mag2 vs mag4 is
a clean relation-vocabulary ablation), keeping FOUR natural relation types:

  * relation 0 : PAP  -- paper-author-paper (co-authorship);
  * relation 1 : citation -- paper-cites-paper (undirected storage);
  * relation 2 : PFP  -- papers sharing at least one field of study
                 (paper-has_topic-field meta-path, the original schema's
                 ``has_topic`` edge, NOT a derived/constructed split);
  * relation 3 : PAIP -- papers whose authors share an affiliated institution
                 (author-affiliated_with-institution meta-path via ``writes``,
                 the original schema's ``affiliated_with`` edge).

Selection rationale (fixed before running, per advisor's core package):
  (i) all four relations come from the heterogeneous schema of a widely used
      benchmark, none is a synthetic binary split of one relation;
  (ii) semantics are distinct: collaboration, knowledge flow, subject
      proximity, institutional proximity;
  (iii) K=4 is the largest natural relation vocabulary available in the data
      already cached on the compute node (DBLP-3 would need a new download).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from model.data.ogbn_mag import _torch_load_compat  # noqa: F401  (weights_only compat)

torch.load = _torch_load_compat


def _sym_pairs_from_matrix(W: np.ndarray, cap: int, rng: np.random.Generator):
    """All (i<j) nonzero pairs of W @ W.T, optionally capped by seeded sampling."""
    S = (W @ W.T).astype(bool)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    src = iu[0][S[iu]]
    dst = iu[1][S[iu]]
    if len(src) > cap:
        idx = rng.choice(len(src), size=cap, replace=False)
        src, dst = src[idx], dst[idx]
    return np.concatenate([src, dst]), np.concatenate([dst, src])


def load_mag4(
    root: str = "data/ogb",
    n_papers: int = 30000,
    n_venues: int = 20,
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Load ogbn-mag and build the four-relation paper graph."""
    from ogb.nodeproppred import PygNodePropPredDataset

    ds = PygNodePropPredDataset(name="ogbn-mag", root=root)
    g = ds[0]
    x = g.x_dict["paper"]
    y_raw = g.y_dict["paper"].squeeze(-1)
    labeled = ~torch.isnan(y_raw)
    x = x[labeled]
    y = y_raw[labeled].long()

    # ---- identical paper subsample as ogbn_mag.load_mag ----------------
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
    venue_map = {int(v): i for i, v in enumerate(sorted(top_venues.tolist()))}
    y_sub = torch.tensor([venue_map[int(v)] for v in y_sub.tolist()], dtype=torch.long)

    # ---- relation 1: citation (identical to load_mag) ------------------
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
    ec = torch.cat([ec, ec.flip(0)], dim=1)

    # ---- relation 0: PAP co-authorship (identical to load_mag) ---------
    writes = g.edge_index_dict[("author", "writes", "paper")]
    keep_w = keep[writes[1]]
    writes = writes[:, keep_w]
    author_to_papers: Dict[int, list] = {}
    for a, p in zip(writes[0].tolist(), writes[1].tolist()):
        author_to_papers.setdefault(int(a), []).append(node_map[int(p)])
    pap_src, pap_dst = [], []
    for papers in author_to_papers.values():
        if len(papers) < 2 or len(papers) > 50:
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

    # ---- relation 2: PFP shared field of study --------------------------
    topics = g.edge_index_dict[("paper", "has_topic", "field_of_study")]
    m_t = keep[topics[0]]
    pt = topics[:, m_t]  # paper -> field
    pt = pt[:, torch.isin(pt[0], sample)]
    n_fields = int(pt[1].max().item()) + 1
    W = np.zeros((n, n_fields), dtype=np.float32)
    for p_old, f in zip(pt[0].tolist(), pt[1].tolist()):
        W[node_map[int(p_old)], int(f)] = 1.0
    pfp_s, pfp_d = _sym_pairs_from_matrix(W, cap, rng)
    ep2 = torch.tensor(np.stack([pfp_s, pfp_d]), dtype=torch.long)

    # ---- relation 3: PAIP shared affiliated institution ------------------
    affil = g.edge_index_dict[("author", "affiliated_with", "institution")]
    author_to_insts: Dict[int, set] = {}
    for a, i in zip(affil[0].tolist(), affil[1].tolist()):
        author_to_insts.setdefault(int(a), set()).add(int(i))
    inst_to_authors: Dict[int, set] = {}
    for a, insts in author_to_insts.items():
        for i in insts:
            inst_to_authors.setdefault(i, set()).add(a)
    pai_s, pai_d = [], []
    for inst, authors in inst_to_authors.items():
        kept_authors = sorted(a for a in authors if a in author_to_papers)
        if len(kept_authors) < 2:
            continue
        pairs = [(kept_authors[i], kept_authors[j])
                 for i in range(len(kept_authors)) for j in range(i + 1, len(kept_authors))]
        if len(pairs) > 200:  # bound per-institution expansion, seeded
            idx = rng.choice(len(pairs), size=200, replace=False)
            pairs = [pairs[i] for i in idx]
        for a1, a2 in pairs:
            p1 = author_to_papers.get(a1, [])
            p2 = author_to_papers.get(a2, [])
            for u in p1[:4]:
                for v in p2[:4]:
                    if u != v:
                        pai_s.append(u)
                        pai_d.append(v)
    if pai_s:
        arr = np.stack([np.asarray(pai_s), np.asarray(pai_d)])
        if arr.shape[1] > cap:
            idx = rng.choice(arr.shape[1], size=cap, replace=False)
            arr = arr[:, idx]
        # symmetrize
        arr = np.concatenate([arr, arr[::-1]], axis=1)
        ep3 = torch.tensor(arr, dtype=torch.long)
    else:
        ep3 = torch.zeros((2, 0), dtype=torch.long)

    # ---- splits (random 60/20/20, same protocol as load_mag) ------------
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
        "edge_index_dict": {"0": ep, "1": ec, "2": ep2, "3": ep3},
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
