"""Real heterogeneous-dataset loading.

Loads the HGB (Heterogeneous Graph Benchmark) ACM / DBLP datasets and projects
them to a *single-node-type, multi-relation* graph via meta-paths, matching the
input contract of :class:`model.hetero_gnn.HeteroGNN` (``edge_index_dict`` keyed
by relation id as string, with the target-node features / labels / masks).

* ACM  -> paper classification, relations = PAP (co-authorship) + PSP (shared subject).
* DBLP -> author classification, relations = APA (co-authored paper) + APTPA (shared term).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import HGBDataset


def _meta_path_edges(
    bipartite: Tensor,
    num_src: int,
    num_dst: int,
    cap: int = 50000,
    seed: int = 0,
) -> Tensor:
    """Compute ``(src x src)`` meta-path edges from a bipartite ``(src, dst)``.

    Returns a ``(2, E)`` tensor of undirected edges (i, j) where i and j share at
    least one ``dst`` node.  Edges are capped at ``cap`` (randomly sampled) to
    keep the graph tractable.
    """
    A = np.zeros((num_src, num_dst), dtype=np.float32)
    A[bipartite[0].numpy(), bipartite[1].numpy()] = 1.0
    S = A @ A.T  # (src, src) co-occurrence counts
    np.fill_diagonal(S, 0.0)
    rows, cols = np.nonzero(S > 0)
    if len(rows) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=cap, replace=False)
        rows, cols = rows[idx], cols[idx]
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)


def load_acm(
    root: str = "/tmp/hgb",
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Load ACM, project to paper-only graph with PAP + PSP relations."""
    data = HGBDataset(root, name="ACM")[0]
    x = data["paper"].x.float()
    y = data["paper"].y.long()

    pa = data[("paper", "pa", "author")].edge_index
    ps = data[("paper", "ps", "subject")].edge_index
    num_paper = int(data["paper"].num_nodes)
    num_author = int(data["author"].num_nodes)
    num_subject = int(data["subject"].num_nodes)

    pap = _meta_path_edges(pa, num_paper, num_author, cap=cap, seed=seed)
    psp = _meta_path_edges(ps, num_paper, num_subject, cap=cap, seed=seed + 1)

    return {
        "x": x,
        "y": y,
        "edge_index_dict": {"0": pap, "1": psp},
        "train_mask": data["paper"].train_mask,
        "val_mask": data["paper"].val_mask,
        "test_mask": data["paper"].test_mask,
    }


def load_dblp(
    root: str = "/tmp/hgb",
    cap: int = 50000,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Load DBLP, project to author-only graph with APA + APTPA relations."""
    data = HGBDataset(root, name="DBLP")[0]
    x = data["author"].x.float()
    y = data["author"].y.long()

    ap = data[("author", "ap", "paper")].edge_index  # author -> paper
    at = data[("author", "at", "term")].edge_index  # author -> term
    num_author = int(data["author"].num_nodes)
    num_paper = int(data["paper"].num_nodes)
    num_term = int(data["term"].num_nodes)

    apa = _meta_path_edges(ap, num_author, num_paper, cap=cap, seed=seed)
    aptpa = _meta_path_edges(at, num_author, num_term, cap=cap, seed=seed + 1)

    return {
        "x": x,
        "y": y,
        "edge_index_dict": {"0": apa, "1": aptpa},
        "train_mask": data["author"].train_mask,
        "val_mask": data["author"].val_mask,
        "test_mask": data["author"].test_mask,
    }
