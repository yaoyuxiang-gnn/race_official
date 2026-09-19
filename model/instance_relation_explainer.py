"""Instance-level relation-type counterfactual search (RACE-v2, P0-1).

Replaces the dataset-level Eq.(4)-(5) of the v1 report with the per-instance
problem of the revision plan (advisor Sec. 2.1-2.2):

    M_v(G)   = p_f(y_v | G, v) - max_{c != y_v} p_f(c | G, v)      (margin)
    S_v*     = argmin_{S subset R} C_v(S)  s.t.  M_v(G \\ S) <= -kappa
    C_v(S)   = lexicographic (|S|,  edge-cost(S, v),   M_v(G \\ S))
    edge-cost(S, v) = sum_{r in S} |E_r cap N_L(v)| / |E cap N_L(v)|

    The third component is *minimized*: among candidates with equal |S| and
    edge cost, the most negative post-intervention margin -- the strongest
    flip -- wins (e.g. margin -0.9 is preferred over -0.1), matching eq:cost
    in the paper.

where N_L(v) is the union of backward balls of radius L-1 (the backbone's
receptive field; L = number of message-passing layers).  ``kappa >= 0`` is the
target flip-confidence margin (kappa = 0 means a plain argmax flip).

Because K is small the search enumerates every 2^K subset in increasing size:
the procedure is exact and deterministic.  A node for which NO subset reaches
the margin constraint is reported as *infeasible* (no relation-level
counterfactual exists) and coverage is reported explicitly.  One forward pass
per subset evaluates all target nodes simultaneously, so the total cost is
2^K forwards plus 2^K complement forwards for the keep-only (sufficiency)
check.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def prediction_margin(probs: Tensor, y: Tensor) -> Tensor:
    """Per-node margin M_v = p(y_v) - max_{c != y_v} p(c)."""
    p_y = probs.gather(1, y.unsqueeze(-1)).squeeze(-1)
    mask = F.one_hot(y, probs.size(-1)).bool()
    p_other = probs.masked_fill(mask, -float("inf")).max(dim=-1).values
    return p_y - p_other


def local_edge_counts(
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    target_nodes: List[int],
    num_layers: int,
) -> Tensor:
    """Per-node per-relation counts of edges inside the receptive field.

    Returns a (len(target_nodes), K) long tensor on CPU: entry [i, r] is the
    number of edges of relation r whose destination lies in the backward ball
    of radius ``num_layers - 1`` of target node i (union over relations).
    """
    N = x.size(0)
    K = len(edge_index_dict)
    device = x.device

    # In-degree per relation: indeg[r, w] = number of r-edges with dst w.
    indeg = torch.zeros((K, N), dtype=torch.long, device=device)
    for r in range(K):
        dst = edge_index_dict[str(r)][1]
        idx = torch.zeros(N, dtype=torch.long, device=device)
        idx.scatter_add_(0, dst, torch.ones(dst.size(0), dtype=torch.long, device=device))
        indeg[r] = idx

    # Backward adjacency (union over relations): dst -> list of src.
    adj: List[List[int]] = [[] for _ in range(N)]
    for r in range(K):
        ei = edge_index_dict[str(r)]
        src, dst = ei[0].tolist(), ei[1].tolist()
        for u, w in zip(src, dst):
            adj[w].append(u)

    radius = max(num_layers - 1, 0)
    counts = torch.zeros((len(target_nodes), K), dtype=torch.long)
    indeg_cpu = indeg.cpu()
    for i, v in enumerate(target_nodes):
        ball = {v}
        frontier = {v}
        for _ in range(radius):
            nxt = set()
            for w in frontier:
                for u in adj[w]:
                    if u not in ball:
                        nxt.add(u)
                        ball.add(u)
            if not nxt:
                break
            frontier = nxt
        for r in range(K):
            counts[i, r] = sum(int(indeg_cpu[r, w].item()) for w in ball)
    return counts


class InstanceRelationExplainer:
    """Exact instance-level minimal relation deletion search on a frozen model.

    Args:
        backbone: Frozen heterogeneous GNN to explain.
        num_relations: Number of relation (edge) types.
        num_layers: Number of message-passing layers of ``backbone`` (defines
            the receptive field used by the local edge cost).
        kappa: Target flip-confidence margin (valid iff M_v <= -kappa).
    """

    def __init__(self, backbone, num_relations: int, num_layers: int = 2, kappa: float = 0.0) -> None:
        self.backbone = backbone
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.kappa = float(kappa)

    @torch.no_grad()
    def explain(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
        y_true: Tensor,
        target_mask: Optional[Tensor] = None,
        max_targets: Optional[int] = None,
        verbose: bool = False,
        select_mode: str = "lexicographic",  # "lexicographic" | "cost_type" | "cost_edge"
    ) -> Dict:
        """Run the exact per-instance search over correctly-classified targets.

        Args:
            x: Node features (N, d).
            edge_index_dict: relation id (str) -> (2, E_r) edge index.
            y_true: Ground-truth labels (N,).
            target_mask: Bool mask selecting the evaluation split (default: all).
            max_targets: If the number of correct targets exceeds this, subsample
                them evenly (deterministic).

        Returns a dict with per-instance ``records`` and aggregate statistics:
        coverage, nf, sf, mean_cost_type, mean_cost_edge, mean_final_margin,
        and per-subset flip rates (the dataset-level Eq.(4) values).
        """
        device = x.device
        K = self.num_relations
        logits_full = self.backbone(x, edge_index_dict)
        probs_full = F.softmax(logits_full, dim=-1)
        y_hat = logits_full.argmax(dim=-1)

        if target_mask is None:
            target_mask = torch.ones(x.size(0), dtype=torch.bool, device=device)
        correct = target_mask & (y_hat == y_true)
        nodes = correct.nonzero().squeeze(-1).tolist()
        if max_targets is not None and len(nodes) > max_targets:
            step = len(nodes) / max_targets
            nodes = [nodes[int(i * step)] for i in range(max_targets)]
        n_t = len(nodes)
        if n_t == 0:
            raise ValueError("no correctly-classified target nodes")
        nodes_t = torch.tensor(nodes, device=device)
        y_nodes = y_hat[nodes_t]

        counts = local_edge_counts(x, edge_index_dict, nodes, self.num_layers)  # (n_t, K) CPU
        total_local = counts.sum(dim=1).clamp(min=1).float()

        marg_full = prediction_margin(probs_full[nodes_t], y_nodes).cpu()

        # ---- exact search by increasing subset size ----------------------
        # Selection key is the lexicographic cost (|S|, edge fraction, margin):
        # within each size level ALL subsets compete per node (edge fraction,
        # then most negative margin = strongest flip), and only then the level
        # is finalized.  (The previous implementation gated feasibility with
        # `& ~solved`, which froze every node at the FIRST feasible subset of
        # its minimal size and made the tie-break dead code.)
        best_S: List[Optional[List[int]]] = [None] * n_t
        best_edge_cost = torch.full((n_t,), float("inf"))
        best_margin = torch.full((n_t,), float("inf"))
        solved = torch.zeros(n_t, dtype=torch.bool)
        flip_rate_per_subset: Dict[str, float] = {}

        for k in range(0, K + 1):
            subsets_k = list(combinations(range(K), k))
            cand_S: List[Optional[List[int]]] = [None] * n_t
            cand_edge_cost = torch.full((n_t,), float("inf"))
            cand_margin = torch.full((n_t,), float("inf"))
            cand_any = torch.zeros(n_t, dtype=torch.bool)
            for subset in subsets_k:
                masks = {str(r): torch.ones(edge_index_dict[str(r)].size(1), device=device)
                         for r in range(K)}
                for r in subset:
                    masks[str(r)] = torch.zeros(edge_index_dict[str(r)].size(1), device=device)
                logits_S = self.backbone(x, edge_index_dict, edge_mask_dict=masks)
                probs_S = F.softmax(logits_S, dim=-1)[nodes_t]
                marg_S = prediction_margin(probs_S, y_nodes).cpu()
                feas = (marg_S <= -self.kappa) & ~solved
                key = ",".join(map(str, subset)) if subset else "EMPTY"
                flip_rate_per_subset[key] = float((marg_S <= -self.kappa).float().mean().item())
                if feas.any():
                    if not subset:
                        edge_cost = torch.zeros(n_t)
                    else:
                        edge_cost = counts[:, list(subset)].sum(dim=1).float() / total_local
                    idx = feas.nonzero().squeeze(-1)
                    if select_mode == "cost_type":
                        # only |S| matters: the first feasible subset of this
                        # (minimal) size wins (combinations order)
                        better = ~cand_any[idx]
                    elif select_mode == "cost_edge":
                        # only the local edge fraction matters (tie: first found)
                        better = edge_cost[idx] < cand_edge_cost[idx]
                    else:  # lexicographic: (|S|, edge cost, margin)
                        better = (edge_cost[idx] < cand_edge_cost[idx]) | (
                            (edge_cost[idx] == cand_edge_cost[idx]) & (marg_S[idx] < cand_margin[idx])
                        )
                    upd = idx[better]
                    for i in upd.tolist():
                        cand_S[i] = list(subset)
                        cand_edge_cost[i] = edge_cost[i]
                        cand_margin[i] = marg_S[i]
                    cand_any[upd] = True
            # finalize this size level: adopt candidates of still-unsolved nodes
            for i in range(n_t):
                if not solved[i] and cand_any[i]:
                    best_S[i] = cand_S[i]
                    best_edge_cost[i] = cand_edge_cost[i]
                    best_margin[i] = cand_margin[i]
                    solved[i] = True
            if verbose:
                print(f"[search] |S|={k}: adopted {int(cand_any.sum().item())} new nodes "
                      f"(cumulative solved {int(solved.sum().item())}/{n_t})")

        # ---- keep-only (sufficiency) forwards ----------------------------
        distinct_S = {tuple(s) for s in best_S if s is not None}
        y_only_by_S: Dict[Tuple[int, ...], Tensor] = {}
        for S in distinct_S:
            masks = {str(r): torch.zeros(edge_index_dict[str(r)].size(1), device=device)
                     for r in range(K)}
            for r in S:
                masks[str(r)] = torch.ones(edge_index_dict[str(r)].size(1), device=device)
            y_only_by_S[S] = self.backbone(x, edge_index_dict, edge_mask_dict=masks).argmax(dim=-1)

        records = []
        for i in range(n_t):
            S = best_S[i]
            records.append({
                "node": nodes[i],
                "y_hat": int(y_nodes[i]),
                "margin_full": float(marg_full[i]),
                "S": S if S is not None else None,
                "success": bool(solved[i]),
                "final_margin": float(best_margin[i]) if S is not None else float(marg_full[i]),
                "cost_type": len(S) if S is not None else None,
                "cost_edge": float(best_edge_cost[i]) if S is not None else None,
                "sf": float(y_only_by_S[tuple(S)][nodes[i]] == y_nodes[i]) if S is not None else None,
            })

        feasible = [r for r in records if r["success"]]
        coverage = len(feasible) / max(n_t, 1)
        summary = {
            "n_targets": n_t,
            "n_feasible": len(feasible),
            "coverage": coverage,
            "nf": coverage,  # NF_v = 1[flip] by construction of the search
            "sf": float(np.mean([r["sf"] for r in feasible])) if feasible else None,
            "mean_cost_type": float(np.mean([r["cost_type"] for r in feasible])) if feasible else None,
            "mean_cost_edge": float(np.mean([r["cost_edge"] for r in feasible])) if feasible else None,
            "mean_final_margin": float(np.mean([r["final_margin"] for r in feasible])) if feasible else None,
            "flip_rate_per_subset": flip_rate_per_subset,
        }
        return {"records": records, **summary}


def dataset_level_s_star(flip_rate_per_subset: Dict[str, float], K: int, tau: float = 0.9) -> List[int]:
    """Reconstruct the v1 dataset-level minimal set S* (Eq. 5) from the
    per-subset flip rates measured by :meth:`InstanceRelationExplainer.explain`.
    Used only for the dataset-level vs instance-level ablation."""
    full = flip_rate_per_subset.get(",".join(map(str, range(K))), 0.0)
    target = tau * full if full > 0 else 0.0
    for k in range(K + 1):
        for subset in combinations(range(K), k):
            key = ",".join(map(str, subset)) if subset else "EMPTY"
            if flip_rate_per_subset.get(key, 0.0) >= target:
                return list(subset)
    return list(range(K))


def relation_lipschitz_bounds(
    backbone, x: Tensor, edge_index_dict: Dict[str, Tensor],
    nodes: List[int], num_layers: int, rel_map, adj,
) -> Dict[int, List[float]]:
    """Per-node, per-relation upper bounds on the margin drop.

    Delta_r(v) = sum_{e in E_r cap N_L(v)} |d M_v / d m_e|, computed by a
    backward pass on v's exact local computation subgraph at the FULL graph
    point (m = 1).  Multiplying by ``calib`` yields the branch-and-bound
    bound: M_v(G \\ (S cup T)) >= M_v(G \\ S) - calib * sum_{r in T} Delta_r(v),
    which is a first-order (empirically calibrated) Lipschitz bound -- NOT a
    proof of global validity; soundness is measured against exact search at
    K <= 10 in the experiment.
    """
    from model.hierarchical_explainer import local_subgraph
    bounds = {}
    for v in nodes:
        x_loc, eid_loc, v_loc, rows = local_subgraph(x, edge_index_dict, rel_map, adj, v, num_layers)
        masks = {str(r): torch.ones(eid_loc[str(r)].size(1), device=x.device, requires_grad=True)
                 for r in range(len(eid_loc))}
        logits = backbone(x_loc, eid_loc, edge_mask_dict=masks)
        probs = torch.softmax(logits, dim=-1)
        y_v = int(logits.argmax(dim=-1)[v_loc].item())
        margin = prediction_margin(probs[v_loc:v_loc + 1], torch.tensor([y_v], device=x.device))[0]
        grads = torch.autograd.grad(margin, list(masks.values()), create_graph=False)
        b = []
        for r in range(len(eid_loc)):
            g = grads[r].detach().abs().sum().item() if eid_loc[str(r)].size(1) > 0 else 0.0
            b.append(g)
        bounds[v] = b
    return bounds


def explain_branch_and_bound(
    backbone, x: Tensor, edge_index_dict: Dict[str, Tensor],
    y_true: Tensor, target_mask: Tensor, num_layers: int,
    kappa: float = 0.0, max_targets: Optional[int] = None,
    calib: float = 2.0, budget: int = 500,
) -> Dict:
    """Instance-level relation search by budgeted branch-and-bound with the
    calibrated Lipschitz bound (large-K approximation; exact under the bound's
    validity, but bounded to ``budget`` local forward evaluations per node --
    when the budget is exhausted the best solution found so far is returned
    and the instance is marked as budget-limited).

    Returns per-node {S, success, margin, n_local_forwards, budget_limited}
    records plus aggregate statistics and wall-clock time.
    """
    from model.hierarchical_explainer import build_structures, local_subgraph
    device = x.device
    K = len(edge_index_dict)
    t0 = time.time()
    with torch.no_grad():
        y_hat = backbone(x, edge_index_dict).argmax(dim=-1)
    correct = (target_mask & (y_hat == y_true)).nonzero().squeeze(-1).tolist()
    if max_targets is not None and len(correct) > max_targets:
        step = len(correct) / max_targets
        correct = [correct[int(i * step)] for i in range(max_targets)]

    rel_map, adj = build_structures(edge_index_dict, x.size(0))
    bounds = relation_lipschitz_bounds(backbone, x, edge_index_dict, correct,
                                       num_layers, rel_map, adj)

    records = []
    n_fwd_total = 0
    n_pruned_total = 0
    for v in correct:
        x_loc, eid_loc, v_loc, rows = local_subgraph(x, edge_index_dict, rel_map, adj, v, num_layers)
        y_v = int(y_hat[v].item())

        def margin_of(S: List[int]) -> float:
            masks = {str(r): torch.ones(eid_loc[str(r)].size(1), device=device) for r in range(K)}
            for r in S:
                masks[str(r)][:] = 0.0
            with torch.no_grad():
                logits = backbone(x_loc, eid_loc, edge_mask_dict=masks)
                probs = torch.softmax(logits, dim=-1)
            return float(prediction_margin(probs[v_loc:v_loc + 1],
                                           torch.tensor([y_v], device=device))[0].item())

        order = sorted(range(K), key=lambda r: -bounds[v][r])
        best, best_m, n_fwd, n_pruned, limited = None, float("inf"), 0, 0, False

        def dfs(S: List[int], idx: int):
            nonlocal best, best_m, n_fwd, n_pruned, limited
            if n_fwd >= budget:
                limited = True
                return
            m = margin_of(S)
            n_fwd += 1
            if m <= -kappa:
                if best is None or len(S) < len(best) or (len(S) == len(best) and m < best_m):
                    best, best_m = list(S), m
                return  # pruning by optimality of size
            # remaining potential: relations after idx in the order
            rem_drop = calib * sum(bounds[v][order[j]] for j in range(idx, K))
            if m - rem_drop > -kappa:
                n_pruned += 1
                return
            if idx >= K:
                return
            dfs(S + [order[idx]], idx + 1)  # include
            dfs(S, idx + 1)                  # exclude

        dfs([], 0)
        records.append({
            "node": v, "S": best, "success": best is not None,
            "final_margin": best_m if best is not None else margin_of([]),
            "n_local_forwards": n_fwd,
            "budget_limited": limited,
        })
        n_fwd_total += n_fwd
        n_pruned_total += n_pruned
    t = time.time() - t0
    n_succ = sum(1 for r in records if r["success"])
    n_limited = sum(1 for r in records if r.get("budget_limited"))
    summary = {
        "n_targets": len(records),
        "coverage": n_succ / max(len(records), 1),
        "time_s": t,
        "n_local_forwards": n_fwd_total,
        "n_pruned": n_pruned_total,
        "n_budget_limited": n_limited,
    }
    return {"records": records, "summary": summary}
