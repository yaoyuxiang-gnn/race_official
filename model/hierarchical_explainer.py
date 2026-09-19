"""Hierarchical relation-to-edge counterfactual explanation (RACE-v2, B4).

For every target node v the explanation is built in three phases:

  Phase 1 (relation level): the exact instance-level search of B1 finds the
      minimal relation set S_v* whose deletion flips v (reuses
      :class:`model.instance_relation_explainer.InstanceRelationExplainer`).
  Phase 2 (edge level, INSIDE S_v*): starting from the graph with the relations
      S_v* deleted, edges of S_v* in v's receptive field are RESTORED one at a
      time in descending keep-importance order (gradient saliency of the
      original prediction); a restoration is accepted iff v remains flipped,
      and full passes repeat until a complete pass changes nothing -- the
      final deleted set carries a single-edge-restoration irreducibility
      certificate ('irreducible'), or 'verified_only' when the per-node
      forward budget (scan_budget) is exhausted first.  Flip verification
      uses the EXACT local computation subgraph of v (backward ball of radius
      L with renumbered nodes) and the unified margin predicate, so each check
      costs a tiny forward instead of a full-graph pass.
  Phase 3 (fallback): nodes for which NO relation set flips the prediction keep
      the flat verified edge explanation (CF2-hetero mask), so the overall
      success rate is never below the flat baseline.

  relation_phase=False runs the FLAT verified variant (B3 relation switch):
      no relation search; greedy batch deletion along saliency until flip,
      then the same single-edge scan with the same budget on the same nodes.

Per-node records include success, relation cost |S_v*|, local edge cost
(deleted local edges / receptive-field edges), final margin, and the number of
local forwards; the orchestrator additionally times every phase, so the method
can be compared with flat baselines on success rate, edge cost, stability and
speed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from model.instance_relation_explainer import InstanceRelationExplainer, prediction_margin


def build_structures(edge_index_dict: Dict[str, Tensor], n_nodes: int):
    """Precompute per-relation dst -> [(src, global_edge_idx)] maps and the
    union backward adjacency (both as plain python lists, built ONCE per graph)."""
    rel_map: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
    for key, ei in edge_index_dict.items():
        src, dst = ei[0].tolist(), ei[1].tolist()
        m: Dict[int, List[Tuple[int, int]]] = {}
        for gi, (u, w) in enumerate(zip(src, dst)):
            m.setdefault(w, []).append((u, gi))
        rel_map[key] = m
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for key, ei in edge_index_dict.items():
        src, dst = ei[0].tolist(), ei[1].tolist()
        for u, w in zip(src, dst):
            adj[w].append(u)
    return rel_map, adj


def local_subgraph(
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    rel_map: Dict[str, Dict[int, List[Tuple[int, int]]]],
    adj: List[List[int]],
    v: int,
    num_layers: int,
) -> Tuple[Tensor, Dict[str, Tensor], int, Dict[str, List[Tuple[int, int, int]]]]:
    """Exact computation subgraph of node v.

    Returns (x_loc, eid_loc, v_loc, rows) where ``rows[str(r)]`` is a list of
    (local_src, local_dst, global_edge_idx) for every r-edge inside the
    subgraph, in the same order as the columns of ``eid_loc[str(r)]``.
    """
    L = num_layers
    ball: Set[int] = {v}
    frontier = {v}
    ball_prev = {v}
    for d in range(L):
        nxt = set()
        for w in frontier:
            for u in adj[w]:
                if u not in ball:
                    nxt.add(u)
                    ball.add(u)
        if d == L - 2:
            ball_prev = set(ball)
        frontier = nxt
        if not frontier:
            break
    if L == 1:
        ball_prev = {v}

    nodes = sorted(ball)
    node_idx = {n: i for i, n in enumerate(nodes)}
    eid_loc: Dict[str, Tensor] = {}
    rows: Dict[str, List[Tuple[int, int, int]]] = {}
    dev = x.device
    for key, ei in edge_index_dict.items():
        sel: List[Tuple[int, int, int]] = []
        for w in ball_prev:
            for (u, gi) in rel_map[key].get(w, []):
                if u in ball:
                    sel.append((u, w, gi))
        if sel:
            rows[key] = [(node_idx[u], node_idx[w], gi) for u, w, gi in sel]
            eid_loc[key] = torch.tensor([[node_idx[u], node_idx[w]] for u, w, _ in sel],
                                        dtype=torch.long, device=dev).t().contiguous()
        else:
            rows[key] = []
            eid_loc[key] = torch.zeros((2, 0), dtype=torch.long, device=dev)
    x_loc = x[torch.tensor(nodes, device=dev)]
    return x_loc, eid_loc, node_idx[v], rows


def _flip_check(backbone, x_loc, eid_loc, v_loc, keep, y_v, kappa, device) -> bool:
    """Unified flip predicate (advisor step 1): the margin constraint
    M_v <= -kappa, ties counting against the original class -- identical to
    the relation search's feasibility test, NOT an argmax comparison."""
    with torch.no_grad():
        logits = backbone(x_loc, eid_loc, edge_mask_dict=keep)
        probs = F.softmax(logits, dim=-1)
    y_t = torch.tensor([y_v], device=device)
    return bool((prediction_margin(probs[v_loc:v_loc + 1], y_t)[0].item()) <= -kappa)


def _single_edge_restore_scan(
    backbone, x_loc, eid_loc, v_loc, keep, cand, y_v, kappa, scan_budget, device,
    assume_flipped: bool = False, n_fwd_start: int = 0,
):
    """Backward single-edge restoration scan over ``cand`` (descending score).

    ``keep`` must currently delete every cand edge.  Edges are restored one at
    a time in cand order; a restoration is accepted iff v stays flipped.  Full
    passes repeat until a complete pass changes nothing -> the remaining
    deleted set is *irreducible w.r.t. single-edge restoration* (status
    'irreducible').  If scan_budget forwards are exhausted first, the status
    is 'verified_only' (the flip is verified; irreducibility NOT certified).

    Returns (n_deleted, n_forwards, status, trace); trace lists
    (n_forwards, n_deleted) checkpoints after every accepted restoration, for
    budget-curve reconstruction (B4).
    """
    def flipped() -> bool:
        return _flip_check(backbone, x_loc, eid_loc, v_loc, keep, y_v, kappa, device)

    n_fwd = n_fwd_start
    if not assume_flipped:
        if not flipped():
            n_fwd += 1
            return len(cand), n_fwd, "no_flip", [(n_fwd, len(cand))]
        n_fwd += 1
    deleted = set(range(len(cand)))
    trace = [(n_fwd, len(deleted))]
    status = "verified_only"
    while n_fwd < scan_budget:
        changed = False
        for i in sorted(deleted):  # cand is saliency-sorted descending
            if n_fwd >= scan_budget:
                break
            r, j, _sc, _gi = cand[i]
            keep[str(r)][j] = 1.0
            n_fwd += 1
            if flipped():
                deleted.discard(i)
                changed = True
                trace.append((n_fwd, len(deleted)))
            else:
                keep[str(r)][j] = 0.0
        if n_fwd >= scan_budget:
            break
        if not changed:
            status = "irreducible"
            break
    return len(deleted), n_fwd, status, trace


def _prune_node(
    backbone,
    x_loc: Tensor,
    eid_loc: Dict[str, Tensor],
    v_loc: int,
    rows: Dict[str, List[Tuple[int, int, int]]],
    S: List[int],
    scores_cpu: Dict[str, List[float]],
    y_v: int,
    scan_budget: int,
    kappa: float,
    device,
):
    """B2 hierarchical edge refinement inside S_v*: start from deleting every
    local edge of the relations S, then run the single-edge restoration scan.

    Returns (final_deleted_local, total_local, success, status, margin,
    n_forwards, deleted_global_ids, trace).  status: 'irreducible' (certified
    single-edge irreducible), 'verified_only' (flip verified, scan budget
    exhausted), or 'no_flip'.
    """
    K = len(eid_loc)
    cand = []  # (relation, position in rows/eid, score, global_edge_id)
    for r in S:
        key = str(r)
        sc = scores_cpu[key]
        for j, (ls, ld, gi) in enumerate(rows[key]):
            cand.append((r, j, sc[gi], gi))
    cand.sort(key=lambda t: -t[2])

    total_local = sum(len(rows[str(r)]) for r in range(K))
    keep = {str(r): torch.ones(eid_loc[str(r)].size(1), device=device)
            for r in range(K)}
    for (r, j, _sc, _gi) in cand:
        keep[str(r)][j] = 0.0

    if total_local == 0:
        return 0, 0, False, "no_flip", 0.0, 1, [], [(1, 0)]

    n_del, n_fwd, status, trace = _single_edge_restore_scan(
        backbone, x_loc, eid_loc, v_loc, keep, cand, y_v, kappa, scan_budget, device)
    if status == "no_flip":
        return len(cand), total_local, False, "no_flip", 0.0, n_fwd, \
            [gi for _, _, _, gi in cand], trace

    with torch.no_grad():
        logits = backbone(x_loc, eid_loc, edge_mask_dict=keep)
        probs = F.softmax(logits, dim=-1)
    n_fwd += 1
    y_t = torch.tensor([y_v], device=device)
    margin = float(prediction_margin(probs[v_loc:v_loc + 1], y_t)[0].item())
    del_ids = [gi for (r, j, _sc, gi) in cand if float(keep[str(r)][j]) < 0.5]
    return n_del, total_local, True, status, margin, n_fwd, del_ids, trace


def _flat_verified_node(
    backbone, x_loc, eid_loc, v_loc, rows, scores_cpu, y_v, kappa,
    del_batches, scan_budget, device,
):
    """B3 flat verified pipeline (relation phase OFF).

    Candidates = every local edge (any relation) in descending saliency.
    Phase A: greedy deletion in ``del_batches`` batches along the order until
    v flips (one forward per batch trial).  Phase B: the SAME single-edge
    restoration scan as the hierarchical pipeline, with the same scan_budget.

    Returns (n_deleted, total_local, success, status, n_forwards, trace).
    """
    K = len(eid_loc)
    cand = []
    for r in range(K):
        key = str(r)
        sc = scores_cpu[key]
        for j, (ls, ld, gi) in enumerate(rows[key]):
            cand.append((r, j, sc[gi], gi))
    cand.sort(key=lambda t: -t[2])
    total_local = len(cand)
    keep = {str(r): torch.ones(eid_loc[str(r)].size(1), device=device)
            for r in range(K)}
    if total_local == 0:
        return 0, 0, False, "no_flip", 1, [(1, 0)]

    def flipped() -> bool:
        return _flip_check(backbone, x_loc, eid_loc, v_loc, keep, y_v, kappa, device)

    if flipped():  # empty intervention flips (only possible on margin ties)
        return 0, total_local, True, "irreducible", 1, [(1, 0)]

    batch = max(total_local // del_batches, 1)
    n_fwd = 1
    flip_n = None
    upto = 0
    for b in range(1, del_batches + 1):
        nxt = min(b * batch, total_local)
        for i in range(upto, nxt):
            r, j, _sc, _gi = cand[i]
            keep[str(r)][j] = 0.0
        upto = nxt
        n_fwd += 1
        if flipped():
            flip_n = upto
            break
    if flip_n is None:
        return 0, total_local, False, "no_flip", n_fwd, [(n_fwd, 0)]

    n_del, n_scan, status, trace = _single_edge_restore_scan(
        backbone, x_loc, eid_loc, v_loc, keep, cand[:flip_n], y_v, kappa,
        scan_budget, device, assume_flipped=True, n_fwd_start=n_fwd)
    return n_del, total_local, True, status, n_scan, trace


def hierarchical_explain(
    backbone,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y_true: Tensor,
    target_mask: Tensor,
    num_layers: int,
    kappa: float = 0.0,
    max_targets: Optional[int] = None,
    restore_batches: int = 8,      # legacy, unused: single-edge scan replaced batch pruning
    scan_budget: int = 128,        # per-node forward cap of the restoration scan
    relation_phase: bool = True,   # False -> flat verified pipeline (B3 relation switch)
    del_batches: int = 20,         # flat variant: greedy deletion batches
    edge_scores: Optional[Dict[str, Tensor]] = None,
    flat_masks: Optional[Dict[str, Tensor]] = None,
) -> Dict:
    """Run the hierarchical (or flat-verified) explanation.

    relation_phase=True  (B2): S_v* search -> delete relations -> single-edge
        restoration scan inside S_v* (status 'irreducible' when a full pass
        changes nothing, 'verified_only' when scan_budget is exhausted) ->
        CF2 fallback for relation-infeasible nodes.
    relation_phase=False (B3): skip the relation phase entirely; greedy batch
        deletion along saliency until flip -> the SAME single-edge scan with
        the SAME scan_budget on the SAME target nodes.
    """
    device = x.device
    K = len(edge_index_dict)

    with torch.no_grad():
        y_hat = backbone(x, edge_index_dict).argmax(dim=-1)

    t_rel = 0.0
    rel_records: List[Dict] = []
    n_forwards_rel = 0
    if relation_phase:
        t0 = time.time()
        rel_exp = InstanceRelationExplainer(backbone, K, num_layers=num_layers, kappa=kappa)
        rel_out = rel_exp.explain(x, edge_index_dict, y_true, target_mask=target_mask,
                                  max_targets=max_targets)
        t_rel = time.time() - t0
        rel_records = rel_out["records"]
        n_forwards_rel = 2 ** (K + 1) + 1
    else:
        # replicate the explainer's deterministic target selection (same nodes
        # as the relation_phase variant would pick)
        correct = target_mask & (y_hat == y_true)
        nodes_all = correct.nonzero().squeeze(-1).tolist()
        if max_targets:
            step = len(nodes_all) / max_targets
            nodes_all = [nodes_all[int(i * step)] for i in range(max_targets)]
        rel_records = [{"node": v, "success": None} for v in nodes_all]

    # ---- edge keep-importance (default: gradient saliency) ---------------
    t_s = time.time()
    if edge_scores is None:
        from benchmarks.baselines import gradient_edge_scores
        tgt = target_mask & (y_hat == y_true)
        edge_scores = gradient_edge_scores(backbone, x, edge_index_dict, y_hat, tgt,
                                           method="saliency")
    t_s = time.time() - t_s
    scores_cpu = {key: es.cpu().tolist() for key, es in edge_scores.items()}

    # ---- fallback flip status (flat mask) --------------------------------
    flat_flip: Dict[int, bool] = {}
    if flat_masks is not None and relation_phase:
        with torch.no_grad():
            probs_full = F.softmax(backbone(x, edge_index_dict), dim=-1)
            probs_flat = F.softmax(
                backbone(x, edge_index_dict, edge_mask_dict=flat_masks), dim=-1)
        y_full_cpu = y_hat.cpu()
        for rec in rel_records:
            v = rec["node"]
            yv = int(y_full_cpu[v].item())
            marg = float(prediction_margin(
                probs_flat[v:v + 1], torch.tensor([yv], device=device))[0].item())
            flat_flip[v] = marg <= -kappa

    # ---- phase 2: per-node verified refinement ---------------------------
    rel_map, adj = build_structures(edge_index_dict, x.size(0))
    t_p = time.time()
    n_forwards_local = 0
    records = []
    for rec in rel_records:
        v = rec["node"]
        if relation_phase and not rec["success"]:
            records.append({
                "node": v, "S": None, "success": bool(flat_flip.get(v, False)),
                "fallback": True, "cost_type": None, "local_deleted": None,
                "local_total": None, "edge_cost": None,
                "final_margin": rec["final_margin"],
                "status": "fallback", "trace": None, "phase": "fallback",
            })
            continue
        x_loc, eid_loc, v_loc, rows = local_subgraph(x, edge_index_dict, rel_map, adj,
                                                     v, num_layers)
        y_v = int(y_hat[v].item())
        if relation_phase:
            S = rec["S"]
            n_del, total_local, ok, status, margin, n_fwd, del_ids, trace = _prune_node(
                backbone, x_loc, eid_loc, v_loc, rows, S, scores_cpu, y_v,
                scan_budget, kappa, device,
            )
            n_forwards_local += n_fwd
            records.append({
                "node": v, "S": S, "success": bool(ok), "fallback": False,
                "cost_type": len(S), "local_deleted": n_del,
                "local_total": total_local,
                "edge_cost": (n_del / total_local) if total_local > 0 else 0.0,
                "final_margin": margin, "deleted_edges": del_ids,
                "n_local_forwards": n_fwd, "status": status, "trace": trace,
                "phase": "hier",
            })
        else:
            n_del, total_local, ok, status, n_fwd, trace = _flat_verified_node(
                backbone, x_loc, eid_loc, v_loc, rows, scores_cpu, y_v, kappa,
                del_batches, scan_budget, device,
            )
            n_forwards_local += n_fwd
            records.append({
                "node": v, "S": None, "success": bool(ok), "fallback": False,
                "cost_type": None, "local_deleted": n_del,
                "local_total": total_local,
                "edge_cost": (n_del / total_local) if total_local > 0 else 0.0,
                "final_margin": None, "deleted_edges": None,
                "n_local_forwards": n_fwd, "status": status, "trace": trace,
                "phase": "flat",
            })
    t_p = time.time() - t_p

    n_targets = len(records)
    n_success = sum(1 for r in records if r["success"])
    feas = [r for r in records if r["phase"] in ("hier", "flat") and r["success"]]
    summary = {
        "n_targets": n_targets,
        "n_feasible_rel": sum(1 for r in rel_records if r["success"]) if relation_phase else 0,
        "n_success": n_success,
        "csr": n_success / max(n_targets, 1),
        "mean_edge_cost_feasible": (
            float(sum(r["edge_cost"] for r in feas) / len(feas)) if feas else None),
        "mean_cost_type_feasible": (
            float(sum(r["cost_type"] for r in records
                      if r["phase"] == "hier" and r["success"] and r["cost_type"] is not None)
                  / max(1, sum(1 for r in records
                               if r["phase"] == "hier" and r["success"] and r["cost_type"] is not None)))
            if any(r["phase"] == "hier" and r["success"] and r["cost_type"] is not None
                   for r in records) else None),
        "n_irreducible": sum(1 for r in records if r.get("status") == "irreducible"),
        "n_verified_only": sum(1 for r in records if r.get("status") == "verified_only"),
        "time_rel_s": t_rel,
        "time_saliency_s": t_s,
        "time_prune_s": t_p,
        "n_forwards_rel": n_forwards_rel,
        "n_forwards_local": n_forwards_local,
    }
    return {"records": records, "summary": summary, "rel_records": rel_records}
