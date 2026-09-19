"""Pairwise baselines on the collapsed (relation-flattened) graph.

The three baselines (GNNExplainer / CF-GNNExplainer / PNS) all operate on a
single-relation graph obtained by unioning every relation, and differ only in
the mask objective.  :func:`collapse` flattens the relations and
:func:`optimize_edge_masks` learns per-edge keep-masks under the chosen
objective.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from model.counterfactual_explainer import CounterfactualExplainer


def collapse(edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Union all relations into a single (collapsed) relation id ``"0"``."""
    eis = [ei for ei in edge_index_dict.values()]
    if not eis:
        return {"0": torch.zeros((2, 0), dtype=torch.long)}
    return {"0": torch.cat(eis, dim=1)}


def _saliency_prior(backbone, x, edge_index_dict, y_orig, tgt) -> Dict[str, Tensor]:
    """Soft keep-template from gradient saliency (C2Explainer customization
    signal): edges with higher |d p(y)/d m| get a prior closer to 1 (keep)."""
    from model.instance_relation_explainer import prediction_margin  # noqa: F401
    dev = x.device
    masks = {k: torch.ones(ei.size(1), device=dev, requires_grad=True)
             for k, ei in edge_index_dict.items()}
    logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
    probs = F.softmax(logits, dim=-1)
    p = probs[tgt].gather(1, y_orig[tgt].unsqueeze(-1)).sum()
    grads = torch.autograd.grad(p, list(masks.values()), create_graph=False)
    out = {}
    for k, g in zip(masks.keys(), grads):
        sal = g.detach().abs()
        # rank-normalize to [0,1]: highest saliency -> prior 1
        ranks = torch.argsort(torch.argsort(sal)).float()
        out[k] = ranks / max(ranks.numel() - 1, 1)
    return out


def optimize_edge_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    num_relations: int,
    mode: str,  # "factual" | "counterfactual" | "pns" | "rc" | "cf2" | "nseg" | "c2"
    steps: int = 150,
    lr: float = 0.1,
    sparsity: float = 0.05,
    artifact_dir: Optional[str] = None,
    gumbel_tau: float = 1.0,
    target_mask: Optional[Tensor] = None,
    c2_coef: float = 0.5,
) -> Dict[str, Tensor]:
    """Learn per-edge keep-masks under the specified objective.

    Returns hard keep-masks (1 = keep edge) per relation id (str).  When
    ``artifact_dir`` is given, the per-step loss is appended to
    ``artifact_dir/train_log.json`` (spec.md Step 1).  ``target_mask`` restricts
    the objective to the nodes being explained (default: all nodes).
    """
    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        log_path = os.path.join(artifact_dir, "train_log.json")
    exp = CounterfactualExplainer(backbone, num_relations, gumbel_tau=gumbel_tau, hard=False, init_logit=3.0)
    exp.reset(edge_index_dict)
    with torch.no_grad():
        y_orig = backbone(x, edge_index_dict).argmax(dim=-1)
    if target_mask is None:
        tgt = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
    else:
        tgt = target_mask.bool().to(x.device)

    def _mean_target(v: Tensor) -> Tensor:
        return v[tgt].mean()

    opt = torch.optim.Adam(exp.mask_logits.parameters(), lr=lr)
    for step in range(1, steps + 1):
        opt.zero_grad()
        masks = exp.sample_masks(edge_index_dict)
        logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
        probs = F.softmax(logits, dim=-1)
        p_orig = probs.gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
        if mode == "factual":
            loss = -_mean_target(p_orig)
        elif mode == "counterfactual":
            loss = _mean_target(p_orig)
        elif mode == "pns":
            keep = {k: (v > 0.5).float() for k, v in masks.items()}
            remove = {k: 1.0 - v for k, v in keep.items()}
            logits_only = backbone(x, edge_index_dict, edge_mask_dict=remove)
            p_only = F.softmax(logits_only, dim=-1).gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
            loss = _mean_target(p_orig - p_only)
        elif mode == "rc":
            # RCExplainer (Bajaj et al., NeurIPS 2021): hinge loss that pushes
            # the best non-original class above the original class by a margin.
            p_other = 1.0 - (probs * F.one_hot(y_orig, probs.size(-1))).sum(-1)
            loss = _mean_target(torch.relu(0.1 - (p_other - p_orig)))
        elif mode == "cf2":
            # CF^2 (Tan et al., WWW 2022): counterfactual reasoning (flip term)
            # + factual reasoning (the *removed* edge set alone must not
            # preserve the prediction) + validity (hinge, as in RCExplainer).
            keep = {k: (v > 0.5).float() for k, v in masks.items()}
            remove = {k: 1.0 - v for k, v in keep.items()}
            logits_factual = backbone(x, edge_index_dict, edge_mask_dict=remove)
            p_factual = F.softmax(logits_factual, dim=-1).gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
            p_other = 1.0 - (probs * F.one_hot(y_orig, probs.size(-1))).sum(-1)
            validity = torch.relu(0.1 - (p_other - p_orig))
            loss = _mean_target(p_orig + p_factual + validity)
        elif mode == "nseg":
            # NSEG (Cai et al., Neural Networks 2025): necessity + sufficiency
            # lower-bound objective.  L_N pushes p(y) down on the graph with the
            # kept (explanation) edges REMOVED, L_S pushes p(y) up on the kept
            # subgraph alone:
            #   loss = mean p(y | G x (1-m)) - mean p(y | G x m).
            keep_h = {k: (v > 0.5).float() for k, v in masks.items()}
            remove_h = {k: 1.0 - v for k, v in keep_h.items()}
            logits_removed = backbone(x, edge_index_dict, edge_mask_dict=remove_h)
            p_removed = F.softmax(logits_removed, dim=-1).gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
            p_keep = F.softmax(logits, dim=-1).gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
            loss = _mean_target(p_removed - p_keep)
        elif mode == "c2":
            # C2Explainer (FAccT 2025): customizable mask-based counterfactual.
            # The customization signal is a prior keep-template from gradient
            # saliency (edges the user deems plausible); the objective is the
            # counterfactual flip plus a pull toward the prior:
            #   loss = p_orig + lambda_c * mean |m - prior|.
            prior = _saliency_prior(backbone, x, edge_index_dict, y_orig, tgt)
            dev = x.device
            pull = sum(((masks[k] - prior[k].to(dev)) ** 2).mean() for k in masks)
            loss = _mean_target(p_orig) + c2_coef * pull
        else:
            raise ValueError(mode)
        total = sum(int(m.numel()) for m in masks.values())
        removed = sum((1.0 - m).sum() for m in masks.values()) / max(total, 1)
        loss = loss + sparsity * removed
        loss.backward()
        opt.step()
        if artifact_dir is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": step, "loss": float(loss.item()), "mode": mode}) + "\n")
    return exp.hard_masks(edge_index_dict)


# ---------------------------------------------------------------------------
# RACE-v2 per-edge counterfactual (advisor Sec. 2.3): margin hinge + binary
# pressure, followed by greedy discrete verification and backward pruning.
# ---------------------------------------------------------------------------
def optimize_race_v2_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    num_relations: int,
    steps: int = 150,
    lr: float = 0.1,
    sparsity: float = 0.05,
    bin_coef: float = 0.2,
    kappa: float = 0.0,
    delete_batches: int = 40,
    prune_batches: int = 20,
    verify_sparsity: Optional[float] = None,
    rounds: int = 1,
    select_mode: str = "score",  # "score" | "max_csr"
    prune_rounds: int = 1,
    cont_mode: str = "margin",  # "margin" | "margin_weighted" | "counterfactual" | "cf2"
    target_mask: Optional[Tensor] = None,
    eval_mask: Optional[Tensor] = None,
    path_log: Optional[list] = None,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Learn per-edge keep-masks with a margin objective and DISCRETE validity.

    Continuous phase (same mask parameterisation as the v1 explainer):

        L_flip = mean( relu( M_v(G_m) + kappa ) )       margin hinge
        L_cost = mean( 1 - m )                          sparsity
        L_bin  = mean( m (1 - m) )                      binary pressure
        L      = L_flip + sparsity*L_cost + bin_coef*L_bin

    where M_v is the margin of the ORIGINAL predicted class (positive = still
    predicted).  ``target_mask`` restricts the flip loss to the nodes being
    explained (default: all nodes, protocol-identical to the v1 baselines).

    Discrete phase (deterministic, no Gumbel noise):

    1. Order every edge by its learned keep-score (ascending = delete first)
       and greedily delete in ``delete_batches`` batches, keeping the point
       that maximises ``satisfied_fraction - verify_sparsity * removed_frac``
       (default: the continuous-phase ``sparsity``).
    2. Backward-prune: restore deleted edges in ``prune_batches`` batches
       (most-confidently-kept first) while the satisfied count is preserved.

    ``rounds > 1`` warm-starts the mask logits from the pruned hard masks and
    repeats continuous + verify + prune, refining the deleted set.

    Returns the final hard keep-masks.  Unlike the soft-thresholded baselines,
    the reported CSR of these masks is verified on the discrete model.
    """
    from model.instance_relation_explainer import prediction_margin

    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        log_path = os.path.join(artifact_dir, "train_log.json")

    if verify_sparsity is None:
        verify_sparsity = sparsity

    exp = CounterfactualExplainer(backbone, num_relations, gumbel_tau=1.0, hard=False, init_logit=3.0)
    exp.reset(edge_index_dict)
    keys = [str(r) for r in range(num_relations)]
    with torch.no_grad():
        y_orig = backbone(x, edge_index_dict).argmax(dim=-1)
    if target_mask is None:
        tgt = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
    else:
        tgt = target_mask.bool().to(x.device)

    total_edges = sum(int(v.numel()) for v in exp.mask_logits.values())
    tgt_n = max(int(tgt.sum().item()), 1)
    if eval_mask is None:
        ev_mask = tgt
    else:
        ev_mask = eval_mask.bool().to(x.device)
    ev_n = max(int(ev_mask.sum().item()), 1)

    with torch.no_grad():
        probs_full = F.softmax(backbone(x, edge_index_dict), dim=-1)
    marg_full = prediction_margin(probs_full, y_orig)

    def _margins(keep: Dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            logits = backbone(x, edge_index_dict, edge_mask_dict=keep)
            probs = F.softmax(logits, dim=-1)
        return prediction_margin(probs, y_orig)

    def satisfied(keep: Dict[str, Tensor]) -> Tensor:
        return _margins(keep)[tgt] <= -kappa

    def eval_satisfied_frac(keep: Dict[str, Tensor]) -> float:
        return float((_margins(keep)[ev_mask] <= -kappa).float().mean().item())

    best_keep: Dict[str, Tensor] = {}
    for rnd in range(rounds):
        opt = torch.optim.Adam(exp.mask_logits.parameters(), lr=lr)
        for step in range(1, steps + 1):
            opt.zero_grad()
            masks = exp.sample_masks(edge_index_dict)
            logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
            probs = F.softmax(logits, dim=-1)
            margin = prediction_margin(probs, y_orig)
            p_orig = probs.gather(1, y_orig.unsqueeze(-1)).squeeze(-1)
            if cont_mode == "margin":
                flip_loss = torch.relu(margin[tgt] + kappa).mean()
            elif cont_mode == "margin_weighted":
                w = 1.0 / (marg_full[tgt] + 0.1)  # difficulty weighting
                flip_loss = (torch.relu(margin[tgt] + kappa) * w).mean()
            elif cont_mode == "counterfactual":
                flip_loss = p_orig[tgt].mean()
            elif cont_mode == "cf2":
                keep_h = {k: (v > 0.5).float() for k, v in masks.items()}
                remove_h = {k: 1.0 - v for k, v in keep_h.items()}
                logits_factual = backbone(x, edge_index_dict, edge_mask_dict=remove_h)
                p_factual = F.softmax(logits_factual, dim=-1).gather(
                    1, y_orig.unsqueeze(-1)).squeeze(-1)
                p_other = 1.0 - (probs * F.one_hot(y_orig, probs.size(-1))).sum(-1)
                validity = torch.relu(0.1 - (p_other - p_orig))
                flip_loss = (p_orig + p_factual + validity)[tgt].mean()
            else:
                raise ValueError(cont_mode)
            removed = sum((1.0 - m).sum() for m in masks.values()) / max(total_edges, 1)
            bin_loss = sum((m * (1.0 - m)).sum() for m in masks.values()) / max(total_edges, 1)
            loss = flip_loss + sparsity * removed + bin_coef * bin_loss
            loss.backward()
            opt.step()
            if artifact_dir is not None and rnd == rounds - 1:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": step, "round": rnd, "loss": float(loss.item()),
                                        "flip_loss": float(flip_loss.item()), "mode": "race_v2"}) + "\n")

        # ---- discrete verification ----------------------------------------
        with torch.no_grad():
            scores = {k: torch.sigmoid(v.detach()) for k, v in exp.mask_logits.items()}
        # offsets has K+1 entries: flat positions [offsets[r], offsets[r+1])
        # belong to relation keys[r].
        offsets = [0]
        for k in keys:
            offsets.append(offsets[-1] + scores[k].numel())
        total = offsets[-1]
        flat_scores = torch.cat([scores[k].flatten() for k in keys])
        order = torch.argsort(flat_scores)  # ascending keep-score: delete first

        def masks_with_n_del(n_del: int) -> Dict[str, Tensor]:
            keep = {k: torch.ones(scores[k].numel(), device=x.device) for k in keys}
            if n_del > 0:
                pos = order[:n_del].tolist()
                for p in pos:
                    r = num_relations - 1
                    for j in range(num_relations):
                        if p < offsets[j + 1]:
                            r = j
                            break
                    keep[keys[r]][p - offsets[r]] = 0.0
            return keep

        base_keep = masks_with_n_del(0)
        base_cnt = int(satisfied(base_keep).sum().item())
        best_score = base_cnt / tgt_n  # n_del = 0: cost term is zero
        best_keep, best_n_del = base_keep, 0
        best_csr = eval_satisfied_frac(base_keep)
        best_csr_keep, best_csr_n_del = base_keep, 0
        if path_log is not None and rnd == rounds - 1:
            path_log.append({"round": rnd, "removed": 0.0, "satisfied_frac": base_cnt / tgt_n,
                             "eval_csr": best_csr})
        batch = max(total // delete_batches, 1)
        n_del = 0
        while n_del < total:
            n_del = min(n_del + batch, total)
            keep = masks_with_n_del(n_del)
            cnt = int(satisfied(keep).sum().item())
            score = cnt / tgt_n - verify_sparsity * (n_del / total)
            if score > best_score:
                best_score, best_keep, best_n_del = score, keep, n_del
            csr_here = eval_satisfied_frac(keep)
            if csr_here > best_csr:
                best_csr, best_csr_keep, best_csr_n_del = csr_here, keep, n_del
            if path_log is not None and rnd == rounds - 1:
                path_log.append({"round": rnd, "removed": n_del / total,
                                 "satisfied_frac": cnt / tgt_n,
                                 "eval_csr": csr_here})

        if select_mode == "max_csr":
            best_keep, best_n_del = best_csr_keep, best_csr_n_del

        # ---- backward pruning (repeatable: restoring edges can also IMPROVE
        # validity under the non-monotone GNN, so sweep several times) -------
        if best_n_del > 0:
            deleted = order[:best_n_del]
            deleted_sorted = deleted[torch.argsort(flat_scores[deleted], descending=True)]
            prune_batch = max(deleted_sorted.numel() // prune_batches, 1)
            current_keep = {k: v.clone() for k, v in best_keep.items()}
            current_cnt = int(satisfied(current_keep).sum().item())
            for _ in range(prune_rounds):
                accepted = False
                for i in range(0, deleted_sorted.numel(), prune_batch):
                    pos = deleted_sorted[i:i + prune_batch].tolist()
                    candidate = {k: v.clone() for k, v in current_keep.items()}
                    for p in pos:
                        r = num_relations - 1
                        for j in range(num_relations):
                            if p < offsets[j + 1]:
                                r = j
                                break
                        candidate[keys[r]][p - offsets[r]] = 1.0
                    cnt = int(satisfied(candidate).sum().item())
                    if cnt >= current_cnt:  # validity preserved (or improved)
                        current_keep, current_cnt = candidate, cnt
                        accepted = True
                if not accepted:
                    break
            best_keep = current_keep

        if rnd < rounds - 1:
            # warm-start logits from the pruned hard masks
            with torch.no_grad():
                for k in keys:
                    exp.mask_logits[k].copy_(torch.where(best_keep[k] > 0.5, 3.0, -3.0))

    if artifact_dir is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"mode": "race_v2", "final_deleted": best_n_del,
                                "final_satisfied": int(satisfied(best_keep).sum().item())}) + "\n")
    return best_keep


# ---------------------------------------------------------------------------
# InduCE (Verma et al., TMLR 2024): inductive counterfactual explainer -- an
# amortized generator (PGExplainer-style MLP per relation) trained with the
# COUNTERFACTUAL objective, so the explanation generalizes to unseen graphs.
# ---------------------------------------------------------------------------
def optimize_induce_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    num_relations: int,
    steps: int = 200,
    lr: float = 0.05,
    sparsity: float = 0.05,
    entropy_coef: float = 0.5,
    temp0: float = 5.0,
    temp1: float = 1.0,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Inductive counterfactual masks from a parameterized generator."""
    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        log_path = os.path.join(artifact_dir, "train_log.json")
    dev = x.device
    with torch.no_grad():
        y_orig = backbone(x, edge_index_dict).argmax(dim=-1)
    generators, feats, edges_per_rel = {}, {}, {}
    for r in range(num_relations):
        key = str(r)
        ei = edge_index_dict[key]
        feats[key] = torch.cat([x[ei[0]], x[ei[1]]], dim=-1)
        edges_per_rel[key] = ei.size(1)
        generators[key] = torch.nn.Sequential(
            torch.nn.Linear(feats[key].size(-1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(dev)
    params = [p for g in generators.values() for p in g.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(1, steps + 1):
        temp = temp0 + (temp1 - temp0) * (step - 1) / max(steps - 1, 1)
        opt.zero_grad()
        masks, entropy = {}, 0.0
        for r in range(num_relations):
            key = str(r)
            logit = generators[key](feats[key]).squeeze(-1)
            u = torch.rand_like(logit).clamp(min=1e-8)
            g = -torch.log(-torch.log(u) + 1e-8)
            y = torch.sigmoid((logit + g) / temp)
            hard = (y > 0.5).float()
            y = (hard - y).detach() + y
            masks[key] = y
            p = torch.sigmoid(logit)
            entropy = entropy - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
        logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
        probs = F.softmax(logits, dim=-1)
        p_orig = probs.gather(1, y_orig.unsqueeze(-1)).squeeze(-1).mean()
        total = sum(edges_per_rel.values())
        removed = sum((1.0 - m).sum() for m in masks.values()) / max(total, 1)
        loss = p_orig + sparsity * removed + entropy_coef * entropy
        loss.backward()
        opt.step()
        if artifact_dir is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": step, "loss": float(loss.item()), "mode": "induce"}) + "\n")
    out = {}
    for r in range(num_relations):
        key = str(r)
        with torch.no_grad():
            logit = generators[key](feats[key]).squeeze(-1)
        out[key] = (torch.sigmoid(logit) > 0.5).float()
    return out


# ---------------------------------------------------------------------------
# PGExplainer: amortized factual explainer (Luo et al., NeurIPS 2020).
# ---------------------------------------------------------------------------
def optimize_pge_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    num_relations: int,
    steps: int = 200,
    lr: float = 0.05,
    sparsity: float = 0.05,
    entropy_coef: float = 0.5,
    temp0: float = 5.0,
    temp1: float = 1.0,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Learn a *parameterized* edge generator (PGExplainer-style).

    For every relation the generator is a two-layer MLP over the concatenated
    source/destination features ``[x_src || x_dst]`` whose sigmoid output is the
    keep probability of the edge; a straight-through Gumbel-Sigmoid keeps the
    mask differentiable and the temperature is annealed from ``temp0`` to
    ``temp1``.  The objective is the factual loss (preserve the prediction)
    plus a size regularizer and an entropy regularizer, as in the original
    PGExplainer.  Returns hard keep-masks per relation.
    """
    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        log_path = os.path.join(artifact_dir, "train_log.json")
    dev = x.device
    with torch.no_grad():
        y_orig = backbone(x, edge_index_dict).argmax(dim=-1)
    # One generator MLP per relation (PGExplainer uses one shared network; the
    # per-relation split keeps the masks typed, analogous to our framework).
    generators, feats, edges_per_rel = {}, {}, {}
    for r in range(num_relations):
        key = str(r)
        ei = edge_index_dict[key]
        feats[key] = torch.cat([x[ei[0]], x[ei[1]]], dim=-1)  # (E_r, 2d)
        edges_per_rel[key] = ei.size(1)
        generators[key] = torch.nn.Sequential(
            torch.nn.Linear(feats[key].size(-1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(dev)
    params = [p for g in generators.values() for p in g.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(1, steps + 1):
        temp = temp0 + (temp1 - temp0) * (step - 1) / max(steps - 1, 1)
        opt.zero_grad()
        masks, entropy = {}, 0.0
        for r in range(num_relations):
            key = str(r)
            logit = generators[key](feats[key]).squeeze(-1)
            u = torch.rand_like(logit).clamp(min=1e-8)
            g = -torch.log(-torch.log(u) + 1e-8)
            y = torch.sigmoid((logit + g) / temp)
            hard = (y > 0.5).float()
            y = (hard - y).detach() + y
            masks[key] = y
            p = torch.sigmoid(logit)
            entropy = entropy - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
        logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
        probs = F.softmax(logits, dim=-1)
        p_orig = probs.gather(1, y_orig.unsqueeze(-1)).squeeze(-1).mean()
        total = sum(edges_per_rel.values())
        removed = sum((1.0 - m).sum() for m in masks.values()) / max(total, 1)
        loss = -p_orig + sparsity * removed + entropy_coef * entropy
        loss.backward()
        opt.step()
        if artifact_dir is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": step, "loss": float(loss.item()), "mode": "pge"}) + "\n")
    out = {}
    for r in range(num_relations):
        key = str(r)
        with torch.no_grad():
            logit = generators[key](feats[key]).squeeze(-1)
        out[key] = (torch.sigmoid(logit) > 0.5).float()
    return out


# ---------------------------------------------------------------------------
# MEG: genetic search over the global binary edge mask
# (Numeroso & Bacciu, IJCNN 2021).
# ---------------------------------------------------------------------------
def _mask_to_vector(masks: Dict[str, Tensor]) -> Tensor:
    return torch.cat([m.flatten() for m in masks.values()])


def _vector_to_masks(vec: Tensor, edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    out, offset = {}, 0
    for key, ei in edge_index_dict.items():
        n = ei.size(1)
        out[key] = vec[offset:offset + n].reshape(-1)
        offset += n
    return out


def _meg_fitness(backbone, x, edge_index_dict, y_orig, masks, sparsity) -> float:
    with torch.no_grad():
        logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
        y_cf = logits.argmax(dim=-1)
    flip = float((y_cf != y_orig).float().mean().item())
    total = sum(int(m.numel()) for m in masks.values())
    removed = float(sum(int((m < 0.5).sum()) for m in masks.values())) / max(total, 1)
    return flip - sparsity * removed


def optimize_meg_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    num_relations: int,
    pop_size: int = 16,
    generations: int = 25,
    keep_frac: float = 0.9,
    mutate_prob: float = 0.02,
    sparsity: float = 0.05,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """MEG-style counterfactual search: evolve a population of binary edge
    keep-masks by tournament selection, uniform crossover and bit-flip
    mutation, maximizing flip rate minus a size penalty."""
    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
        log_path = os.path.join(artifact_dir, "train_log.json")
    with torch.no_grad():
        y_orig = backbone(x, edge_index_dict).argmax(dim=-1)
    total_edges = sum(int(ei.size(1)) for ei in edge_index_dict.values())
    dev = x.device
    pop = torch.bernoulli(torch.full((pop_size, total_edges), keep_frac, device=dev))
    best_vec, best_fit = pop[0].clone(), -1e9
    for gen in range(1, generations + 1):
        fits = []
        for i in range(pop_size):
            masks = _vector_to_masks(pop[i], edge_index_dict)
            fits.append(_meg_fitness(backbone, x, edge_index_dict, y_orig, masks, sparsity))
        fits_t = torch.tensor(fits, device=dev)
        if fits_t.max() > best_fit:
            best_fit = float(fits_t.max())
            best_vec = pop[int(fits_t.argmax())].clone()
        # tournament selection (k=2) -> uniform crossover -> bit-flip mutation,
        # keeping the elite (best individual) in every generation
        new_pop = [best_vec]
        while len(new_pop) < pop_size:
            def _tournament() -> Tensor:
                i, j = torch.randint(pop_size, (2,))
                return pop[i] if fits_t[i] >= fits_t[j] else pop[j]
            p1, p2 = _tournament(), _tournament()
            mask_cross = torch.rand(total_edges, device=dev) < 0.5
            child = torch.where(mask_cross, p1, p2)
            child = torch.where(torch.rand(total_edges, device=dev) < mutate_prob, 1.0 - child, child)
            new_pop.append(child)
        pop = torch.stack(new_pop)
        if artifact_dir is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": gen, "best_fitness": best_fit,
                                    "mean_fitness": float(fits_t.mean()), "mode": "meg"}) + "\n")
    return _vector_to_masks(best_vec, edge_index_dict)


# ---------------------------------------------------------------------------
# Gradient-based edge attribution (Saliency / GradCAM / Integrated Gradients)
# ---------------------------------------------------------------------------
def gradient_edge_scores(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y_orig: Tensor,
    target_nodes: Tensor,
    method: str = "saliency",
    ig_steps: int = 10,
) -> Dict[str, Tensor]:
    """Score every edge by the gradient of the target-class probability with
    respect to a unit edge mask (standard GNN gradient attribution).

    * ``saliency``: |d p / d m| at m = 1 (Baldassarre & Azizpour, 2019);
    * ``gradcam``: ReLU(d p / d m) (Pope et al., 2019, attention-weighted
      gradient for GNNs);
    * ``ig``: integrated gradients of the mask from 0 to 1 in ``ig_steps``
      steps (Sundararajan et al., 2017).

    The scores are aggregated over ``target_nodes``.  Returns one score tensor
    per relation id (str).
    """
    dev = x.device
    E = {k: ei.size(1) for k, ei in edge_index_dict.items()}
    if method == "ig":
        acc = {k: torch.zeros(E[k], device=dev) for k in edge_index_dict}
        for t in range(1, ig_steps + 1):
            frac = t / ig_steps
            masks = {k: torch.full((E[k],), frac, device=dev, requires_grad=True)
                     for k in edge_index_dict}
            logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
            prob = F.softmax(logits, dim=-1)[target_nodes].gather(1, y_orig[target_nodes].unsqueeze(-1)).sum()
            grads = torch.autograd.grad(prob, list(masks.values()), create_graph=False)
            for k, g in zip(masks.keys(), grads):
                acc[k] = acc[k] + (g.detach() / ig_steps)
        return acc
    masks = {k: torch.ones(E[k], device=dev, requires_grad=True) for k in edge_index_dict}
    logits = backbone(x, edge_index_dict, edge_mask_dict=masks)
    prob = F.softmax(logits, dim=-1)[target_nodes].gather(1, y_orig[target_nodes].unsqueeze(-1)).sum()
    grads = torch.autograd.grad(prob, list(masks.values()), create_graph=False)
    out = {}
    for k, g in zip(masks.keys(), grads):
        g = g.detach()
        if method == "gradcam":
            out[k] = torch.relu(g)
        else:  # saliency
            out[k] = g.abs()
    return out


def topk_keep_masks(edge_index_dict: Dict[str, Tensor], scores: Dict[str, Tensor],
                    budget_frac: float) -> Dict[str, Tensor]:
    """Keep-mask that deletes the top ``budget_frac`` fraction of edges by
    score (global threshold across all relations)."""
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    total = all_scores.numel()
    k = max(int(round(budget_frac * total)), 1)
    if k >= total:
        k = total - 1
    thr = torch.topk(all_scores, k).values[-1]
    return {k: (s <= thr).float() for k, s in scores.items()}


def random_keep_masks(edge_index_dict: Dict[str, Tensor], budget_frac: float,
                      generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
    """Keep-mask that deletes a random ``budget_frac`` fraction of edges."""
    out = {}
    for k, ei in edge_index_dict.items():
        keep = torch.rand(ei.size(1), device=ei.device, generator=generator) >= budget_frac
        out[k] = keep.float()
    return out


# ---------------------------------------------------------------------------
# SubgraphX-style node-level Shapley explanation (Yuan et al., ICML 2021),
# converted to an edge deletion.  Simplified: MCTS over connected node
# subsets with a surrogate GCN trained on the backbone's final embeddings
# (as in SubgraphX), Shapley values estimated by coalition sampling.
# ---------------------------------------------------------------------------
def _bfs_candidates(edge_index, start: int, cap: int = 60,
                    adj: Optional[Dict[int, list]] = None) -> list:
    """BFS node order from ``start`` along ``edge_index`` (undirected)."""
    if adj is None:
        adj = {}
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for u, v in zip(src, dst):
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    seen, frontier, order = {start}, [start], [start]
    while frontier and len(order) < cap:
        nxt = []
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen and len(order) < cap:
                    seen.add(v)
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    return order


def optimize_subgraphx_masks(
    backbone: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y_orig: Tensor,
    target_nodes: Tensor,
    n_targets: int = 200,
    mcts_iters: int = 20,
    n_coalitions: int = 4,
    surrogate_epochs: int = 60,
    candidate_cap: int = 60,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Tensor]:
    """SubgraphX-style explanation mapped to a counterfactual edge deletion.

    (1) Train a surrogate GCN on the backbone's final embeddings with the
        backbone's own predictions (SubgraphX's surrogate-model trick).
    (2) For ``n_targets`` deterministically sampled target nodes, run MCTS
        (UCB1, 20 iterations) over connected node subsets of the node's
        two-hop computation graph; the payoff of a subset is its
        Monte-Carlo Shapley value w.r.t. the surrogate's probability of the
        original class.
    (3) Delete every edge with both endpoints inside the best subgraph.
    """
    if artifact_dir is not None:
        os.makedirs(artifact_dir, exist_ok=True)
    dev = x.device
    # Step 1: surrogate GCN on the embeddings (collapsed adjacency).
    from model.han_hgt import GCN  # local import avoids a cycle
    with torch.no_grad():
        h = backbone.embed(x, edge_index_dict)
    collapsed = collapse(edge_index_dict)
    surrogate = GCN(h.size(1), 32, int(y_orig.max()) + 1, 2, dropout=0.5).to(dev)
    opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(surrogate_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(surrogate(h, collapsed), y_orig)
        loss.backward()
        opt.step()
    surrogate.eval()

    def shapley_value(nodes: list, target: int) -> float:
        """MC-Shapley value of the node subset w.r.t. p(original class)."""
        if not nodes:
            return 0.0
        base = torch.zeros(x.size(0), device=dev, dtype=torch.bool)
        base[nodes] = True
        vals = 0.0
        for _ in range(n_coalitions):
            coalition = torch.rand(x.size(0), device=dev) < 0.3
            coalition[nodes] = False
            with torch.no_grad():
                emb_t = h * coalition.unsqueeze(-1)
                emb_s = h * (coalition | base).unsqueeze(-1)
                p_t = F.softmax(surrogate(emb_t, collapsed)[target], dim=-1)[y_orig[target]].item()
                p_s = F.softmax(surrogate(emb_s, collapsed)[target], dim=-1)[y_orig[target]].item()
            vals += p_s - p_t
        return vals / n_coalitions

    # Step 2: MCTS per target node.
    targets = target_nodes.nonzero().squeeze(-1).tolist()
    if len(targets) > n_targets:
        targets = targets[:: max(len(targets) // n_targets, 1)][:n_targets]
    ei0 = collapsed["0"]
    shared_adj: Dict[int, list] = {}
    src_l, dst_l = ei0[0].tolist(), ei0[1].tolist()
    for u, v in zip(src_l, dst_l):
        shared_adj.setdefault(u, []).append(v)
        shared_adj.setdefault(v, []).append(u)
    best_subgraphs: Dict[int, list] = {}
    for t in targets:
        cand = _bfs_candidates(ei0, t, cap=candidate_cap, adj=shared_adj)
        best_nodes, best_val = [t], -1e9
        # MCTS tree: nodes are subsets, stored as (members, node dict).
        tree: Dict[tuple, Dict] = {(t,): {"n": 0, "w": 0.0, "children": {}, "val": shapley_value([t], t)}}
        for _ in range(mcts_iters):
            path = [(t,)]
            state = (t,)
            # selection
            while tree[state]["children"]:
                node = tree[state]
                best_c = None
                best_ucb = -1e9
                for c in node["children"]:
                    cn = tree[c]
                    ucb = (cn["w"] / max(cn["n"], 1)) + 2.0 * math.sqrt(math.log(max(node["n"], 1) + 1) / max(cn["n"], 1))
                    if ucb > best_ucb:
                        best_ucb, best_c = ucb, c
                state = best_c
                path.append(state)
            members = set(state)
            # expansion: add one candidate node (BFS order, capped)
            new_state = None
            for v in cand:
                if v in members:
                    continue
                s2 = tuple(sorted(members | {v}))
                if s2 not in tree:
                    new_state = s2
                    break
            if new_state is None:
                val = tree[state]["val"]
            else:
                val = shapley_value(list(new_state), t)
                tree[new_state] = {"n": 0, "w": 0.0, "children": {}, "val": val}
                tree[state]["children"][new_state] = None
                path.append(new_state)
                state = new_state
            # backprop (max payoff)
            for s in reversed(path):
                node = tree[s]
                node["n"] += 1
                node["w"] = max(node["w"], val)
            if val > best_val:
                best_val, best_nodes = val, list(state)
        best_subgraphs[t] = best_nodes
        if artifact_dir is not None and len(best_subgraphs) % 50 == 0:
            with open(os.path.join(artifact_dir, "train_log.json"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": len(best_subgraphs), "target": t, "subgraph_size": len(best_nodes)}) + "\n")

    # Step 3: delete edges with both endpoints inside each best subgraph.
    removed_edges = set()
    for t, nodes in best_subgraphs.items():
        ns = set(nodes)
        for i, (u, v) in enumerate(zip(src_l, dst_l)):
            if u in ns and v in ns:
                removed_edges.add(i)
    keep = torch.ones(ei0.size(1), device=dev)
    if removed_edges:
        keep[torch.tensor(sorted(removed_edges), device=dev)] = 0.0
    return {"0": keep}
