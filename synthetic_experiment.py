"""Synthetic ground-truth experiments (E2 of experiment_plan.md).

Three parts:
  A. Relation-type identification: exhaustive search vs. two differentiable
     scalar-mask relaxations (deterministic sigmoid, Gumbel-Sigmoid) over
     multiple seeds -> root-cause hit rate (supports claim C5).
  B. Main five-method comparison on the synthetic graph (reuses
     ``real_data_experiment.run_seed`` with ``full_artifacts=False``) plus
     causal-edge precision/recall of the per-edge interventions.
  C. PN/PS/PNS validation: necessity/sufficiency of the injected causal
     relation vs. noise relations, and the K=1 collapse degeneracy check.

Artifacts follow spec.md: per-epoch metrics under ``results/{seed}/...`` with
final-epoch checkpoints only (diagnostic volume, documented in utils.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.metrics import compute_cf_metrics
from model.data.synthetic import load_synthetic
from model.hetero_gnn import HeteroGNN
from model.relation_type_explainer import RelationTypeExplainer
from utils import set_seed, train_backbone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_synthetic")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_synthetic")


def _hit(removed, causal=0):
    removed = set(int(r) for r in removed)
    return float(removed == {causal})


def _optimize_scalar_masks(backbone, x, eid, target, steps=100, lr=0.1, sparsity=0.05, gumbel=False, tau=1.0, seed=0):
    """Optimize one scalar keep-mask per relation; return removed type ids.

    ``gumbel=False`` uses the deterministic sigmoid (as RelationTypeExplainer);
    ``gumbel=True`` adds Logistic noise (Gumbel-Sigmoid relaxation).
    """
    torch.manual_seed(seed)
    K = len(eid)
    logits = torch.full((K,), 3.0, requires_grad=True, device=x.device)
    with torch.no_grad():
        y_full = backbone(x, eid).argmax(dim=-1)
        if target is not None:
            y_full = y_full[target]
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = logits
        if gumbel:
            eps = -torch.log(-torch.log(torch.rand_like(z).clamp(min=1e-8)) + 1e-8)
            z = z + eps
        scalars = torch.sigmoid(z / tau)
        masks = {str(r): scalars[r].expand(eid[str(r)].size(1)) for r in range(K)}
        out = backbone(x, eid, edge_mask_dict=masks)
        if target is not None:
            out = out[target]
        p_orig = F.softmax(out, dim=-1).gather(1, y_full.unsqueeze(-1)).squeeze(-1).mean()
        removed = (1.0 - torch.sigmoid(logits)).mean()
        loss = p_orig + sparsity * removed
        loss.backward()
        opt.step()
    keep = (torch.sigmoid(logits) > 0.5).float()
    return [r for r in range(K) if keep[r].item() < 0.5]


def part_a(args, device):
    """Hit-rate comparison of relation-type identification strategies (C5)."""
    rows = {}
    for seed in args.seeds:
        set_seed(seed)
        data = load_synthetic(n_nodes=args.nodes, n_relations=args.relations,
                              n_classes=args.classes, feat_dim=args.feat_dim, feat_noise=args.feat_noise,
                              p_in=args.p_in, p_noise=args.p_noise, seed=seed,
                              split=args.split)
        x = data["x"].to(device)
        y = data["y"].to(device)
        eid = {k: v.to(device) for k, v in data["edge_index_dict"].items()}
        out_dim = int(y.max().item()) + 1
        model = HeteroGNN(x.size(1), args.hidden_dim, out_dim, args.num_layers,
                          len(eid), dropout=0.0).to(device)
        train_backbone(model, x, eid, y, data["train_mask"].to(device),
                       data["val_mask"].to(device), data["test_mask"].to(device),
                       steps=args.backbone_steps, lr=args.lr,
                       artifact_dir=os.path.join(RESULTS_DIR, f"seed_{seed}", "syn_partA_backbone"),
                       ckpt_dir=os.path.join(CKPT_DIR, f"seed_{seed}", "syn_partA_backbone"),
                       full_artifacts=False)
        model.eval()

        rte = RelationTypeExplainer(model, len(eid), gumbel_tau=1.0, init_logit=3.0)
        s_exh = rte.explain_exhaustive(x, eid, target_nodes=data["test_mask"].to(device), rel_threshold=args.tau_rel)
        from real_data_experiment import relation_flip_rates as _rfr
        flips, full = _rfr(model, x, eid, data["test_mask"].to(device))
        s_arg = [int(max(flips, key=flips.get))] if full > 0 else []
        s_sig = _optimize_scalar_masks(model, x, eid, data["test_mask"].to(device),
                                       steps=args.explain_steps, lr=args.explain_lr,
                                       sparsity=args.sparsity, gumbel=False, seed=seed)
        s_gum = _optimize_scalar_masks(model, x, eid, data["test_mask"].to(device),
                                       steps=args.explain_steps, lr=args.explain_lr,
                                       sparsity=args.sparsity, gumbel=True, tau=1.0, seed=seed)
        rows[seed] = {
            "exhaustive_tau": {"removed": s_exh, "hit": _hit(s_exh)},
            "exhaustive_argmax": {"removed": s_arg, "hit": _hit(s_arg)},
            "sigmoid": {"removed": s_sig, "hit": _hit(s_sig)},
            "gumbel_sigmoid": {"removed": s_gum, "hit": _hit(s_gum)},
        }
        print(f"[A] seed {seed}: exh_tau={s_exh} exh_arg={s_arg} sig={s_sig} gum={s_gum}", flush=True)

    summary = {k: {m: float(np.mean([rows[s][k][m] for s in args.seeds])) for m in ["hit"]}
               for k in ["exhaustive_tau", "exhaustive_argmax", "sigmoid", "gumbel_sigmoid"]}
    out = {"part": "A_relation_type_identification", "seeds": args.seeds, "per_seed": {str(s): rows[s] for s in args.seeds}, "summary": summary}
    path = os.path.join(RESULTS_DIR, f"synthetic_hitrate{tag_sfx}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[A] hit-rate summary: {summary} -> {path}", flush=True)
    return summary


def part_b(args, device):
    """Main five-method comparison on synthetic + causal-edge precision/recall."""
    from real_data_experiment import run_seed, relation_flip_rates
    from benchmarks.baselines import collapse

    cfg = vars(args)
    cfg.update({
        "use_rel_weights": True, "early_stop": True, "full_artifacts": False,
        "tau_rel_scan": [args.tau_rel], "tau_rel": args.tau_rel, "no_regularization": False,
        "artifacts_root": RESULTS_DIR, "ckpt_root": CKPT_DIR,
    })
    per_seed = {}
    causal_precision, causal_recall = {}, {}
    for seed in args.seeds:
        data = load_synthetic(n_nodes=args.nodes, n_relations=args.relations,
                              n_classes=args.classes, feat_dim=args.feat_dim, feat_noise=args.feat_noise,
                              p_in=args.p_in, p_noise=args.p_noise, seed=seed,
                              split=args.split)
        res = run_seed(seed, data, device, cfg)
        per_seed[seed] = res

        # causal-edge precision/recall of Ours(per-edge): removed edges in rel0
        masks = torch.load(os.path.join(RESULTS_DIR, f"seed_{seed}", "Ours(per-edge)", "edge_masks.pt"), map_location="cpu")
        e0 = data["edge_index_dict"]["0"]
        e1 = data["edge_index_dict"]["1"]
        removed_total, causal_removed = 0, 0
        for r_key in masks:
            is_causal = (r_key == "0")
            removed_total += int((masks[r_key] < 0.5).sum().item())
            if is_causal:
                causal_removed += int((masks[r_key] < 0.5).sum().item())
        causal_precision[seed] = causal_removed / max(removed_total, 1)
        causal_recall[seed] = causal_removed / max(int(masks["0"].numel()), 1)

    methods = ["GNNExplainer", "CF-GNNExplainer", "PNS", "Ours(per-edge)", "Ours(rel-type)"]
    metrics = ["backbone_acc", "csr", "minimality", "ps", "pns"]
    print(f"\n===== Synthetic main results (mean±std over {args.seeds}) =====")
    print("| 方法 | " + " | ".join(metrics) + " |")
    print("|" + "---|" * (len(metrics) + 1) + "|")
    summary = {}
    for m in methods:
        vals = {k: [per_seed[s][m][k] for s in args.seeds] for k in metrics}
        summary[m] = {k: {"mean": float(np.mean(vals[k])), "std": float(np.std(vals[k]))} for k in metrics}
        ms = [f"{np.mean(vals[k]):.4f}±{np.std(vals[k]):.4f}" for k in metrics]
        print(f"| {m} | " + " | ".join(ms) + " |")
    summary["causal_precision_Ours_per_edge"] = {str(s): causal_precision[s] for s in args.seeds}
    summary["causal_recall_Ours_per_edge"] = {str(s): causal_recall[s] for s in args.seeds}
    out = {"part": "B_synthetic_main", "seeds": args.seeds, "per_seed": {str(s): per_seed[s] for s in args.seeds}, "summary": summary}
    path = os.path.join(RESULTS_DIR, f"synthetic_main{tag_sfx}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[B] saved {path}", flush=True)
    return summary


def part_c(args, device):
    """PN/PS/PNS validation: causal vs. noise relations + K=1 degeneracy."""
    from real_data_experiment import relation_flip_rates

    rows = {}
    for seed in args.seeds:
        set_seed(seed)
        data = load_synthetic(n_nodes=args.nodes, n_relations=args.relations,
                              n_classes=args.classes, feat_dim=args.feat_dim, feat_noise=args.feat_noise,
                              p_in=args.p_in, p_noise=args.p_noise, seed=seed,
                              split=args.split)
        x = data["x"].to(device)
        y = data["y"].to(device)
        eid = {k: v.to(device) for k, v in data["edge_index_dict"].items()}
        test_mask = data["test_mask"].to(device)
        out_dim = int(y.max().item()) + 1
        model = HeteroGNN(x.size(1), args.hidden_dim, out_dim, args.num_layers,
                          len(eid), dropout=0.0).to(device)
        train_backbone(model, x, eid, y, data["train_mask"].to(device),
                       data["val_mask"].to(device), test_mask,
                       steps=args.backbone_steps, lr=args.lr)
        model.eval()

        per_rel, full = relation_flip_rates(model, x, eid, test_mask)
        # PS(r) = keeping only relation r preserves prediction
        K = len(eid)
        with torch.no_grad():
            y_full = model(x, eid).argmax(dim=-1)[test_mask]
        ps = {}
        for r in range(K):
            keep = {str(k): torch.zeros(eid[str(k)].size(1), device=device) for k in range(K)}
            keep[str(r)] = torch.ones(eid[str(r)].size(1), device=device)
            with torch.no_grad():
                y_only = model(x, eid, edge_mask_dict=keep).argmax(dim=-1)[test_mask]
            ps[r] = float((y_only == y_full).float().mean().item())
        rows[seed] = {"pn_per_relation": per_rel, "ps_per_relation": ps, "flip_full_removal": full}
        print(f"[C] seed {seed}: PN={per_rel} PS={ps}", flush=True)

    out = {"part": "C_pnps_validation", "seeds": args.seeds, "per_seed": {str(s): rows[s] for s in args.seeds}}
    path = os.path.join(RESULTS_DIR, f"synthetic_pnps_validation{tag_sfx}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[C] saved {path}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ABC", help="A/B/C/AB/ABC")
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--relations", type=int, default=3)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--feat_dim", type=int, default=16)
    ap.add_argument("--feat_noise", type=float, default=4.0)
    ap.add_argument("--p_in", type=float, default=0.6)
    ap.add_argument("--p_noise", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="60/20/20")
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--backbone_steps", type=int, default=300)
    ap.add_argument("--explain_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--explain_lr", type=float, default=0.1)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--tau_rel", type=float, default=0.9)
    ap.add_argument("--cap", type=int, default=50000)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--tag", default="", help="suffix for the result JSON files")
    args = ap.parse_args()
    global tag_sfx
    tag_sfx = f"_{args.tag}" if args.tag else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}", flush=True)
    if "A" in args.part:
        part_a(args, device)
    if "B" in args.part:
        part_b(args, device)
    if "C" in args.part:
        part_c(args, device)


if __name__ == "__main__":
    main()
