"""B4: hierarchical relation-to-edge explanation vs flat baselines.

Per seed, on the SAME frozen typed checkpoint:
  1. Hierarchical explanation (B1 relation search + per-node restoration
     pruning inside S_v* + CF2 fallback), with per-node records and timings.
  2. Flat targeted baselines (CF2 / PNS / counterfactual) with per-node flip
     status and per-node LOCAL edge cost (deleted receptive-field edges), so
     all four dimensions (success rate, edge cost, stability, speed) can be
     compared on equal footing.

Artifacts:
  results{tag}/hierarchical_{dataset}.json     per-seed records + baselines
  results{tag}/b4_masks/{dataset}/seed{s}/{method}.pt   hard masks (stability)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.baselines import optimize_edge_masks
from model.hierarchical_explainer import build_structures, hierarchical_explain
from model.instance_relation_explainer import prediction_margin
from utils import build_backbone, load_dataset, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE_METHODS = [("CF2-hetero", "cf2"), ("PNS-hetero", "pns"), ("RACE-per-edge", "counterfactual")]


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def local_masked_counts(rel_map, adj, edge_index_dict, keep_masks, nodes, num_layers):
    """Per node: (deleted_local, total_local) under the given hard masks."""
    out = {}
    K = len(edge_index_dict)
    for v in nodes:
        L = num_layers
        ball = {v}
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
        deleted, total = 0, 0
        for r in range(K):
            key = str(r)
            mask = keep_masks[key]
            for w in ball_prev:
                for (u, gi) in rel_map[key].get(w, []):
                    if u in ball:
                        total += 1
                        if float(mask[gi]) < 0.5:
                            deleted += 1
        out[v] = (deleted, total)
    return out


def run_seed(seed: int, data: Dict, device: torch.device, cfg: Dict) -> Dict:
    set_seed(seed)
    x = data["x"].to(device)
    y = data["y"].to(device)
    eid = {k: v.to(device) for k, v in data["edge_index_dict"].items()}
    test_mask = data["test_mask"].to(device)
    nr = len(eid)
    out_dim = int(y.max().item()) + 1

    ckpt = torch.load(os.path.join(cfg["ckpt_root"], f"seed_{seed}", "backbone.pt"),
                      map_location=device)
    backbone = build_backbone(
        ckpt["cfg"]["kind"], x.size(1), out_dim, nr,
        hidden_dim=ckpt["cfg"]["hidden_dim"], num_layers=ckpt["cfg"]["num_layers"],
        num_heads=ckpt["cfg"].get("num_heads", 4), dropout=ckpt["cfg"]["dropout"],
        use_rel_weights=ckpt["cfg"].get("use_rel_weights", True), device=device,
    )
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.eval()

    rel_map, adj = build_structures(eid, x.size(0))

    # ---- flat targeted baselines first (fallback + comparison) ----------
    baselines: Dict[str, Dict] = {}
    masks_dir = os.path.join(cfg["artifacts_root"], "b4_masks",
                             cfg["dataset"] + (f"_{cfg['backbone']}" if cfg["backbone"] != "hetero" else ""),
                             f"seed{seed}")
    os.makedirs(masks_dir, exist_ok=True)
    cf2_masks = None
    for name, mode in BASELINE_METHODS:
        t0 = time.time()
        keep = optimize_edge_masks(
            backbone, x, eid, nr, mode,
            int(cfg["explain_steps"]), float(cfg["explain_lr"]), float(cfg["sparsity"]),
            target_mask=test_mask, gumbel_tau=float(cfg["gumbel_tau"]),
        )
        time_s = time.time() - t0
        torch.save({k: v.cpu() for k, v in keep.items()},
                   os.path.join(masks_dir, f"{name}.pt"))
        if name == "CF2-hetero":
            cf2_masks = keep
        with torch.no_grad():
            logits_full = backbone(x, eid)
            logits_cf = backbone(x, eid, edge_mask_dict=keep)
            probs_full = F.softmax(logits_full, dim=-1)
            probs_cf = F.softmax(logits_cf, dim=-1)
            y_full = logits_full.argmax(dim=-1)
        correct = test_mask & (y_full == y)
        nodes = correct.nonzero().squeeze(-1).tolist()
        # unified flip predicate (advisor step 1): margin <= -kappa, ties flip
        yv = y_full[nodes]
        marg = prediction_margin(probs_cf[nodes], yv).cpu()
        flip = [v for i, v in enumerate(nodes) if float(marg[i]) <= -float(cfg["kappa"])]
        lmc = local_masked_counts(rel_map, adj, eid, keep, nodes, int(cfg["num_layers"]))
        loc_cost = [lmc[v][0] / lmc[v][1] for v in nodes if lmc[v][1] > 0]
        baselines[name] = {
            "csr": len(flip) / max(len(nodes), 1),
            "n_correct": len(nodes),
            "mean_local_cost": float(sum(loc_cost) / len(loc_cost)) if loc_cost else None,
            "time_s": time_s,
        }
        print(f"[seed {seed}] {name}: csr={baselines[name]['csr']:.4f} "
              f"local_cost={baselines[name]['mean_local_cost']} "
              f"({time_s:.1f}s)", flush=True)

    # ---- hierarchical + flat-verified (relation switch, B2/B3) -----------
    t0 = time.time()
    hier = hierarchical_explain(
        backbone, x, eid, y, test_mask,
        num_layers=int(cfg["num_layers"]), kappa=float(cfg["kappa"]),
        max_targets=cfg.get("max_targets") or None,
        restore_batches=int(cfg["restore_batches"]),
        scan_budget=int(cfg["scan_budget"]),
        relation_phase=True,
        flat_masks=cf2_masks,
    )
    hier["summary"]["time_total_s"] = time.time() - t0
    print(f"[seed {seed}] hierarchical: csr={hier['summary']['csr']:.4f} "
          f"feasible={hier['summary']['n_feasible_rel']}/{hier['summary']['n_targets']} "
          f"mean_edge_cost={hier['summary']['mean_edge_cost_feasible']} "
          f"irr={hier['summary']['n_irreducible']} "
          f"ver_only={hier['summary']['n_verified_only']} "
          f"(total {hier['summary']['time_total_s']:.1f}s: "
          f"rel {hier['summary']['time_rel_s']:.1f}s + "
          f"prune {hier['summary']['time_prune_s']:.1f}s)", flush=True)

    t0 = time.time()
    flat = hierarchical_explain(
        backbone, x, eid, y, test_mask,
        num_layers=int(cfg["num_layers"]), kappa=float(cfg["kappa"]),
        max_targets=cfg.get("max_targets") or None,
        restore_batches=int(cfg["restore_batches"]),
        scan_budget=int(cfg["scan_budget"]),
        relation_phase=False,
        del_batches=int(cfg["del_batches"]),
    )
    flat["summary"]["time_total_s"] = time.time() - t0
    print(f"[seed {seed}] flat-verified: csr={flat['summary']['csr']:.4f} "
          f"mean_edge_cost={flat['summary']['mean_edge_cost_feasible']} "
          f"irr={flat['summary']['n_irreducible']} "
          f"ver_only={flat['summary']['n_verified_only']} "
          f"({flat['summary']['time_total_s']:.1f}s)", flush=True)
    return {"baselines": baselines, "summary": hier["summary"], "records": hier["records"],
            "flat_summary": flat["summary"], "flat_records": flat["records"]}


def main() -> None:
    import yaml

    def _load_yaml(filename: str) -> Dict:
        with open(os.path.join(PROJECT_ROOT, "configs", filename), encoding="utf-8") as f:
            return yaml.safe_load(f)

    real_cfg = _load_yaml("real_experiment.yaml")
    data_cfg = _load_yaml("data.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acm", choices=["acm", "cora", "ACM", "DBLP", "mag", "mag4", "arxiv", "dblp"])
    ap.add_argument("--cap", type=int, default=int(real_cfg["cap"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(real_cfg["seeds"]))
    ap.add_argument("--num_layers", type=int, default=int(real_cfg["num_layers"]))
    ap.add_argument("--explain_steps", type=int, default=int(real_cfg["explain_steps"]))
    ap.add_argument("--explain_lr", type=float, default=float(real_cfg["explain_lr"]))
    ap.add_argument("--sparsity", type=float, default=float(real_cfg["sparsity"]))
    ap.add_argument("--gumbel_tau", type=float, default=float(real_cfg["gumbel_tau"]))
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--max_targets", type=int, default=2500)
    ap.add_argument("--restore_batches", type=int, default=8)
    ap.add_argument("--scan_budget", type=int, default=128,
                    help="per-node forward cap of the single-edge restoration scan")
    ap.add_argument("--del_batches", type=int, default=20,
                    help="flat-verified variant: greedy deletion batches")
    ap.add_argument("--backbone", default="hetero", choices=["hetero", "han", "hgt", "rgcn"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device != "auto"
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device={device}, dataset={args.dataset}", flush=True)

    # B1b: seed the global RNGs BEFORE data loading so that loaders using
    # torch.randperm (mag citation cap, cora common-neighbor cap) reproduce
    # the same graph on every run (advisor step 4: one graph constructor for
    # every compared method).
    set_seed(0)
    data = load_dataset(args.dataset, data_cfg, cap=args.cap)
    print(f"[info] {args.dataset}: N={data['x'].size(0)}, "
          f"edges r0={data['edge_index_dict']['0'].size(1)}, "
          f"r1={data['edge_index_dict']['1'].size(1)}", flush=True)

    cfg = vars(args)
    cfg["artifacts_root"] = os.path.join(PROJECT_ROOT, f"results_{args.tag}")
    ckpt_scope = args.dataset + (f"_{args.backbone}" if args.backbone != "hetero" else "")
    cfg["ckpt_root"] = os.path.join(PROJECT_ROOT, f"checkpoints_{args.tag}", ckpt_scope)
    os.makedirs(cfg["artifacts_root"], exist_ok=True)

    per_seed = {s: run_seed(s, data, device, cfg) for s in args.seeds}

    print(f"\n===== Hierarchical summary ({args.dataset}) =====")
    import numpy as np
    for m in ["csr", "mean_edge_cost_feasible"]:
        vals = [per_seed[s]["summary"][m] for s in args.seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"{m}: {np.mean(vals):.4f} +- {np.std(vals):.4f}")
    for name, _ in BASELINE_METHODS:
        csr = np.mean([per_seed[s]["baselines"][name]["csr"] for s in args.seeds])
        lc = [per_seed[s]["baselines"][name]["mean_local_cost"] for s in args.seeds]
        lc = [v for v in lc if v is not None]
        print(f"{name}: csr={csr:.4f} mean_local_cost={np.mean(lc):.4f}")

    out = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "config": {k: v for k, v in cfg.items() if k != "seeds"},
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
    }
    suffix = f"_{args.backbone}" if args.backbone != "hetero" else ""
    out_path = os.path.join(cfg["artifacts_root"], f"hierarchical_{args.dataset}{suffix}.json")
    _write_json(out_path, out)
    print(f"[info] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
