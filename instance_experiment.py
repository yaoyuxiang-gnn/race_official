"""B1 (P0-1): instance-level relation-type counterfactual search on real data.

Trains the relation-aware typed backbone once per seed (same hyperparameters as
the v1 main experiment), freezes it, and runs the exact instance-level search
S_v* (advisor Sec. 2.1-2.2) over every correctly-classified test node.  Saves
per-instance records plus the aggregate summary (coverage, C_type, C_edge,
final margin, NF/SF, per-subset flip rates for the dataset-level Eq.(5)
comparison).

Artifacts (spec.md Sec. 3/4):
  results{tag}/instance_rel_{dataset}.json   per-seed records + summary
  checkpoints{tag}/seed_{s}/backbone.pt      frozen typed backbone (B2 reuse)

Run (remote, project root):
  python code/instance_experiment.py --dataset acm --seeds 0 1 2 --tag v2 \
      --max_targets 2500 --kappa 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.metrics import accuracy
from model.instance_relation_explainer import InstanceRelationExplainer, dataset_level_s_star
from utils import build_backbone, load_dataset, set_seed, train_backbone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_seed(seed: int, data: Dict, device: torch.device, cfg: Dict) -> Dict:
    """Train (or load) the typed backbone and run the instance-level search."""
    set_seed(seed)
    x = data["x"].to(device)
    y = data["y"].to(device)
    eid = {k: v.to(device) for k, v in data["edge_index_dict"].items()}
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)
    nr = len(eid)
    out_dim = int(y.max().item()) + 1

    ckpt_dir = os.path.join(cfg["ckpt_root"], f"seed_{seed}")
    ckpt_path = os.path.join(ckpt_dir, "backbone.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    if cfg.get("reuse_ckpt", False) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone = build_backbone(
            ckpt["cfg"]["kind"], x.size(1), out_dim, nr,
            hidden_dim=ckpt["cfg"]["hidden_dim"], num_layers=ckpt["cfg"]["num_layers"],
            num_heads=ckpt["cfg"].get("num_heads", 4), dropout=ckpt["cfg"]["dropout"],
            use_rel_weights=ckpt["cfg"].get("use_rel_weights", True), device=device,
        )
        backbone.load_state_dict(ckpt["model_state_dict"])
        backbone_acc = float(ckpt.get("backbone_acc", np.nan))
        print(f"[seed {seed}] loaded backbone from {ckpt_path} (acc={backbone_acc:.4f})", flush=True)
    else:
        backbone = build_backbone(
            cfg["backbone"], x.size(1), out_dim, nr,
            hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], dropout=0.0 if cfg.get("no_regularization", False) else cfg["dropout"],
            use_rel_weights=cfg["use_rel_weights"], device=device,
        )
        train_backbone(
            backbone, x, eid, y, train_mask, val_mask, test_mask,
            steps=int(cfg["backbone_steps"]), lr=float(cfg["lr"]),
            weight_decay=0.0 if cfg.get("no_regularization", False) else float(cfg["weight_decay"]),
            early_stop=cfg["early_stop"], artifact_dir=None, ckpt_dir=None, full_artifacts=False,
        )
        backbone.eval()
        backbone_acc = accuracy(backbone, x, eid, y, test_mask)
        torch.save({
            "model_state_dict": backbone.state_dict(),
            "cfg": {
                "kind": cfg["backbone"], "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"], "num_heads": cfg["num_heads"],
                "dropout": cfg["dropout"], "use_rel_weights": cfg["use_rel_weights"],
            },
            "backbone_acc": backbone_acc,
        }, ckpt_path)
        print(f"[seed {seed}] trained backbone, test acc={backbone_acc:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    backbone.eval()
    explainer = InstanceRelationExplainer(
        backbone, nr, num_layers=int(cfg["num_layers"]), kappa=float(cfg["kappa"]),
    )
    t1 = time.time()
    out = explainer.explain(
        x, eid, y, target_mask=test_mask,
        max_targets=cfg.get("max_targets") or None,
        verbose=bool(cfg.get("verbose", False)),
        select_mode=cfg.get("select_mode", "lexicographic"),
    )
    print(f"[seed {seed}] instance search: coverage={out['coverage']:.4f} "
          f"n_feasible={out['n_feasible']}/{out['n_targets']} "
          f"mean_cost_type={out['mean_cost_type']} mean_cost_edge={out['mean_cost_edge']} "
          f"sf={out['sf']} ({time.time() - t1:.1f}s)", flush=True)

    # dataset-level S* (v1 Eq. 5) reconstructed from the same forwards: ablation.
    s_star = {}
    for tau in cfg.get("tau_rel_scan", [0.9]):
        s_star[f"S*_tau_{tau}"] = dataset_level_s_star(out["flip_rate_per_subset"], nr, float(tau))
    return {
        "backbone_acc": backbone_acc,
        "summary": {k: v for k, v in out.items() if k != "records"},
        "records": out["records"],
        "dataset_level_S": s_star,
    }


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
    ap.add_argument("--hidden_dim", type=int, default=int(real_cfg["hidden_dim"]))
    ap.add_argument("--num_layers", type=int, default=int(real_cfg["num_layers"]))
    ap.add_argument("--backbone_steps", type=int, default=int(real_cfg["backbone_steps"]))
    ap.add_argument("--lr", type=float, default=float(real_cfg["lr"]))
    ap.add_argument("--dropout", type=float, default=float(real_cfg["dropout"]))
    ap.add_argument("--weight_decay", type=float, default=float(real_cfg["weight_decay"]))
    ap.add_argument("--tau_rel_scan", type=float, nargs="+", default=list(real_cfg["tau_rel_scan"]))
    ap.add_argument("--kappa", type=float, default=0.0, help="flip-confidence margin (valid iff M_v <= -kappa)")
    ap.add_argument("--select_mode", default="lexicographic",
                    choices=["lexicographic", "cost_type", "cost_edge"],
                    help="cost criterion of the instance-level search (B10 ablation)")
    ap.add_argument("--max_targets", type=int, default=2500,
                    help="cap on the number of correctly-classified test nodes to explain")
    ap.add_argument("--backbone", default="hetero", choices=["hetero", "han", "hgt", "rgcn"])
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--reuse_ckpt", action="store_true", help="load backbone.pt if present (skip retraining)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device != "auto"
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device={device}, dataset={args.dataset}, kappa={args.kappa}", flush=True)

    # B1b: seed the global RNGs BEFORE data loading so that loaders using
    # torch.randperm (mag citation cap, cora common-neighbor cap) reproduce
    # the same graph on every run (advisor step 4: one graph constructor for
    # every compared method).
    set_seed(0)
    data = load_dataset(args.dataset, data_cfg, cap=args.cap)
    print(f"[info] {args.dataset}: N={data['x'].size(0)}, "
          f"edges r0={data['edge_index_dict']['0'].size(1)}, "
          f"r1={data['edge_index_dict']['1'].size(1)}, classes={int(data['y'].max()) + 1}", flush=True)

    cfg = vars(args)
    cfg["use_rel_weights"] = True
    cfg["early_stop"] = True
    cfg["no_regularization"] = False
    cfg["artifacts_root"] = os.path.join(PROJECT_ROOT, f"results_{args.tag}")
    ckpt_scope = args.dataset + (f"_{args.backbone}" if args.backbone != "hetero" else "")
    cfg["ckpt_root"] = os.path.join(PROJECT_ROOT, f"checkpoints_{args.tag}", ckpt_scope)
    os.makedirs(cfg["artifacts_root"], exist_ok=True)
    os.makedirs(cfg["ckpt_root"], exist_ok=True)

    per_seed = {s: run_seed(s, data, device, cfg) for s in args.seeds}

    # ---- aggregate summary -------------------------------------------------
    print(f"\n===== Instance-level relation search ({args.dataset}, kappa={args.kappa}) =====")
    keys = ["coverage", "mean_cost_type", "mean_cost_edge", "mean_final_margin", "sf"]
    print("| seed | " + " | ".join(keys) + " | backbone_acc |")
    for s in args.seeds:
        summ = per_seed[s]["summary"]
        row = " | ".join(
            f"{summ[k]:.4f}" if isinstance(summ.get(k), float) else str(summ.get(k))
            for k in keys
        )
        print(f"| {s} | {row} | {per_seed[s]['backbone_acc']:.4f} |")
    print("\n| metric | mean | std |")
    print("|---|---|---|")
    for k in keys:
        vals = [per_seed[s]["summary"].get(k) for s in args.seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"| {k} | {np.mean(vals):.4f} | {np.std(vals):.4f} |")
    for s in args.seeds:
        print(f"[info] seed {s}: dataset-level S* = {per_seed[s]['dataset_level_S']}")
        print(f"[info] seed {s}: flip rates = {per_seed[s]['summary']['flip_rate_per_subset']}")

    out = {
        "dataset": args.dataset,
        "config": {k: v for k, v in cfg.items() if k != "seeds"},
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
    }
    sm_suffix = "" if args.select_mode == "lexicographic" else f"_{args.select_mode}"
    if float(args.kappa) != 0.0:
        sm_suffix += f"_kappa{args.kappa:g}"
    out_path = os.path.join(cfg["artifacts_root"], f"instance_rel_{args.dataset}{sm_suffix}.json")
    _write_json(out_path, out)
    print(f"[info] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
