"""B3 (P1-1): RACE-v2 per-edge explainer — margin loss + discrete verification
+ backward pruning — evaluated against the baselines on the SAME frozen typed
checkpoints (``checkpoints{tag}/seed_{s}/backbone.pt`` saved by the B1 run).

Artifacts:
  results{tag}/race_v2_{variant}_{dataset}.json

Run (remote, project root code/):
  python race_v2_experiment.py --dataset acm --seeds 0 1 2 3 4 --tag v2 \
      --variant v0 --reuse_ckpt
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

from analysis.metrics import compute_cf_metrics
from benchmarks.baselines import optimize_race_v2_masks
from utils import build_backbone, load_dataset, set_seed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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

    target = None
    if cfg.get("target") == "test":
        target = test_mask

    path_log = []
    t0 = time.time()
    keep = optimize_race_v2_masks(
        backbone, x, eid, nr,
        steps=int(cfg["steps"]), lr=float(cfg["lr"]), sparsity=float(cfg["sparsity"]),
        bin_coef=float(cfg["bin_coef"]), kappa=float(cfg["kappa"]),
        delete_batches=int(cfg["delete_batches"]), prune_batches=int(cfg["prune_batches"]),
        verify_sparsity=cfg.get("verify_sparsity") or None,
        rounds=int(cfg["rounds"]),
        select_mode=cfg.get("select_mode", "score"),
        prune_rounds=int(cfg["prune_rounds"]),
        cont_mode=cfg.get("cont_mode", "margin"),
        target_mask=target, eval_mask=test_mask, path_log=path_log,
        artifact_dir=None,
    )
    m = compute_cf_metrics(backbone, x, eid, y, test_mask, keep)
    m["backbone_acc"] = float(ckpt.get("backbone_acc", np.nan))
    m["nf"] = m["csr"]
    m["sf"] = m["ps"]
    m["time_s"] = time.time() - t0
    m["path"] = path_log
    return m


def main() -> None:
    import yaml

    def _load_yaml(filename: str) -> Dict:
        with open(os.path.join(PROJECT_ROOT, "configs", filename), encoding="utf-8") as f:
            return yaml.safe_load(f)

    real_cfg = _load_yaml("real_experiment.yaml")
    data_cfg = _load_yaml("data.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acm", choices=["acm", "cora", "ACM", "DBLP", "mag"])
    ap.add_argument("--cap", type=int, default=int(real_cfg["cap"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(real_cfg["seeds"]))
    ap.add_argument("--steps", type=int, default=int(real_cfg["explain_steps"]))
    ap.add_argument("--lr", type=float, default=float(real_cfg["explain_lr"]))
    ap.add_argument("--sparsity", type=float, default=float(real_cfg["sparsity"]))
    ap.add_argument("--bin_coef", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--delete_batches", type=int, default=40)
    ap.add_argument("--prune_batches", type=int, default=20)
    ap.add_argument("--verify_sparsity", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--select_mode", default="score", choices=["score", "max_csr"])
    ap.add_argument("--prune_rounds", type=int, default=1)
    ap.add_argument("--cont_mode", default="margin",
                    choices=["margin", "margin_weighted", "counterfactual", "cf2"])
    ap.add_argument("--target", default="all", choices=["all", "test"])
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--reuse_ckpt", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, dataset={args.dataset}, variant={args.variant}", flush=True)

    data = load_dataset(args.dataset, data_cfg, cap=args.cap)
    print(f"[info] {args.dataset}: N={data['x'].size(0)}, "
          f"edges r0={data['edge_index_dict']['0'].size(1)}, "
          f"r1={data['edge_index_dict']['1'].size(1)}", flush=True)

    cfg = vars(args)
    cfg["artifacts_root"] = os.path.join(PROJECT_ROOT, f"results_{args.tag}")
    cfg["ckpt_root"] = os.path.join(PROJECT_ROOT, f"checkpoints_{args.tag}", args.dataset)
    os.makedirs(cfg["artifacts_root"], exist_ok=True)

    per_seed = {s: run_seed(s, data, device, cfg) for s in args.seeds}

    print(f"\n===== RACE-v2 ({args.dataset}, variant {args.variant}, "
          f"mean over {args.seeds}) =====")
    for k in ["csr", "minimality", "ps", "pns"]:
        vals = [per_seed[s][k] for s in args.seeds]
        print(f"{k}: {np.mean(vals):.4f} +- {np.std(vals):.4f}")

    out = {
        "dataset": args.dataset,
        "variant": args.variant,
        "config": {k: v for k, v in cfg.items() if k != "seeds"},
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
    }
    out_path = os.path.join(cfg["artifacts_root"], f"race_v2_{args.variant}_{args.dataset}.json")
    _write_json(out_path, out)
    print(f"[info] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
