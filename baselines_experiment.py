"""Extended-baseline comparison for the RACE paper (Sec. IV-C expansion).

Runs the additional baseline families against the two RACE granularities on
ACM / ogbn-mag / Cora:

* Pairwise explainers on a **shared collapsed backbone** (one per seed):
  PGExplainer (amortized factual), RCExplainer (hinge counterfactual),
  CF^2 (counterfactual + factual + validity), MEG (genetic search),
  SubgraphX-style (MCTS + Shapley surrogate, optional), GradCAM / Saliency /
  Integrated Gradients (gradient edge attribution + top-k removal) and
  Random deletion (control).
* Heterogeneous-graph baselines on a **shared heterogeneous backbone**:
  GNNExplainer-hetero (factual objective, per-relation masks) and PNS-hetero
  (PN/PS objective, per-relation masks).  These isolate the explanation
  objective from the backbone, complementing the pairwise baselines which
  differ from RACE in both dimensions.
* Backbone-accuracy rows: GCN / GAT on the collapsed graph and HAN / HGT on
  the typed graph, trained with the identical protocol (Adam, weight decay,
  early stopping, 60/20/20 splits, same seeds).

Artifacts follow spec.md Sec. 3/4 with the documented high-volume deviation
``full_artifacts=False`` (per-epoch metrics + final checkpoint only, see
experiment_plan.md E5).
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

from analysis.metrics import accuracy, compute_cf_metrics
from benchmarks.baselines import (
    collapse,
    gradient_edge_scores,
    optimize_edge_masks,
    optimize_meg_masks,
    optimize_pge_masks,
    optimize_subgraphx_masks,
    random_keep_masks,
    topk_keep_masks,
)
from model.han_hgt import GAT, GCN, HAN, HGT
from model.hetero_gnn import HeteroGNN
from utils import set_seed, train_backbone

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
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)
    nr = len(eid)
    out_dim = int(y.max().item()) + 1
    no_reg = bool(cfg.get("no_regularization", False))
    early_stop = bool(cfg.get("early_stop", True)) and not no_reg
    artifacts_root = cfg["artifacts_root"]
    ckpt_root = cfg["ckpt_root"]
    seed_dir = lambda m: os.path.join(artifacts_root, f"seed_{seed}", m)
    ckpt_dir_fn = lambda m: os.path.join(ckpt_root, f"seed_{seed}", m)

    def train_and_eval(model, edges, method, steps=None):
        train_backbone(
            model, x, edges, y, train_mask, val_mask, test_mask,
            steps=int(cfg["backbone_steps"]) if steps is None else steps,
            lr=float(cfg["lr"]),
            weight_decay=0.0 if no_reg else float(cfg["weight_decay"]),
            early_stop=early_stop,
            artifact_dir=seed_dir(method),
            ckpt_dir=ckpt_dir_fn(method),
            full_artifacts=False,
        )
        model.eval()
        return accuracy(model, x, edges, y, test_mask)

    results: Dict[str, Dict] = {}

    # --- shared heterogeneous backbone (RACE backbone) ----------------------
    t0 = time.time()
    hetero = HeteroGNN(
        x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
        dropout=0.0 if no_reg else cfg["dropout"],
    ).to(device)
    acc_hetero = train_and_eval(hetero, eid, "HeteroBackbone")
    results["HeteroBackbone"] = {
        "backbone_acc": acc_hetero,
        "params": sum(p.numel() for p in hetero.parameters()),
        "train_time_s": time.time() - t0,
    }

    # --- shared collapsed backbone for all pairwise baselines --------------
    collapsed = collapse(eid)
    t0 = time.time()
    gcn_backbone = HeteroGNN(
        x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], 1,
        dropout=0.0 if no_reg else cfg["dropout"],
    ).to(device)
    acc_coll = train_and_eval(gcn_backbone, collapsed, "CollapsedBackbone")
    results["CollapsedBackbone"] = {
        "backbone_acc": acc_coll,
        "params": sum(p.numel() for p in gcn_backbone.parameters()),
        "train_time_s": time.time() - t0,
    }

    # --- backbone-accuracy baselines (GCN / GAT / HAN / HGT) ----------------
    for name, model, edges in [
        ("GCN", GCN(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"],
                    dropout=0.0 if no_reg else cfg["dropout"]).to(device), collapsed),
        ("GAT", GAT(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"],
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=0.0 if no_reg else cfg["dropout"]).to(device), collapsed),
        ("HAN", HAN(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=0.0 if no_reg else cfg["dropout"]).to(device), eid),
        ("HGT", HGT(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
                    num_heads=int(cfg.get("num_heads", 4)),
                    dropout=0.0 if no_reg else cfg["dropout"]).to(device), eid),
    ]:
        t0 = time.time()
        acc = train_and_eval(model, edges, name)
        results[name] = {
            "backbone_acc": acc,
            "params": sum(p.numel() for p in model.parameters()),
            "train_time_s": time.time() - t0,
        }

    # --- pairwise explainers on the shared collapsed backbone --------------
    explain_steps = int(cfg["explain_steps"])
    explain_lr = float(cfg["explain_lr"])
    sparsity = float(cfg["sparsity"])
    budget = float(cfg["budget"])

    t0 = time.time()
    keep = optimize_pge_masks(
        gcn_backbone, x, collapsed, 1,
        steps=explain_steps, lr=float(cfg.get("pge_lr", 0.05)), sparsity=sparsity,
        artifact_dir=seed_dir("PGExplainer"),
    )
    results["PGExplainer"] = {
        "backbone_acc": acc_coll,
        **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
        "train_time_s": time.time() - t0,
    }

    for name, mode in [("RCExplainer", "rc"), ("CF2", "cf2")]:
        t0 = time.time()
        keep = optimize_edge_masks(
            gcn_backbone, x, collapsed, 1, mode,
            explain_steps, explain_lr, sparsity,
            artifact_dir=seed_dir(name),
        )
        results[name] = {
            "backbone_acc": acc_coll,
            **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
            "train_time_s": time.time() - t0,
        }

    t0 = time.time()
    keep = optimize_meg_masks(
        gcn_backbone, x, collapsed, 1,
        pop_size=int(cfg.get("meg_pop", 16)),
        generations=int(cfg.get("meg_generations", 25)),
        sparsity=sparsity,
        artifact_dir=seed_dir("MEG"),
    )
    results["MEG"] = {
        "backbone_acc": acc_coll,
        **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
        "train_time_s": time.time() - t0,
    }

    with torch.no_grad():
        y_orig = gcn_backbone(x, collapsed).argmax(dim=-1)
    for name in ["Saliency", "GradCAM", "IG"]:
        t0 = time.time()
        method = {"Saliency": "saliency", "GradCAM": "gradcam", "IG": "ig"}[name]
        scores = gradient_edge_scores(gcn_backbone, x, collapsed, y_orig, test_mask, method=method)
        keep = topk_keep_masks(collapsed, scores, budget)
        results[name] = {
            "backbone_acc": acc_coll,
            **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
            "train_time_s": time.time() - t0,
        }

    t0 = time.time()
    keep = random_keep_masks(collapsed, budget)
    results["Random"] = {
        "backbone_acc": acc_coll,
        **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
        "train_time_s": time.time() - t0,
    }

    if cfg.get("with_subgraphx", False):
        t0 = time.time()
        keep = optimize_subgraphx_masks(
            gcn_backbone, x, collapsed, y_orig, test_mask,
            n_targets=int(cfg.get("sx_n_targets", 150)),
            mcts_iters=int(cfg.get("sx_mcts_iters", 20)),
            n_coalitions=int(cfg.get("sx_coalitions", 3)),
            surrogate_epochs=int(cfg.get("sx_surrogate_epochs", 60)),
            artifact_dir=seed_dir("SubgraphX"),
        )
        results["SubgraphX"] = {
            "backbone_acc": acc_coll,
            **compute_cf_metrics(gcn_backbone, x, collapsed, y, test_mask, keep),
            "train_time_s": time.time() - t0,
        }

    # --- heterogeneous-graph baselines on the shared hetero backbone -------
    for name, mode in [("GNNExplainer-hetero", "factual"), ("PNS-hetero", "pns")]:
        t0 = time.time()
        keep = optimize_edge_masks(
            hetero, x, eid, nr, mode,
            explain_steps, explain_lr, sparsity,
            artifact_dir=seed_dir(name),
        )
        results[name] = {
            "backbone_acc": acc_hetero,
            **compute_cf_metrics(hetero, x, eid, y, test_mask, keep),
            "train_time_s": time.time() - t0,
        }

    return results


def main() -> None:
    import yaml
    PROJECT = PROJECT_ROOT

    def _load_yaml(filename: str) -> Dict:
        with open(os.path.join(PROJECT, "configs", filename), encoding="utf-8") as f:
            return yaml.safe_load(f)

    real_cfg = _load_yaml("real_experiment.yaml")
    data_cfg = _load_yaml("data.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=real_cfg["dataset"], choices=["acm", "cora", "mag"])
    ap.add_argument("--cap", type=int, default=int(real_cfg["cap"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(real_cfg["seeds"]))
    ap.add_argument("--hidden_dim", type=int, default=int(real_cfg["hidden_dim"]))
    ap.add_argument("--num_layers", type=int, default=int(real_cfg["num_layers"]))
    ap.add_argument("--backbone_steps", type=int, default=int(real_cfg["backbone_steps"]))
    ap.add_argument("--explain_steps", type=int, default=int(real_cfg["explain_steps"]))
    ap.add_argument("--lr", type=float, default=float(real_cfg["lr"]))
    ap.add_argument("--explain_lr", type=float, default=float(real_cfg["explain_lr"]))
    ap.add_argument("--sparsity", type=float, default=float(real_cfg["sparsity"]))
    ap.add_argument("--dropout", type=float, default=float(real_cfg["dropout"]))
    ap.add_argument("--weight_decay", type=float, default=float(real_cfg["weight_decay"]))
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.30,
                    help="edge-removal budget (fraction) for gradient/random baselines")
    ap.add_argument("--with_subgraphx", action="store_true")
    ap.add_argument("--sx_n_targets", type=int, default=150)
    ap.add_argument("--sx_mcts_iters", type=int, default=20)
    ap.add_argument("--sx_coalitions", type=int, default=3)
    ap.add_argument("--sx_surrogate_epochs", type=int, default=60)
    ap.add_argument("--meg_pop", type=int, default=16)
    ap.add_argument("--meg_generations", type=int, default=25)
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--no_regularization", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, dataset={args.dataset}", flush=True)

    if args.dataset == "cora":
        from model.data.cora import load_cora
        data = load_cora(cache_dir=data_cfg["cora"]["cache_dir"], cap=args.cap)
    elif args.dataset == "acm":
        from model.data.acm_han import load_acm_han
        data = load_acm_han(
            mat_path=data_cfg["acm_han"]["mat_path"],
            cap=args.cap,
            seed=data_cfg["acm_han"]["seed"],
        )
    else:
        from model.data.ogbn_mag import load_mag
        data = load_mag(root=data_cfg["mag"]["root"], n_papers=data_cfg["mag"]["n_papers"],
                        n_venues=data_cfg["mag"]["n_venues"], cap=args.cap,
                        seed=data_cfg["mag"]["seed"])
    print(f"[info] {args.dataset}: N={data['x'].size(0)}, "
          f"edges r0={data['edge_index_dict']['0'].size(1)}, "
          f"r1={data['edge_index_dict']['1'].size(1)}, classes={int(data['y'].max())+1}", flush=True)

    cfg = vars(args)
    cfg["early_stop"] = True
    cfg["artifacts_root"] = os.path.join(PROJECT, f"results_{args.tag}")
    cfg["ckpt_root"] = os.path.join(PROJECT, f"checkpoints_{args.tag}")
    os.makedirs(cfg["artifacts_root"], exist_ok=True)
    os.makedirs(cfg["ckpt_root"], exist_ok=True)

    per_seed = {s: run_seed(s, data, device, cfg) for s in args.seeds}
    out = {
        "dataset": args.dataset,
        "config": {k: v for k, v in cfg.items() if k != "seeds"},
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
    }
    out_path = os.path.join(cfg["artifacts_root"], f"extended_baselines_{args.dataset}.json")
    _write_json(out_path, out)
    print(f"[info] saved {out_path}", flush=True)

    methods = list(per_seed[args.seeds[0]].keys())
    metrics = ["backbone_acc", "csr", "minimality", "ps", "pns"]
    print(f"\n===== Extended baselines ({args.dataset}, mean±std over {args.seeds}) =====")
    for m in methods:
        parts = []
        for k in metrics:
            vals = [per_seed[s][m].get(k) for s in args.seeds]
            vals = [v for v in vals if v is not None]
            parts.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "-")
        print(f"{m:24s} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
