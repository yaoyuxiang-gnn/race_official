"""Real-data baseline comparison for the relation-aware counterfactual explainer.

Runs on the HGB ACM / DBLP heterogeneous graphs, projected to a single-node-type
multi-relation graph via meta-paths.  Compares three pairwise baselines against
two variants of our relation-aware explainer:

* GNNExplainer    — factual importance on the collapsed (pairwise) graph;
* CF-GNNExplainer — counterfactual flip on the collapsed graph;
* PNS             — necessity + sufficiency on the collapsed graph;
* Ours (per-edge) — relation-aware counterfactual masks (one mask per edge);
* Ours (rel-type) — relation-TYPE-level counterfactual masks (exhaustive search).

Metrics are identical across methods: backbone accuracy, counterfactual success
rate (CSR), minimality (fraction of edges removed), and PS / PNS.  For the
relation-type explainer we additionally report per-relation flip rates, the
recovered minimal causal set S* under several thresholds tau, and the
single-relation argmax criterion (Sec. E3 of experiment_plan.md).

Artifact protocol (spec.md Sec. 3/4): per (seed, method) the backbone training
loop appends ``results/{seed}/{method}/train_log.json``, writes
``val_metrics_epoch{N}.json`` / ``test_metrics_epoch{N}.json`` and checkpoints
``checkpoints/{seed}/{method}/epoch_{N}.pt`` (full state incl. optimizer).
Explainers append their optimisation loss to the same train log; their final
masks and metrics are saved under ``results/{seed}/{method}/``.
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
from benchmarks.baselines import collapse, optimize_edge_masks
from model.han_hgt import HAN, HGT
from model.hetero_gnn import HeteroGNN
from model.relation_type_explainer import RelationTypeExplainer
from utils import set_seed, train_backbone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def relation_flip_rates(backbone, x, edge_index_dict, target_nodes):
    """Per-relation and full-removal flip rates on ``target_nodes`` (Eq. 4)."""
    K = len(edge_index_dict)
    with torch.no_grad():
        y_full = backbone(x, edge_index_dict).argmax(dim=-1)
        if target_nodes is not None:
            y_full = y_full[target_nodes]

        def flip(subset) -> float:
            masks = {str(r): torch.ones(edge_index_dict[str(r)].size(1), device=x.device) for r in range(K)}
            for r in subset:
                masks[str(r)] = torch.zeros(edge_index_dict[str(r)].size(1), device=x.device)
            y_cf = backbone(x, edge_index_dict, edge_mask_dict=masks).argmax(dim=-1)
            if target_nodes is not None:
                y_cf = y_cf[target_nodes]
            return float((y_cf != y_full).float().mean().item())

    per_rel = {int(r): flip((r,)) for r in range(K)}
    full = flip(tuple(range(K)))
    return per_rel, full


def run_seed(seed: int, data: Dict, device: torch.device, cfg: Dict) -> Dict:
    """Run one seed of the five-method comparison. ``cfg`` must contain every
    hyperparameter key (see argparse defaults in ``main``)."""
    set_seed(seed)
    x = data["x"].to(device)
    y = data["y"].to(device)
    eid = {k: v.to(device) for k, v in data["edge_index_dict"].items()}
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)
    nr = len(eid)
    out_dim = int(y.max().item()) + 1
    full_artifacts = bool(cfg.get("full_artifacts", True))
    use_rel_weights = bool(cfg.get("use_rel_weights", True))
    no_reg = bool(cfg.get("no_regularization", False))
    early_stop = bool(cfg.get("early_stop", True)) and not no_reg
    results: Dict[str, Dict] = {}
    artifacts_root = cfg.get("artifacts_root") or os.path.join(PROJECT_ROOT, "results")
    ckpt_root = cfg.get("ckpt_root") or os.path.join(PROJECT_ROOT, "checkpoints")
    seed_dir = lambda m: os.path.join(artifacts_root, f"seed_{seed}", m)
    ckpt_dir_fn = lambda m: os.path.join(ckpt_root, f"seed_{seed}", m)

    def train_and_eval(model, edges, method):
        resume_path = None
        if cfg.get("resume", False):
            ckpt_dir = ckpt_dir_fn(method)
            ckpts = sorted(
                (f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")),
                key=lambda f: int(f[len("epoch_"):-len(".pt")]),
            ) if os.path.isdir(ckpt_dir) else []
            if ckpts:
                resume_path = os.path.join(ckpt_dir, ckpts[-1])
        train_backbone(
            model, x, edges, y, train_mask, val_mask, test_mask,
            steps=int(cfg["backbone_steps"]), lr=float(cfg["lr"]),
            weight_decay=0.0 if no_reg else float(cfg["weight_decay"]),
            early_stop=early_stop,
            artifact_dir=seed_dir(method),
            ckpt_dir=ckpt_dir_fn(method),
            full_artifacts=full_artifacts,
            resume_path=resume_path,
        )
        model.eval()
        return accuracy(model, x, edges, y, test_mask)

    # --- Ours: heterogeneous backbone, per-edge + rel-type ----------------
    t0 = time.time()
    backbone_kind = cfg.get("backbone", "hetero")
    num_heads = int(cfg.get("num_heads", 4))
    if backbone_kind == "han":
        hetero = HAN(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
                     num_heads=num_heads, dropout=0.0 if no_reg else cfg["dropout"]).to(device)
    elif backbone_kind == "hgt":
        hetero = HGT(x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
                     num_heads=num_heads, dropout=0.0 if no_reg else cfg["dropout"]).to(device)
    else:
        hetero = HeteroGNN(
            x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], nr,
            dropout=0.0 if no_reg else cfg["dropout"], use_rel_weights=use_rel_weights,
        ).to(device)
    acc_hetero = train_and_eval(hetero, eid, "Ours(backbone)")
    t1 = time.time()

    keep_edge = optimize_edge_masks(
        hetero, x, eid, nr, "counterfactual",
        int(cfg["explain_steps"]), float(cfg["explain_lr"]), float(cfg["sparsity"]),
        artifact_dir=seed_dir("Ours(per-edge)"),
        gumbel_tau=float(cfg.get("gumbel_tau", 1.0)),
    )
    torch.save({k: v.cpu() for k, v in keep_edge.items()},
               os.path.join(seed_dir("Ours(per-edge)"), "edge_masks.pt"))
    results["Ours(per-edge)"] = {
        "backbone_acc": acc_hetero,
        **compute_cf_metrics(hetero, x, eid, y, test_mask, keep_edge),
        "params": sum(p.numel() for p in hetero.parameters()),
        "train_time_s": t1 - t0,
        "rel_weights": ([float(w) for w in hetero.layers[0].rel_weight.detach().cpu()]
                        if hasattr(hetero.layers[0], "rel_weight") else []),
    }

    rte = RelationTypeExplainer(hetero, nr, gumbel_tau=1.0, init_logit=3.0)
    per_rel, full_flip = relation_flip_rates(hetero, x, eid, test_mask)
    attribution = {"flip_per_relation": per_rel, "flip_full_removal": full_flip}
    for tau in cfg.get("tau_rel_scan", [0.9]):
        try:
            attribution[f"S*_tau_{tau}"] = rte.explain_exhaustive(x, eid, target_nodes=test_mask, rel_threshold=float(tau))
        except Exception as e:  # exhaustive search must never fail, but guard anyway
            attribution[f"S*_tau_{tau}"] = f"error: {e}"
    argmax_rel = int(max(per_rel, key=per_rel.get))
    attribution["S*_argmax"] = [argmax_rel] if full_flip > 0 else []
    removed_types = attribution[f"S*_tau_{cfg.get('tau_rel', 0.9)}"]
    keep_type = {str(r): torch.ones(eid[str(r)].size(1), device=device) for r in range(nr)}
    for r in removed_types:
        keep_type[str(r)] = torch.zeros(eid[str(r)].size(1), device=device)
    results["Ours(rel-type)"] = {
        "backbone_acc": acc_hetero,
        **compute_cf_metrics(hetero, x, eid, y, test_mask, keep_type),
        "params": sum(p.numel() for p in hetero.parameters()),
        "train_time_s": time.time() - t1,
        "n_types_removed": len(removed_types),
        "removed_types": removed_types,
    }
    results["Ours(rel-type)"]["attribution"] = attribution
    _write_json(os.path.join(seed_dir("Ours(rel-type)"), "attribution.json"), attribution)

    # --- Pairwise baselines on the collapsed graph ------------------------
    collapsed = collapse(eid)
    for name, mode in [("GNNExplainer", "factual"), ("CF-GNNExplainer", "counterfactual"), ("PNS", "pns")]:
        if cfg.get("skip_baselines", False):
            results[name] = {"skipped": True}
            continue
        t0 = time.time()
        gcn = HeteroGNN(
            x.size(1), cfg["hidden_dim"], out_dim, cfg["num_layers"], 1,
            dropout=0.0 if no_reg else cfg["dropout"],
        ).to(device)
        acc_gcn = train_and_eval(gcn, collapsed, name)
        keep = optimize_edge_masks(
            gcn, x, collapsed, 1, mode,
            int(cfg["explain_steps"]), float(cfg["explain_lr"]), float(cfg["sparsity"]),
            artifact_dir=seed_dir(name),
        )
        results[name] = {
            "backbone_acc": acc_gcn,
            **compute_cf_metrics(gcn, x, collapsed, y, test_mask, keep),
            "params": sum(p.numel() for p in gcn.parameters()),
            "train_time_s": time.time() - t0,
        }
    return results


def main() -> None:
    import yaml
    def _load_yaml(filename: str) -> Dict:
        with open(os.path.join(PROJECT_ROOT, "configs", filename), encoding="utf-8") as f:
            return yaml.safe_load(f)

    real_cfg = _load_yaml("real_experiment.yaml")
    data_cfg = _load_yaml("data.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=real_cfg["dataset"], choices=["acm", "cora", "ACM", "DBLP", "mag"])
    ap.add_argument("--cap", type=int, default=int(real_cfg["cap"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(real_cfg["seeds"]))
    ap.add_argument("--hidden_dim", type=int, default=int(real_cfg["hidden_dim"]))
    ap.add_argument("--num_layers", type=int, default=int(real_cfg["num_layers"]))
    ap.add_argument("--backbone_steps", type=int, default=int(real_cfg["backbone_steps"]))
    ap.add_argument("--explain_steps", type=int, default=int(real_cfg["explain_steps"]))
    ap.add_argument("--lr", type=float, default=float(real_cfg["lr"]))
    ap.add_argument("--explain_lr", type=float, default=float(real_cfg["explain_lr"]))
    ap.add_argument("--sparsity", type=float, default=float(real_cfg["sparsity"]))
    ap.add_argument("--gumbel_tau", type=float, default=float(real_cfg["gumbel_tau"]))
    ap.add_argument("--dropout", type=float, default=float(real_cfg["dropout"]))
    ap.add_argument("--weight_decay", type=float, default=float(real_cfg["weight_decay"]))
    ap.add_argument("--tau_rel", type=float, default=float(real_cfg["tau_rel"]))
    ap.add_argument("--tau_rel_scan", type=float, nargs="+", default=list(real_cfg["tau_rel_scan"]))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_regularization", action="store_true", help="ablation E4d: dropout=0, wd=0, no early stop")
    ap.add_argument("--no_rel_weights", action="store_true", help="ablation E4e: fix relation attention at 1")
    ap.add_argument("--no_full_artifacts", action="store_true", help="sweep mode: metrics per epoch, final checkpoint only")
    ap.add_argument("--resume", action="store_true", help="resume each backbone from its latest epoch checkpoint (spec.md Sec. 5)")
    ap.add_argument("--backbone", default="hetero", choices=["hetero", "han", "hgt"],
                    help="backbone for Ours(per-edge) / Ours(rel-type): RACE backbone, HAN, or HGT")
    ap.add_argument("--num_heads", type=int, default=4, help="attention heads for HAN/HGT backbones")
    ap.add_argument("--skip_baselines", action="store_true",
                    help="skip the pairwise baselines (backbone-agnostic; reused from results/)")
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
    elif args.dataset == "mag":
        from model.data.ogbn_mag import load_mag
        data = load_mag(root=data_cfg["mag"]["root"], n_papers=data_cfg["mag"]["n_papers"],
                        n_venues=data_cfg["mag"]["n_venues"], cap=args.cap,
                        seed=data_cfg["mag"]["seed"])
    else:
        from model.data.hgb import load_acm, load_dblp
        loader = load_acm if args.dataset == "ACM" else load_dblp
        data = loader(root=data_cfg["hgb"]["root"], cap=args.cap, seed=data_cfg["hgb"]["seed"])
    print(f"[info] {args.dataset}: N={data['x'].size(0)}, "
          f"edges r0={data['edge_index_dict']['0'].size(1)}, "
          f"r1={data['edge_index_dict']['1'].size(1)}, classes={int(data['y'].max())+1}", flush=True)

    cfg = vars(args)
    cfg["use_rel_weights"] = not args.no_rel_weights
    cfg["early_stop"] = True
    cfg["full_artifacts"] = not args.no_full_artifacts
    cfg["skip_baselines"] = args.skip_baselines
    tag_suffix = f"_{args.tag}" if args.tag else ""
    cfg["artifacts_root"] = os.path.join(PROJECT_ROOT, f"results{tag_suffix}")
    cfg["ckpt_root"] = os.path.join(PROJECT_ROOT, f"checkpoints{tag_suffix}")
    os.makedirs(cfg["artifacts_root"], exist_ok=True)
    os.makedirs(cfg["ckpt_root"], exist_ok=True)
    per_seed = {s: run_seed(s, data, device, cfg) for s in args.seeds}
    methods = ["GNNExplainer", "CF-GNNExplainer", "PNS", "Ours(per-edge)", "Ours(rel-type)"]
    metrics = ["backbone_acc", "csr", "minimality", "ps", "pns"]
    # Drop methods that were not run (--skip_baselines marks them {"skipped": True}).
    first = per_seed[args.seeds[0]]
    methods = [m for m in methods if isinstance(first.get(m), dict) and "backbone_acc" in first[m]]

    print(f"\n===== Real-data results ({args.dataset}, mean±std over {args.seeds}) =====")
    print("| Method | " + " | ".join(metrics) + " | Params | Time(s) |")
    print("|" + "---|" * (len(metrics) + 2) + "|")
    for m in methods:
        vals = {k: [per_seed[s][m][k] for s in args.seeds] for k in metrics}
        ms = [f"{np.mean(vals[k]):.4f}±{np.std(vals[k]):.4f}" for k in metrics]
        p = per_seed[args.seeds[0]][m]["params"]
        t = f"{np.mean([per_seed[s][m]['train_time_s'] for s in args.seeds]):.2f}"
        print(f"| {m} | " + " | ".join(ms) + f" | {p} | {t} |")

    for s in args.seeds:
        rt = per_seed[s]["Ours(rel-type)"]
        print(f"[info] seed {s}: attribution={rt.get('attribution', {})}")

    out = {
        "dataset": args.dataset,
        "config": {k: v for k, v in cfg.items() if k != "seeds"},
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
    }
    out_path = os.path.join(cfg["artifacts_root"], "real_experiment_results.json")
    _write_json(out_path, out)
    print(f"[info] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
