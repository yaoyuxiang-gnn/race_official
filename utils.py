"""Shared training utilities: random seeding and backbone training.

Follows spec.md Section 3/4 artifact protocol:
- results/{seed_id}/{method}/train_log.json      (append per epoch, JSON lines)
- results/{seed_id}/{method}/val_metrics_epoch{N}.json
- results/{seed_id}/{method}/test_metrics_epoch{N}.json
- checkpoints/{seed_id}/{method}/epoch_{N}.pt
  (model_state_dict + optimizer_state_dict + epoch + best_val_metric)

For high-volume diagnostic runs (sweeps, ablations) set ``full_artifacts=False``:
per-epoch metric files are still written, but only the final checkpoint is kept
(documented deviation to keep disk usage tractable; see experiment_plan.md E5).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def set_seed(seed: int) -> None:
    """Seed PyTorch and NumPy for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: str, record: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _save_epoch_artifacts(
    artifact_dir: Optional[str],
    epoch: int,
    loss: float,
    val_acc: Optional[float],
    test_acc: Optional[float],
    val_extra: Optional[Dict[str, float]] = None,
    test_extra: Optional[Dict[str, float]] = None,
) -> None:
    """spec.md Step 1-3: append train log, write val/test metric JSONs."""
    if artifact_dir is None:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    _append_jsonl(os.path.join(artifact_dir, "train_log.json"), {"epoch": epoch, "loss": loss})
    if val_acc is not None:
        rec = {"epoch": epoch, "accuracy": val_acc}
        if val_extra:
            rec.update(val_extra)
        _write_json(os.path.join(artifact_dir, f"val_metrics_epoch{epoch}.json"), rec)
    if test_acc is not None:
        rec = {"epoch": epoch, "accuracy": test_acc}
        if test_extra:
            rec.update(test_extra)
        _write_json(os.path.join(artifact_dir, f"test_metrics_epoch{epoch}.json"), rec)


def _save_checkpoint(ckpt_dir: str, epoch: int, model, optimizer, best_val: float) -> None:
    """spec.md Step 4: full checkpoint with the four required fields."""
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_metric": best_val,
        },
        os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
    )


def train_backbone(
    model: torch.nn.Module,
    x: Tensor,
    edge_index_dict: Dict[str, Tensor],
    y: Tensor,
    train_mask: Tensor,
    val_mask: Optional[Tensor] = None,
    test_mask: Optional[Tensor] = None,
    steps: int = 300,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    early_stop: bool = True,
    artifact_dir: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    full_artifacts: bool = True,
    resume_path: Optional[str] = None,
    val_extra_fn: Optional[Callable] = None,
    test_extra_fn: Optional[Callable] = None,
) -> Tuple[torch.nn.Module, List[float]]:
    """Train a node-classification backbone, saving per-epoch artifacts.

    Uses L2 regularisation and, when ``val_mask`` is provided and ``early_stop``,
    keeps the best checkpoint by validation accuracy (early stopping against
    over-fitting on the small semi-supervised split).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 0
    best_state, best_val = None, -1.0
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=x.device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_val = float(ckpt.get("best_val_metric", -1.0))
        print(f"[resume] loaded epoch {start_epoch}, best_val={best_val}")

    def _eval(mask: Tensor) -> float:
        with torch.no_grad():
            pred = model(x, edge_index_dict).argmax(dim=-1)
        return float((pred[mask] == y[mask]).float().mean().item())

    losses: List[float] = []
    for step in range(1, steps + 1):
        epoch = start_epoch + step
        opt.zero_grad()
        logits = model(x, edge_index_dict)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        loss_val = float(loss.item())
        losses.append(loss_val)

        val_acc = _eval(val_mask) if val_mask is not None else None
        test_acc = _eval(test_mask) if test_mask is not None else None
        val_extra = val_extra_fn(model) if val_extra_fn is not None else None
        test_extra = test_extra_fn(model) if test_extra_fn is not None else None

        _save_epoch_artifacts(artifact_dir, epoch, loss_val, val_acc, test_acc, val_extra, test_extra)
        if ckpt_dir is not None and full_artifacts:
            _save_checkpoint(ckpt_dir, epoch, model, opt, best_val)

        if val_acc is not None:
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}

    if ckpt_dir is not None and not full_artifacts:
        # final checkpoint only (sweep/diagnostic mode)
        _save_checkpoint(ckpt_dir, start_epoch + steps, model, opt, best_val)

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    return model, losses


def load_dataset(name: str, data_cfg: Dict, cap: int = 50000) -> Dict:
    """Load a benchmark dataset by name using the paths in ``data.yaml``.

    Returns the dict {x, y, train_mask, val_mask, test_mask, edge_index_dict}
    used by every experiment script (lazy imports avoid import cycles).
    """
    if name == "cora":
        from model.data.cora import load_cora
        return load_cora(cache_dir=data_cfg["cora"]["cache_dir"], cap=cap)
    if name == "acm":
        from model.data.acm_han import load_acm_han
        return load_acm_han(
            mat_path=data_cfg["acm_han"]["mat_path"],
            cap=cap,
            seed=int(data_cfg["acm_han"]["seed"]),
        )
    if name == "mag":
        from model.data.ogbn_mag import load_mag
        return load_mag(
            root=data_cfg["mag"]["root"],
            n_papers=int(data_cfg["mag"]["n_papers"]),
            n_venues=int(data_cfg["mag"]["n_venues"]),
            cap=cap,
            seed=int(data_cfg["mag"]["seed"]),
        )
    if name in ("ACM", "DBLP"):
        from model.data.hgb import load_acm, load_dblp
        loader = load_acm if name == "ACM" else load_dblp
        return loader(root=data_cfg["hgb"]["root"], cap=cap, seed=int(data_cfg["hgb"]["seed"]))
    if name == "arxiv":
        from model.data.arxiv import load_arxiv
        return load_arxiv(root=data_cfg["arxiv"]["root"], n_nodes=data_cfg["arxiv"]["n_nodes"],
                          n_classes=int(data_cfg["arxiv"]["n_classes"]), cap=cap,
                          seed=int(data_cfg["arxiv"]["seed"]))
    if name == "dblp":
        from model.data.dblp_classic import load_dblp_classic
        return load_dblp_classic(root=data_cfg["dblp"]["root"], topk=int(data_cfg["dblp"]["topk"]),
                                 cap=cap, seed=int(data_cfg["dblp"]["seed"]))
    if name == "mag4":
        # C2: natural 4-relation variant of ogbn-mag (PAP/citation/PFP/PAIP);
        # same paper sample as "mag" (relation-vocabulary ablation)
        from model.data.mag4 import load_mag4
        return load_mag4(
            root=data_cfg["mag"]["root"],
            n_papers=int(data_cfg["mag"]["n_papers"]),
            n_venues=int(data_cfg["mag"]["n_venues"]),
            cap=cap,
            seed=int(data_cfg["mag"]["seed"]),
        )
    raise ValueError(f"unknown dataset: {name}")


def build_backbone(
    kind: str,
    in_dim: int,
    out_dim: int,
    num_relations: int,
    hidden_dim: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.5,
    use_rel_weights: bool = True,
    device=None,
):
    """Build a heterogeneous backbone by kind: RACE backbone, HAN, HGT or RGCN."""
    from model.han_hgt import HAN, HGT, RGCN
    from model.hetero_gnn import HeteroGNN
    if kind == "han":
        return HAN(in_dim, hidden_dim, out_dim, num_layers, num_relations,
                   num_heads=num_heads, dropout=dropout).to(device)
    if kind == "hgt":
        return HGT(in_dim, hidden_dim, out_dim, num_layers, num_relations,
                   num_heads=num_heads, dropout=dropout).to(device)
    if kind == "rgcn":
        return RGCN(in_dim, hidden_dim, out_dim, num_layers, num_relations,
                    dropout=dropout).to(device)
    return HeteroGNN(in_dim, hidden_dim, out_dim, num_layers, num_relations,
                     dropout=dropout, use_rel_weights=use_rel_weights).to(device)
