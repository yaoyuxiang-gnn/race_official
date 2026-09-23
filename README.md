# RACE: Relation-Aware Counterfactual Explanation for Heterogeneous Graphs

RACE (Relation-Aware Counterfactual Explanation) is a counterfactual explanation framework for heterogeneous graph neural networks. Given a trained node classification model, RACE answers at the **relation type** level the question "removing which types of edges would change the model's prediction", and supports hierarchical drill-down from relation types to concrete edges, balancing minimality, stability, and computational efficiency of the explanations.

## Project Architecture

```
code_refactored/
├── model/                          # Core package: backbones + explainers
│   ├── hetero_gnn.py               # RACE heterogeneous GNN backbone (relation-weighted message passing)
│   ├── han_hgt.py                  # HAN / HGT / GCN / GAT / RGCN backbones
│   ├── counterfactual_explainer.py # Per-edge counterfactual explainer (Gumbel-Sigmoid mask)
│   ├── relation_type_explainer.py  # Relation-type-level explainer (exhaustive search + differentiable relaxation)
│   ├── instance_relation_explainer.py  # Instance-level relation search S_v* (exact per-node flip)
│   ├── hierarchical_explainer.py   # Hierarchical explainer (relation search → edge-level pruning drill-down)
│   └── data/                       # Dataset loaders
│       ├── cora.py                 # Cora (auto-download, 2 relations)
│       ├── acm_han.py              # ACM (HAN version, PAP/PTP 2 relations)
│       ├── hgb.py                  # HGB ACM / DBLP (auto-download via PyG)
│       ├── ogbn_mag.py / mag4.py   # ogbn-mag subsampling (2 / 4 relations)
│       ├── arxiv.py                # ogbn-arxiv (2 relations)
│       ├── dblp_classic.py         # DBLP classic
│       └── synthetic.py / scm_synthetic.py  # Synthetic data (with ground-truth causality)
├── benchmarks/
│   └── baselines.py                # Baselines: GNNExplainer / CF-GNNExplainer / PNS /
│                                   # PGExplainer / RCExplainer / CF² / MEG /
│                                   # SubgraphX / GradCAM / IG / RACE-v2, etc.
├── analysis/
│   └── metrics.py                  # Evaluation metrics (accuracy, CSR, PS/PNS, etc.)
├── configs/
│   ├── data.yaml                   # Dataset paths and sampling parameters
│   └── real_experiment.yaml        # Default experiment hyperparameters (loaded as CLI defaults)
├── utils.py                        # Common utilities: seeding, backbone training, model building, data loading
├── real_data_experiment.py         # Entry ①: main real-data comparison of five methods
├── synthetic_experiment.py         # Entry ②: synthetic ground-truth validation (parts A/B/C)
├── instance_experiment.py          # Entry ③: instance-level relation counterfactual search (train and freeze backbone)
├── race_v2_experiment.py           # Entry ④: RACE-v2 per-edge explainer (margin loss + discrete validation + pruning)
├── hierarchical_experiment.py      # Entry ⑤: hierarchical explanation vs flat baselines
├── baselines_experiment.py         # Entry ⑥: extended baseline comparison + backbone accuracy reference
├── requirements.txt
└── README.md
```

**Workflow**: each entry script runs the full pipeline of "train backbone → freeze → run explainers on top → compute metrics". Training artifacts are saved in per-seed directories: metrics are written to `results{tag}/seed_{s}/...`, and model weights to `checkpoints{tag}/seed_{s}/...` (created automatically at runtime; both are designated intermediate artifacts that are not committed to the repo).

## Environment Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torch_geometric`, `numpy`, `scipy`, `PyYAML`.

- To run the `mag` / `mag4` / `arxiv` datasets, additionally install `pip install ogb` (lazy-loaded; only needed for these datasets).
- A GPU is not required; all scripts support `--device cpu` (default `auto` detects the device automatically).

## Data Preparation

Data is stored under the `data/` directory by default (relative to the project root; paths can be modified in `configs/data.yaml`):

| Dataset | How to obtain |
|---|---|
| Cora | Automatically downloaded from GitHub on first run |
| HGB ACM / DBLP | Automatically downloaded via PyG `HGBDataset` |
| ogbn-mag / ogbn-arxiv | Automatically downloaded via the `ogb` library (requires `ogb` to be installed first) |
| ACM (HAN version) | Manually download `ACM.mat` (hosted at `data.dgl.ai`) and place it at `data/ACM.mat` |
| Synthetic data | Generated in code, no download needed |

## Quick Start

Run from the project root (`code_refactored/`). All entries support arguments such as `--seeds`, `--dataset`, and `--device`; default values come from `configs/real_experiment.yaml`. Use `--help` to see the full argument list.

```bash
# ① Main real-data comparison (GNNExplainer / CF-GNNExplainer / PNS / Ours at two granularities)
python real_data_experiment.py --dataset acm --seeds 0 1 2 3 4

# ② Synthetic ground-truth validation (relation identification hit rate, five-method comparison, PN/PS/PNS validation)
python synthetic_experiment.py --part ABC --seeds 0 1 2 3 4

# ③ Instance-level relation search (also produces the frozen backbone checkpoints{tag}/seed_{s}/backbone.pt)
python instance_experiment.py --dataset acm --seeds 0 1 2 --tag v2

# ④ RACE-v2 per-edge explainer (reuses the frozen backbone from ③)
python race_v2_experiment.py --dataset acm --seeds 0 1 2 --tag v2 --variant v0 --reuse_ckpt

# ⑤ Hierarchical explanation vs flat baselines
python hierarchical_experiment.py --dataset acm --seeds 0 1 2 --tag v2

# ⑥ Extended baselines (PGExplainer / RCExplainer / CF² / MEG / gradient attribution, etc.)
python baselines_experiment.py --dataset acm --seeds 0 1 2 3 4
```

Available datasets: `acm` (default), `cora`, `mag`, `ACM`, `DBLP` (HGB), `mag4`, `arxiv`, `dblp` (each script supports a slightly different subset; refer to `--help`). Available backbones: `hetero` (RACE default), `han`, `hgt`, `rgcn`.
