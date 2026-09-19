# RACE: 异构图关系感知反事实解释

RACE（Relation-Aware Counterfactual Explanation）是一个面向异构图神经网络的反事实解释框架。给定一个已训练的节点分类模型，RACE 在**关系类型**层面回答"移除哪些类型的边会改变模型的预测"，并支持从关系类型到具体边的分层下钻，兼顾解释的最小性、稳定性与计算效率。

## 项目架构

```
code_refactored/
├── model/                          # 核心包：骨干网络 + 解释器
│   ├── hetero_gnn.py               # RACE 异构 GNN 骨干（关系加权消息传递）
│   ├── han_hgt.py                  # HAN / HGT / GCN / GAT / RGCN 骨干
│   ├── counterfactual_explainer.py # 逐边反事实解释器（Gumbel-Sigmoid 掩码）
│   ├── relation_type_explainer.py  # 关系类型级解释器（穷举搜索 + 可微松弛）
│   ├── instance_relation_explainer.py  # 实例级关系搜索 S_v*（逐节点精确翻转）
│   ├── hierarchical_explainer.py   # 分层解释器（关系搜索 → 边级剪枝下钻）
│   └── data/                       # 数据集加载器
│       ├── cora.py                 # Cora（自动下载，2 关系）
│       ├── acm_han.py              # ACM（HAN 版，PAP/PTP 2 关系）
│       ├── hgb.py                  # HGB ACM / DBLP（PyG 自动下载）
│       ├── ogbn_mag.py / mag4.py   # ogbn-mag 子采样（2 / 4 关系）
│       ├── arxiv.py                # ogbn-arxiv（2 关系）
│       ├── dblp_classic.py         # DBLP classic
│       └── synthetic.py / scm_synthetic.py  # 合成数据（含真值因果关系）
├── benchmarks/
│   └── baselines.py                # 基线：GNNExplainer / CF-GNNExplainer / PNS /
│                                   # PGExplainer / RCExplainer / CF² / MEG /
│                                   # SubgraphX / GradCAM / IG / RACE-v2 等
├── analysis/
│   └── metrics.py                  # 评估指标（accuracy、CSR、PS/PNS 等）
├── configs/
│   ├── data.yaml                   # 数据集路径与采样参数
│   └── real_experiment.yaml        # 实验默认超参数（作为 CLI 默认值加载）
├── utils.py                        # 公共工具：种子设置、骨干训练、模型构建、数据加载
├── real_data_experiment.py         # 入口①：真实数据五方法主对比实验
├── synthetic_experiment.py         # 入口②：合成数据真值验证（A/B/C 三部分）
├── instance_experiment.py          # 入口③：实例级关系反事实搜索（训练并冻结骨干）
├── race_v2_experiment.py           # 入口④：RACE-v2 逐边解释器（margin 损失 + 离散验证 + 剪枝）
├── hierarchical_experiment.py      # 入口⑤：分层解释 vs 扁平基线
├── baselines_experiment.py         # 入口⑥：扩展基线对比 + 骨干精度对照
├── requirements.txt
└── README.md
```

**运行流程**：每个入口脚本负责"训练骨干 → 冻结 → 在其上运行解释器 → 计算指标"的完整流水线。训练产物按 seed 分目录保存：指标写入 `results{tag}/seed_{s}/...`，模型权重写入 `checkpoints{tag}/seed_{s}/...`（运行时自动创建，均已约定为不入库的中间产物）。

## 环境安装

```bash
pip install -r requirements.txt
```

依赖：`torch`、`torch_geometric`、`numpy`、`scipy`、`PyYAML`。

- 如需运行 `mag` / `mag4` / `arxiv` 数据集，额外安装 `pip install ogb`（懒加载，仅这些数据集需要）。
- GPU 非必需，所有脚本支持 `--device cpu`（默认 `auto` 自动检测）。

## 数据准备

数据默认存放于 `data/` 目录（相对于项目根，路径可在 `configs/data.yaml` 中修改）：

| 数据集 | 获取方式 |
|---|---|
| Cora | 首次运行时自动从 GitHub 下载 |
| HGB ACM / DBLP | 由 PyG `HGBDataset` 自动下载 |
| ogbn-mag / ogbn-arxiv | 由 `ogb` 库自动下载（需先安装 `ogb`） |
| ACM（HAN 版） | 手动下载 `ACM.mat`（托管于 `data.dgl.ai`）并放到 `data/ACM.mat` |
| 合成数据 | 代码内生成，无需下载 |

## 快速启动

在项目根目录（`code_refactored/`）下运行。所有入口均支持 `--seeds`、`--dataset`、`--device` 等参数，默认值取自 `configs/real_experiment.yaml`，可用 `--help` 查看完整参数列表。

```bash
# ① 真实数据主对比（GNNExplainer / CF-GNNExplainer / PNS / Ours 两种粒度）
python real_data_experiment.py --dataset acm --seeds 0 1 2 3 4

# ② 合成数据真值验证（关系识别命中率、五方法对比、PN/PS/PNS 验证）
python synthetic_experiment.py --part ABC --seeds 0 1 2 3 4

# ③ 实例级关系搜索（同时产出冻结骨干 checkpoints{tag}/seed_{s}/backbone.pt）
python instance_experiment.py --dataset acm --seeds 0 1 2 --tag v2

# ④ RACE-v2 逐边解释器（复用 ③ 的冻结骨干）
python race_v2_experiment.py --dataset acm --seeds 0 1 2 --tag v2 --variant v0 --reuse_ckpt

# ⑤ 分层解释 vs 扁平基线
python hierarchical_experiment.py --dataset acm --seeds 0 1 2 --tag v2

# ⑥ 扩展基线（PGExplainer / RCExplainer / CF² / MEG / 梯度归因等）
python baselines_experiment.py --dataset acm --seeds 0 1 2 3 4
```

可选数据集：`acm`（默认）、`cora`、`mag`、`ACM`、`DBLP`（HGB）、`mag4`、`arxiv`、`dblp`（各脚本支持范围略有差异，以 `--help` 为准）。可选骨干：`hetero`（RACE 默认）、`han`、`hgt`、`rgcn`。
