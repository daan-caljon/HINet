# HINet: Heterogeneous Interference Network

Code for the paper **[Estimating Treatment Effects in Networks under Unknown Exposure Mappings](https://arxiv.org/abs/2510.21457)**
by Daan Caljon, Jente Van Belle, and Wouter Verbeke.

HINet estimates heterogeneous treatment effects under network interference without requiring a
prespecified exposure mapping. It combines an expressive GNN outcome model with network-aware
domain-adversarial training.

## Structure

```
├── config/
│   └── default.yaml              # All experiment parameters
├── data/
│   ├── simulated/                # Auto-generated at runtime
│   └── semi_synthetic/           # place BC, Flickr, and Coauthor-CS here (see Data)
├── scripts/
│   └── run.py                    # Main entry point
├── src/
│   ├── config.py                 # Config loading and CLI overrides
│   ├── data/
│   │   ├── data_generator.py     # Data-generating process
│   │   └── datatools.py          # Loaders for BC, Flickr, and Coauthor-CS
│   ├── models/
│   │   ├── __init__.py           # MODEL_REGISTRY and create_model()
│   │   ├── hinet.py              # HINet, HINet_no_net_conf
│   │   ├── netest.py             # NetEst, GINNetEst
│   │   ├── baselines.py          # GINModel, TARNet, GCN_DECONF
│   │   ├── spnet.py              # SPNet
│   │   ├── targeted.py           # TargetedModel_DoubleBSpline
│   │   ├── idenet.py             # IDENet
│   │   ├── components.py         # Shared Encoder, Predictor, Discriminator
│   │   └── layers.py             # GCN, GIN, GAT, GraphSAGE, GRL, Attention layers
│   ├── trainers/
│   │   ├── __init__.py           # TRAINER_REGISTRY and create_trainer()
│   │   ├── base.py               # Shared pipeline: tuning, alpha selection, evaluation
│   │   ├── grl_trainer.py        # HINet, HINet_no_net_conf
│   │   ├── adversarial_trainer.py# NetEst, GINNetEst
│   │   ├── simple_trainer.py     # GINModel, TARNet
│   │   ├── netdeconf_trainer.py  # GCN_DECONF
│   │   ├── spnet_trainer.py      # SPNet
│   │   ├── targeted_trainer.py   # TargetedModel_DoubleBSpline
│   │   └── idenet_trainer.py     # IDENet
│   ├── evaluation/
│   │   └── metrics.py            # CNEE and PEHNE implementations
│   └── utils/utils.py            # Normalization, Wasserstein distance
└── utils/utils.py                # Data loading and PyG Data construction
```

## Installation

All code was written for `python 3.12.3`. `requirements.txt` pins the exact package
versions used to produce the reported results.

```bash
pip install -r requirements.txt
```

## Data

The `full_sim` dataset is generated automatically at runtime and needs no download. Set
`data.homophily: true` for Homophily Sim or `false` for BA Sim.

The three semi-synthetic datasets are not redistributed here and must be placed under
`data/semi_synthetic/` before use. Only treatments and outcomes are simulated; the network
structures and node features are used as published.

**BC and Flickr.** The networks come from the [network-deconfounder-wsdm20](https://github.com/rguo12/network-deconfounder-wsdm20)
repository of Guo, Li & Liu (2020), used in the same preprocessed form as
[Jiang & Sun (2022)](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data),
who also provide the METIS train/validation/test partitions. A copy of the four files is
available [here](https://drive.google.com/drive/folders/1CGOKpd7NU-brk9PpiO6nJcVYp3idi97E?usp=sharing).
Place them as:

```
data/semi_synthetic/BC/BC0.mat
data/semi_synthetic/BC/BC_parts.pkl
data/semi_synthetic/Flickr/Flickr01.mat
data/semi_synthetic/Flickr/Flickr_parts.pkl
```

**Coauthor-CS.** Download `ms_academic_cs.npz` from the [gnn-benchmark](https://github.com/shchur/gnn-benchmark)
repository of Shchur et al. (2019) (MIT license) and place it at
`data/semi_synthetic/CS/ms_academic_cs.npz`:

```
https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz
```

(SHA-256 `933c745734e78908c7eb77172a673fee0503f30e26e5a99b36814d88534c42e3`, 12,835,626 bytes.)
The stored directed adjacency is symmetrised, binarised, and stripped of self-loops at load
time. The METIS partition `data/semi_synthetic/CS/CS_parts.pkl` ships with this repository;
all reported Coauthor-CS results depend on it.

## Usage

All parameters live in `config/default.yaml` and can be overridden on the command line with dotted keys.

```bash
# Run with the default config
python scripts/run.py

# Override any config value
python scripts/run.py model.type=NetEst data.dataset=BC

# Run the W&B sweep defined in the sweep: section of the config
python scripts/run.py --sweep

# Use a custom config file
python scripts/run.py --config config/my_experiment.yaml
```

Results are logged to Weights & Biases under the project set in `experiment.wandb_project`; each run's summary contains `avg_cnee`, `st_dev_cnee`, `avg_pehne`, `st_dev_test_pehne`, and the selected `best_alpha`. Runs also work offline.

## Reproducing the paper

Method names in the paper map to `model.type` as follows:

| Paper | `model.type` |
|---|---|
| TARNet | `TARNet` |
| NetDeconf | `GCN_DECONF` |
| NetEst | `NetEst` |
| TNet | `TargetedModel_DoubleBSpline` |
| GIN model | `GINModel` |
| SPNet | `SPNet` |
| IDE-Net | `IDENet` |
| HINet | `HINet` |
| HINet without GIN_T | `HINet_no_net_conf` |

Datasets map as follows:

| Paper | Config |
|---|---|
| BC | `data.dataset=BC` |
| Flickr | `data.dataset=Flickr` |
| BA Sim | `data.dataset=full_sim data.homophily=false` |
| Homophily Sim | `data.dataset=full_sim data.homophily=true` |
| Coauthor-CS | `data.dataset=CS` |

Exposure mappings map to `data.exposure_type`: `weight` (feature-weighted mean, the default), `sum`, `average` (proportion of treated neighbors), `entropy`, and `weight_squared`.

Everything else follows `config/default.yaml`. The parameters the experiments vary are:

- `data.beta_confounding` — the treatment-assignment strength β_XT
- `data.beta_neighbor_covariate_to_outcome=0` — removes the direct influence of neighbors' covariates on the outcome
- `model.gnn_layer` — `GIN`, `GAT`, `GraphSAGE`, or `GCN`
- `tuning.alpha_range` — `[0]` gives the unbalanced variant; any singleton list pins α to that value instead of running the selection heuristic. The full range must start at 0, since the tolerance is measured against its first entry.
- `tuning.p_alpha` — the tolerance of the α-selection heuristic


## Citation

```bibtex
@misc{caljon2026estimating,
  title         = {Estimating Treatment Effects in Networks under Unknown Exposure Mappings},
  author        = {Caljon, Daan and Van Belle, Jente and Verbeke, Wouter},
  year          = {2026},
  eprint        = {2510.21457v2},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

Copyright (C) 2026 Daan Caljon, Jente Van Belle, Wouter Verbeke.

This repository is released under the GNU General Public License v3.0; see `LICENSE`.

## Acknowledgements

Our code builds upon the code released by Jiang & Sun (2022), Guo, Li & Liu (2020), and
Chen et al. (2024). The IDE-Net implementation is adapted from the authors' original code
(Adhikari & Zheleva, 2025). The BC and Flickr datasets originate from the repository of
Guo, Li & Liu (2020) and are used in the same form as Jiang & Sun (2022). The Coauthor-CS
dataset is from the gnn-benchmark repository of Shchur et al. (2019), MIT license,
Copyright (c) 2018 Maximilain Mumme, Oleksandr Shchur, Technical University of Munich.

Jiang, S. & Sun, Y. (2022). Estimating causal effects on networked observational data via representation learning. In *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*, (pp. 852–861).

Guo, R., Li, J. & Liu, H. (2020). Learning individual causal effects from networked observational data. In *Proceedings of the 13th International Conference on Web Search and Data Mining*, (pp. 232–240).

Chen, W., Cai, R., Yang, Z., Qiao, J., Yan, Y., Li, Z. & Hao, Z. (2024). Doubly Robust Causal Effect Estimation under Networked Interference via Targeted Learning. *Proceedings of the 41st International Conference on Machine Learning*, in Proceedings of Machine Learning Research 235:6457–6485.

Adhikari, S. & Zheleva, E. (2025). Inferring individual direct causal effects under heterogeneous peer influence. *Machine Learning*, 114(4): 113.

Shchur, O., Mumme, M., Bojchevski, A. & Günnemann, S. (2019). Pitfalls of Graph Neural Network Evaluation. *arXiv:1811.05868*.
