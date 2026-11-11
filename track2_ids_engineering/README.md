# Track 2: Practical Intrusion Detection System Engineering (`track2_ids_engineering/`)

This track encompasses all engineering, modeling, optimization, deployment, and system-design research focused on **building a more accurate, efficient, and operationally useful IoT Intrusion Detection System** on CICIoT2023—including the initial multi-granularity baselines, edge latency optimization, wire-speed profiling, interactive SOC console, and the proposed **Three-Stage Hierarchical IDS (Flood vs. Non-Flood Specialists)**.

---

## Directory Contents

| Path | Role & Contents |
|---|---|
| [`src/`](src/) | Initial baseline pipeline across three classification granularities: Binary (`train_binary.py`), 8-Class Category (`train_category_8class.py`), 34-Class Fine-Grained (`train_attack_34class.py`), Top-20 feature selection (`feature_selection.py`), master synthesis (`capstone_synthesis.py`), data streaming (`data_loader.py`, `create_sample.py`), preprocessing, models, and evaluation utilities. |
| [`models/`](models/README.md) | Trained baseline model checkpoints and scalers for `binary/`, `category_8class/`, `attack_34class/`, and `reduced_features/` (Top-20 edge subset). *(Gitignored binaries; see [`models/README.md`](models/README.md).)* |
| [`results/`](results/) | Evaluation tables (`tables/`: 24 benchmark/ablation/throughput CSV & JSON files) and visualizations (`figures/`: 30 confusion matrices, ROC/PR curves, feature importance/decay plots, throughput chart, and multi-tier defense architecture diagram). |
| [`experiments/`](experiments/) | IDS model & architecture optimization ablations: `weight_calibration.py`, `hyperparameter_tuning.py`, `feature_pruning.py`, `feature_decay_sweep.py`, and the earlier 2-stage 8-category cascade prototype (`hierarchical_cascade.py`). |
| [`scripts/`](scripts/) | Operational IDS tools: Archon Edge IDS Streamlit SOC Console ([`demo_app.py`](scripts/demo_app.py)) and 10 Gbps wire-speed throughput/burst latency benchmark ([`benchmark_throughput.py`](scripts/benchmark_throughput.py)). |
| [`docs/`](docs/) | Original engineering roadmap ([`PROGRESS.md`](docs/PROGRESS.md)) and technical insights ledger ([`INSIGHTS.md`](docs/INSIGHTS.md)). |
| [`data/`](data/) | Data specification for Track 2 derived training splits (future Datasets A, B, C1, C2); raw/stratified flows are read from [`../data/`](../data/). |
| [`hierarchical_ids/`](hierarchical_ids/) | **Implemented & Evaluated** comparison of 3-Stage Flood vs. Non-Flood Cascade (E2), Parallel Fusion (E3), and Improved Flat 34-Class (E4) against Track 1 E1 (`docs/COMPARISON_REPORT.md`). |

---

## Research Progression in Track 2

1. **Multi-Granularity Baselines (Completed)**: Benchmarked SGD Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch Tabular DNN across 2-class, 8-class, and 34-class tasks (`src/`, `results/`).
2. **Feature Reduction & Edge Optimization (Completed)**: Ranked consensus feature importance across RF, XGBoost, and LightGBM; evaluated Top-20 edge subset (`feature_selection.py`) and feature decay sweeps (`feature_decay_sweep.py`, `feature_pruning.py`).
3. **Wire-Speed & Multi-Tier Deployment Prototyping (Completed)**: Benchmarked Tier 1–3 models (`benchmark_throughput.py`) and built an interactive SOC inspection console (`demo_app.py`).
4. **Early 2-Stage Cascade Prototype (Exploratory)**: Tested an 8-category macro classifier routing into per-category specialists (`experiments/hierarchical_cascade.py`).
5. **3-Stage Hierarchical vs. Improved Flat IDS Study (Completed)**: Implemented and evaluated under Track 1 leakage-free splits in [`hierarchical_ids/README.md`](hierarchical_ids/README.md), [`hierarchical_ids/docs/COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md), and [`../notebooks/09_hierarchical_ids_evaluation.ipynb`](../notebooks/09_hierarchical_ids_evaluation.ipynb).

---

## Cross-References

- Shared raw & stratified datasets: [`../data/`](../data/)
- Track 1 dataset audit & academic paper: [`../track1_dataset_evaluation/`](../track1_dataset_evaluation/)
- Exploratory notebooks retained at root: [`../notebooks/`](../notebooks/) (`01`–`06` and `08` align with Track 2 workflows)
- Overall two-track guide: [`../TRACKS_OVERVIEW.md`](../TRACKS_OVERVIEW.md)
