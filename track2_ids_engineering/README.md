# Track 2: Practical IDS Engineering (`track2_ids_engineering/`)

Active Track 2 engineering work focused on practical IoT intrusion detection performance, feature/latency optimization, operational dashboards, and evaluating whether hierarchical Flood vs. Non-Flood specialization beats improved flat models on CICIoT2023.

---

## Contents

| Path | Role & Key Files |
|---|---|
| [`src/`](src/) | Multi-granularity baselines across Binary (`train_binary.py`), 8-Class (`train_category_8class.py`), 34-Class (`train_attack_34class.py`), Top-20 feature selection (`feature_selection.py`), sampling (`create_sample.py`), and synthesis (`capstone_synthesis.py`). |
| [`models/`](models/) | Saved `.joblib`/`.pt` checkpoints for `binary/`, `category_8class/`, `attack_34class/`, and `reduced_features/` *(gitignored binaries)*. |
| [`results/`](results/) | Baseline tables (`tables/*.csv`, `*.json`) and 30 plots (`figures/*.png`). |
| [`experiments/`](experiments/) | Standalone ablations: `weight_calibration.py`, `hyperparameter_tuning.py`, `feature_pruning.py`, `feature_decay_sweep.py`, and early 2-stage prototype (`hierarchical_cascade.py`). |
| [`scripts/`](scripts/) | `demo_app.py` (Streamlit 3-tier SOC console) and `benchmark_throughput.py` (10 Gbps host throughput & DMA burst latency profiler). |
| [`docs/INSIGHTS.md`](docs/INSIGHTS.md) | Complete chronological engineering ledger (Steps 0–9), schemas, trade-offs, hardware limits, and IP-less `DDoS/DoS` findings. |
| [`hierarchical_ids/`](hierarchical_ids/docs/COMPARISON_REPORT.md) | **Completed E1–E4 Study**: clean modular implementation (`src/{data_utils,architectures,metrics_eval}.py`), Train/Val trainer (`experiments/train_and_tune_val.py`), frozen read-only evaluator (`experiments/run_final_evaluation.py`), 18 `.joblib` models (gitignored), CSV/JSON/`.npy` results (`results/`), and [`docs/COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md). |

---

## E1–E4 Hierarchical vs. Improved Flat Summary

Full charts live in [`../notebooks/09_hierarchical_ids_evaluation.ipynb`](../notebooks/09_hierarchical_ids_evaluation.ipynb) and detailed error/per-class tables in [`hierarchical_ids/docs/COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md):

| Architecture | Test Acc (N=75,105) | Test Macro-F1 | Test Minority F1 | Cross Acc (N=57,280) | Cross Macro-F1 | Execs / Flow | P50 Latency (Test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`E1_Flat34_LGBM_Balanced`** (Track 1) | 72.65% | 0.6145 | 0.1624 | 69.64% | **0.6498** | 1.00 | **3.43 ms** |
| **`E2_Cascade_Default_T050`** | 73.59% | 0.6185 | 0.2115 | 68.61% | 0.6213 | 2.63 | 7.88 ms |
| **`E2_Cascade_ValTuned`** (`tau_att=0.15`) | 75.04% | **0.6361** | **0.2264** | 70.50% | 0.6453 | 2.82 | 53.30 ms* |
| **`E4_LGBM_Leaves63_Sqrt`** (Flat L63) | **75.37%** | 0.6352 | 0.2196 | **70.72%** | 0.6441 | 1.00 | 4.15 ms |

*\*E2 Tuned uses Track 1 `binary_random_forest.joblib` (`n_jobs=-1`), creating single-row thread dispatch overhead (`8,042.5 fps` in batches).*  
**Operational Note**: `hierarchical_ids/src/data_utils.py` slices Train/Val/Test/Cross in memory directly from [`../track1_dataset_evaluation/data/partitions/`](../track1_dataset_evaluation/data/partitions/) and [`../data/processed/sample_stratified.csv`](../data/processed/sample_stratified.csv) without duplicating CSV datasets on disk.

---

## Key Commands (Run From Repo Root)

```powershell
# Re-train/tune E2/E3/E4 checkpoints (Train + Val only) & evaluate read-only on Val/Test/Cross
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase all
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_final_evaluation

# Launch SOC demo or 10 Gbps benchmark
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py
.\.venv\Scripts\python.exe track2_ids_engineering/scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```
