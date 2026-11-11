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
| [`scripts/`](scripts/) | `demo_app.py` (Streamlit SOC console powered by **E4 Primary + Fast E2 switcher + Track 1 `preprocessor_strategy1.joblib`**) and `benchmark_throughput.py` (10 Gbps host throughput & DMA burst latency profiler). |
| [`docs/INSIGHTS.md`](docs/INSIGHTS.md) | Complete chronological engineering ledger (Steps 0–9), schemas, trade-offs, hardware limits, and IP-less `DDoS/DoS` findings. |
| [`hierarchical_ids/`](hierarchical_ids/docs/COMPARISON_REPORT.md) | **Finalized E1–E4 Study**: modular implementation (`src/{data_utils,architectures,metrics_eval}.py`), Train/Val trainer (`experiments/train_and_tune_val.py`), frozen evaluator (`experiments/run_final_evaluation.py`), Phase 1 presentation evaluator (`experiments/run_presentation_updates.py`), 18 `.joblib` models (gitignored), CSV/JSON/`.npy`/PNG outputs (`results/`), and [`docs/COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md). |

---

## Final E1–E4 Comparison & Operational Summary (`Internal Test N = 75,105`, `Cross N = 57,280`)

Full charts live in [`hierarchical_ids/results/figures/`](hierarchical_ids/results/figures/) and [`../notebooks/09_hierarchical_ids_evaluation.ipynb`](../notebooks/09_hierarchical_ids_evaluation.ipynb); full stage/per-class tables live in [`hierarchical_ids/docs/COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md):

| Architecture | Role / Design | Test Acc | Test Macro-F1 | Test MCC | Test Minority F1 (Rec) | Cross Acc | Cross Macro-F1 | P50 Latency (Test) | Execs / Flow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`E1_Flat34_LGBM_Balanced`** | Track 1 Baseline | 72.65% | 0.6145 | 0.7170 | 0.1624 (**0.3360**) | 69.64% | **0.6498** | **3.43 ms** | 1.00 |
| **`E2_Cascade_Default_T050`** | S1 LGBM (`T=0.50`) + S2/3 | 73.59% | 0.6185 | 0.7289 | 0.2115 (0.1519) | 68.61% | 0.6213 | 7.88 ms | 2.63 |
| **`E2_Cascade_ValTuned`** | Legacy S1 RF (`T=0.15`) | 75.04% | **0.6361** | 0.7405 | **0.2264** (0.2094) | 70.50% | 0.6453 | 53.30 ms* | 2.82 |
| **`E2_Cascade_ValTuned_LGBM`** | **Comparative Fast Hierarchy (S1 LGBM)** | 75.13% | 0.6357 | 0.7416 | 0.2263 (0.2094) | 70.60% | 0.6444 | **8.44 ms** | 2.81 |
| **`E3_Parallel_MetaGate`** *(vs Naive `0.4389`)* | Unsuccessful Parallel Fusion | 73.78% | 0.6239 | 0.7275 | 0.2253 (0.2094) | 69.41% | 0.6327 | 52.40 ms* | 2.82 |
| **`E4_LGBM_Leaves63_Sqrt`** | **Primary Recommended Deployment** | **75.37%** | 0.6352 | **0.7440** | 0.2196 (0.2063) | **70.72%** | 0.6441 | **4.15 ms** | **1.00** |

*\*Legacy E2 & E3 invoked Track 1 `binary_random_forest.joblib` (`n_jobs=-1`), creating single-row thread overhead. Swapping Stage 1 to existing `binary_lightgbm.joblib` cuts E2 P50 latency **6.31×** (`53.30 ms -> 8.44 ms`, P95 `10.23 ms`, batch `7,527.6 fps`) while slightly improving Test Acc & MCC. `E4_LGBM_Leaves63_Sqrt` remains the recommended production IDS due to top Test Accuracy/MCC/AUC (`0.9823`), single-pass simplicity (`1.0` exec), and `4.15 ms` latency.*

### Operational Merged-Flood Taxonomy (`OP29` vs. Standard 34-Class Task)
- **Why Separate**: Raw CICIoT2023 exports **0 / 40 IP/host-cardinality columns**, causing **`7,840` of E4's `18,501` Test errors (`42.4%`)** inside four protocol-identical mitigation twin groups (`SYN_TRIO` = `DDOS-SYN_FLOOD`+`DOS-SYN_FLOOD`+`DDOS-SYNONYMOUSIP_FLOOD`: `4,118` swaps / `99.62%` retention; `TCP`: `1,817`; `UDP`: `1,536`; `HTTP`: `369`).
- **E4 on Exact Internal Test (`N = 75,105`)**: Original 34-Class (`75.37%` Acc / `0.6352` Macro-F1 / `0.7440` MCC) vs. Alternative 29-Class Operational Mitigation (`85.81%` Acc / `0.6808` Macro-F1 / `0.8504` MCC / `0.9893` Flood F1; Non-Flood `0.4789` and Minority `0.2196` F1 unchanged). **`OP29` evaluates a coarser operational firewall taxonomy—NOT an improvement on the 34-class benchmark.**

---

## Key Commands (Run From Repo Root)

```powershell
# Reproduce Fast E2 (LGBM S1), Merged-Flood Operational Audit, & Presentation Figures (read-only on Track 1)
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_presentation_updates

# Launch interactive SOC Defense Console (E4 Primary + Fast E2 Switcher + Track 1 Strategy 1 Preprocessor)
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py

# Benchmark 10 Gbps host batch throughput
.\.venv\Scripts\python.exe track2_ids_engineering/scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```
