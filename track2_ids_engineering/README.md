# Track 2: Practical IDS Engineering (`track2_ids_engineering/`)

Active Track 2 engineering work focused on practical IoT intrusion detection, hierarchical vs. flat specialization, minority-class optimization, wire-speed throughput, and the interactive SOC console on CICIoT2023.

---

## Promoted Production Architecture (`E4 + Exp3 Hybrid Specialist`)

Track 2 unifies the best model from the E1–E4 study (**E4 Flat34 LightGBM**) with the best non-flood specialist from Minority Optimization (**Experiment 3: Flood-Free Stage A + Stage B Gate**):
1. **E4 Base Router (`hierarchical_ids/models/e4_lgbm_leaves63_sqrt.joblib`)** predicts 34-class or `OP29` flood-merged traffic. Flood predictions bypass directly (`0` true rare flows misrouted to Flood across Val/Test/Cross).
2. **Flood-Free Rare Specialist (`minority_optimization/models/exp3_specialist/`)** runs only on non-flood / benign flows (`StageA_Bin_unweighted` `tau_rare=0.30` -> `StageB_Rare8_sqrt_a050` `tau_cls=0.25`, overriding E4 when `max(P_E4) < 0.65`).
3. **Why Promoted**: Only approach improving **both** Internal Test (`OP29 Acc=85.76%`, `Macro-F1=0.6834`, `Min-F1=0.2288`) and Cross-Capture generalization (`OP29 Acc=78.16%`, `Macro-F1=0.6878`, `Min-F1=0.2930`) simultaneously without raising Benign false positives (`21.45%` Test vs `21.37%` E4).

---

## Clean Folder Map

| Path | Role & Key Files |
|---|---|
| [`src/`](src/) | **Main Track 2 Production & Baselines**: **`hybrid_pipeline.py`** (unified `HybridE4SpecialistIDS` API combining E4 + Exp3 for `OP29`/`ORIG34`), multi-granularity training (`train_*.py`), feature selection (`feature_selection.py`), and sampling (`create_sample.py`). |
| [`scripts/`](scripts/) | **Main Track 2 Entry Points**: **`run_hybrid_ids.py`** (CLI verification & CSV batch inference for the promoted Hybrid IDS), **`demo_app.py`** (Streamlit SOC Console featuring Promoted Hybrid, E4, and Fast E2), and `benchmark_throughput.py` (10 Gbps profiler). |
| [`hierarchical_ids/`](hierarchical_ids/) | **E1–E4 Hierarchical vs. Flat Study**: Modular evaluation (`src/`), Val-tuned trainer (`experiments/train_and_tune_val.py`), frozen evaluator (`experiments/run_final_evaluation.py`), CSV/JSON results (`results/`), and [`COMPARISON_REPORT.md`](hierarchical_ids/docs/COMPARISON_REPORT.md). |
| [`minority_optimization/`](minority_optimization/) | **Controlled Minority Optimization Study (Exp 1–3)**: Pre-training SMOTE realism audit (`src/smote_audit.py`), train-only SMOTE (`run_exp1_smote.py`), power class weighting (`run_exp2_class_weights.py`), flood-free specialist (`run_exp3_specialist.py`), Phase 4 evaluator (`run_phase4_evaluation.py`), plot generator (`generate_plots.py`), CSVs/configs (`configs/`, `results/`), and [`FINAL_REPORT.md`](minority_optimization/FINAL_REPORT.md). |
| [`experiments/`](experiments/) | Early Track 2 ablations (`weight_calibration.py`, `hyperparameter_tuning.py`, `feature_pruning.py`, `feature_decay_sweep.py`, `hierarchical_cascade.py`). |
| [`results/`](results/) & [`docs/`](docs/) | Baseline CSV/JSON tables (`results/tables/`), PNG plots (`results/figures/`), and technical ledger ([`docs/INSIGHTS.md`](docs/INSIGHTS.md)). |

---

## Key Commands (Run From Repo Root)

```powershell
# 1. Run Promoted Main Track 2 Hybrid Pipeline (OP29 default or ORIG34)
.\.venv\Scripts\python.exe -m track2_ids_engineering.scripts.run_hybrid_ids --taxonomy OP29 --split test
.\.venv\Scripts\python.exe -m track2_ids_engineering.scripts.run_hybrid_ids --taxonomy OP29 --split cross
.\.venv\Scripts\python.exe -m track2_ids_engineering.scripts.run_hybrid_ids --input path\to\flows.csv --output preds.csv

# 2. Launch Interactive SOC Defense Console (Promoted Hybrid + E4 + Fast E2 Switcher)
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py

# 3. Regenerate Models, Plots, & Evaluation Tables (if cloning fresh)
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase all
.\.venv\Scripts\python.exe -u track2_ids_engineering\minority_optimization\scripts\run_exp3_specialist.py
.\.venv\Scripts\python.exe -u track2_ids_engineering\minority_optimization\scripts\generate_plots.py

# 4. Benchmark 10 Gbps Host Batch Throughput
.\.venv\Scripts\python.exe track2_ids_engineering\scripts\benchmark_throughput.py --samples 50000 --warmup 1000
```
