# CICIoT2023 IoT Intrusion Detection Research Capstone

Leak-free benchmarking, hierarchical vs. flat multi-class IDS engineering, and wire-speed throughput evaluation on the **CICIoT2023** dataset (105 IoT devices, 33 attack classes + `BENIGN`, 39 tabular packet/flow features).

---

## Repository Tracks & Layout

| Path | Track / Role | Key Contents |
|---|---|---|
| [`track1_dataset_evaluation/`](track1_dataset_evaluation/) | **Track 1 — Dataset Integrity & Paper (Frozen)** | Zero-leakage SHA-256 hash partitioning (`src/`), 21 baseline/ablation models (`models/`, gitignored), Stage 1–9 audit reports (`reports/`), and IEEE/Typst manuscript ([`paper/research_paper.pdf`](track1_dataset_evaluation/paper/research_paper.pdf)). |
| [`track2_ids_engineering/`](track2_ids_engineering/) | **Track 2 — Practical IDS Engineering (Active)** | 2/8/34-class & Top-20 baselines (`src/`), Streamlit SOC console & 10G throughput benchmark (`scripts/`), technical ledger ([`docs/INSIGHTS.md`](track2_ids_engineering/docs/INSIGHTS.md)), and E1–E4 Hierarchical vs. Improved Flat IDS study ([`hierarchical_ids/`](track2_ids_engineering/hierarchical_ids/docs/COMPARISON_REPORT.md)). |
| [`notebooks/`](notebooks/) | **Interactive Notebooks (`01`–`09`)** | `01`–`06`, `08`, `09` cover Track 2 (`09`: E1 vs. E2 vs. E4 evaluation); `07` covers Track 1 uncapped raw distribution evaluation. |
| [`data/`](data/) | **Shared Dataset (Gitignored CSVs)** | Raw CICIoT2023 captures (`raw/Merged01–63.csv`) and stratified working sample (`processed/sample_stratified.csv`: 529,562 rows × 40 cols, SHA-256 `39370152e9b2cf95e64e1df650d3bc677241f811cd38983894502fd1f8baa227`). |
| [`research/`](research/) & [`learning/`](learning/) | **Literature & Notes** | 14 surveyed papers + critical analysis (`research/`) and educational notes (`learning/module_1/`). |

---

## Verified Dataset Partitions (Zero Feature-Vector Overlap)

All Track 1 benchmarks and Track 2 E1–E4 models evaluate strictly on Track 1 Strategy 1 hash-grouped partitions (`verify_partitions.py`: **0** hash and **0** exact feature-vector overlaps across all 6 split pairs):

| Split | Origin | Total Flows | Role / Key Composition Note |
|---|---|---:|---|
| **Train** | `Merged01–25` (`sample_stratified.csv` deduped) | 350,821 | **70.01%** of internal stratum (`27,855` Benign, `189,288` Flood, `133,678` Non-Flood, `2.60%` Minority). |
| **Validation** | `Merged01–25` (`sample_stratified.csv` deduped) | 75,172 | **15.00%** of internal stratum — threshold, calibration, and model selection only (`2.45%` Minority). |
| **Internal Test** | `Merged01–25` (`sample_stratified.csv` deduped) | 75,105 | **14.99%** held-out read-only test (`2.58%` Minority). |
| **Cross-Capture Clean** | `Merged26–40` (unseen later capture files) | 57,280 | Independent same-testbed holdout (`<=2,000`/attack, `<=5,000` Benign; `14.64%` Minority = **5.67x** denser than Test). |

---

## Key Empirical Findings (Track 1 + Track 2)

1. **Random Splits Leak Identical Flow Vectors (Track 1)**: Raw `sample_stratified.csv` contains `5.38%` (`28,464`) exact duplicate rows and `8,740` unique feature vectors with conflicting labels (`28,608` rows). Standard random 80/20 splits leak `81.39%` of conflicting vectors into test, inflating 34-class accuracy.
2. **Capacity & `sqrt_balanced` Weights Match the 3-Stage Cascade (Track 2 — [`COMPARISON_REPORT.md`](track2_ids_engineering/hierarchical_ids/docs/COMPARISON_REPORT.md))**:
   - Track 1 baseline **E1 (`E1_Flat34_LGBM_Balanced`)**: `72.65%` Test Acc / `0.6145` Macro-F1 / `0.1624` Minority F1 (`0.3360` Recall) / `69.64%` Cross Acc / `0.6498` Cross Macro-F1 (`1.0` exec, `3.43 ms` P50).
   - Tuned 3-Stage Cascade **E2 Tuned** (`tau_att=0.15` Stage 1 -> 7-cat Stage 2 -> 16-Flood / 17-NonFlood specialists): `75.04%` Test Acc / **`0.6361`** Macro-F1 / **`0.2264`** Minority F1 (`70.50%` Cross Acc / `0.6453` Cross Macro-F1; `2.82` execs, `53.30 ms` P50 due to Stage 1 RF `n_jobs=-1`).
   - Upgraded Flat **E4 (`E4_LGBM_Leaves63_Sqrt`)**: **`75.37%` Test Acc** / `0.6352` Macro-F1 / **`0.7440` MCC** / **`70.72%` Cross Acc** (`1.0` exec, **`4.15 ms` P50**) — proving tree capacity + `sqrt_balanced` weights achieve nearly identical Macro-F1 to hierarchy at ~13x faster single-flow latency.
   - Stage 2 Flood-vs-Non-Flood routing is already **99.98% Recall** (`50 / 75,105` misroutes in E2 vs. `46` in flat E4); naive Stage 1 threshold `0.50` drops `11.57%` (`7,990`) Test attacks as Benign; naive parallel fusion (`E3_NaiveMax`) collapses to `56.11%` Test Acc without a learned family gate (`76.16%` dual-confident OOD firing).
3. **IP-Stripped Schema Creates the DDoS/DoS Error Ceiling**: Direct header inspection confirms **0 of 40 raw columns** include IPs, ports, timestamps, or distinct attacker-IP counts. **`7,840` of E4's `18,501` Test errors (`42.4%`)** are swaps between 4 protocol-identical `DDoS`/`DoS`/`SynonymousIP` twin pairs; merging those mitigation twins raises E4 Test Accuracy from **`75.37% -> 85.81%`**.

---

## Reproduction & Quickstart (Windows PowerShell)

```powershell
# 0. Activate environment (Python 3.13, dependencies in requirements.txt)
.\.venv\Scripts\Activate.ps1

# 1. Regenerate sample_stratified.csv from data/raw/Merged01-25.csv (optional if sample exists)
.\.venv\Scripts\python.exe -m track2_ids_engineering.src.create_sample

# 2. Verify zero overlap across Track 1 Strategy 1 & Cross-Capture splits
.\.venv\Scripts\python.exe -u track1_dataset_evaluation\src\verify_partitions.py

# 3. Run Track 2 E1-E4 training/tuning (Train+Val only) & frozen read-only evaluation
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase all
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_final_evaluation

# 4. Run 10 Gbps throughput benchmark or launch interactive SOC console (Track 2)
.\.venv\Scripts\python.exe track2_ids_engineering\scripts\benchmark_throughput.py --samples 50000 --warmup 1000
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py

# 5. Re-run Track 1 Stage 4 evaluation & Typst PDF compilation (read-only on frozen models)
.\.venv\Scripts\python.exe -m track1_dataset_evaluation.src.run_stage4_eval
.\.venv\Scripts\python.exe -c "import typst; typst.compile('track1_dataset_evaluation/paper/paper.typ', output='track1_dataset_evaluation/paper/research_paper.pdf')"
```

**Git Policy**: Source code, docs, notebooks, Track 1/Track 2 summary tables & JSONs, and the compiled PDF are tracked. Large CSVs (`data/`, `track1_dataset_evaluation/data/partitions/*.csv`), binary models (`*.joblib`, `*.pt`), `.venv/`, and `snuc/` are excluded via [`.gitignore`](.gitignore).
