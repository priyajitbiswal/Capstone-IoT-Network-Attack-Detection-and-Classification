# IoT Network Intrusion Detection: Empirical Benchmarking & Methodological Auditing (CICIoT2023)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/LightGBM-4.1.0-green.svg" alt="LightGBM" />
  <img src="https://img.shields.io/badge/XGBoost-1.7.6-orange.svg" alt="XGBoost" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4.0-blue.svg?logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/CICIoT2023-Benchmark-red.svg" alt="CICIoT2023 Benchmark" />
</p>

---

## Executive Summary

This repository presents an end-to-end Machine Learning, Deep Learning, and Methodological Auditing investigation for IoT Network Intrusion Detection evaluated on the **CICIoT2023** benchmark (Canadian Institute for Cybersecurity, 105 IoT devices, 33 attack types, ~45 million raw flows).

The codebase is organized into two distinct research generations:
1. **Generation 1 (`src/`, `models/legacy/`, `results/`, `notebooks/`)**: Original exploratory pipeline implementing random 80/20 train/test splits, single-model and multi-model baselines (SGD, Random Forest, LightGBM, XGBoost, PyTorch MLP), feature selection (39 &rarr; 20 features), wire-speed host throughput benchmarking, and Streamlit demo.
2. **Generation 2 (`src_v2/`, `models/active/`, `results_v2/`, `reports/`)**: Rigorous, leak-free re-architecture executing Stages 1 through 7:
   - **Stage 1 (Data Audit)**: Audited 529,562 flows, quantifying exact duplicates, identical-feature vectors, and 8,740 conflicting label collision hashes.
   - **Stage 2 (Partitioning)**: Implemented atomic feature-hash grouped partitioning enforcing 0% vector overlap across splits (Strategy 1 Primary, Strategy 2 Purged, and Cross-Capture Clean).
   - **Stage 3 (Model Suite)**: Trained and audited 21 baseline model checkpoints across 5 phases with stateful `LeakFreePreprocessor` serialization.
   - **Stage 4 (Evaluation)**: Evaluated all 21 models strictly read-only across Internal Test ($N=75,105$) and same-testbed Temporal Cross-Capture Test ($N=57,280$).
   - **Stage 5 (Error Analysis)**: Comprehensive confusion, false-negative, and ablation error analysis.
   - **Stage 6 (Research Synthesis)**: Rigorous synthesis resolving scientific claims, trade-offs, and calibrated findings.
   - **Stage 7 (Literature Review)**: Deep-dive review of 14 peer-reviewed research papers and capstone technical documentation.

---

## Master Research Trajectory & Reports

All formal research stage reports and structural audits are consolidated in the [`reports/`](reports/) directory:

| Research Stage | Report Link | Primary Scope & Focus |
|:---:|---|---|
| **Stage 1** | [`reports/STAGE_1_AUDIT_REPORT.md`](reports/STAGE_1_AUDIT_REPORT.md) | Forensic dataset audit; identified 8,740 conflicting feature-label collisions. |
| **Stage 2** | [`reports/STAGE_2_INVESTIGATION_REPORT.md`](reports/STAGE_2_INVESTIGATION_REPORT.md) | Zero-leakage feature-hash grouped partitioning & temporal cross-capture design. |
| **Stage 3** | [`reports/STAGE_3_METHODOLOGY_REVIEW.md`](reports/STAGE_3_METHODOLOGY_REVIEW.md) | Preprocessing parity audit, class-weight calibration, and DNN budget verification. |
| **Stage 3** | [`reports/STAGE_3_CANARY_REPORT.md`](reports/STAGE_3_CANARY_REPORT.md) | Canary execution and runtime verification of initial baseline models. |
| **Stage 4** | [`reports/STAGE_4_EVALUATION_REPORT.md`](reports/STAGE_4_EVALUATION_REPORT.md) | Read-only dual-test evaluation across all 21 Stage 3 model checkpoints. |
| **Stage 5** | [`reports/STAGE_5_ERROR_ANALYSIS.md`](reports/STAGE_5_ERROR_ANALYSIS.md) | Fine-grained error analysis, worst-10 attack profiles, and weighting ablations. |
| **Stage 6** | [`reports/STAGE_6_RESEARCH_SYNTHESIS.md`](reports/STAGE_6_RESEARCH_SYNTHESIS.md) | Comprehensive contribution synthesis, calibrated claims, and limitations. |
| **Stage 7** | [`reports/STAGE_7_LITERATURE_REVIEW.md`](reports/STAGE_7_LITERATURE_REVIEW.md) | Systematic literature review of 14 peer-reviewed CICIoT2023 studies. |
| **Pre-Audit**| [`reports/PROJECT_ORIENTATION_REPORT.md`](reports/PROJECT_ORIENTATION_REPORT.md) | Pre-execution codebase and schema orientation report. |
| **Architecture**| [`reports/PROJECT_STRUCTURE_AUDIT.md`](reports/PROJECT_STRUCTURE_AUDIT.md) | Read-only structural audit of all project files and dependencies. |

---

## Repository Directory Overview

```text
capstone/
├── src_v2/             # [ACTIVE] Verified Stage 2-7 leak-free research pipeline
├── src/                # [LEGACY] Generation 1 exploratory baseline code
├── data/               # Datasets, processed samples, and leak-free partition maps
│   ├── raw/            # 63 raw captures from CICIoT2023 (gitignored)
│   ├── processed/      # Curated working sample (sample_stratified.csv)
│   └── partitions/     # Strategy 1 & Strategy 2 partition assignment maps
├── models/             # Serialized model weights & preprocessors (gitignored)
│   ├── active/         # 21 verified Stage 3 checkpoints + 2 preprocessors
│   └── legacy/         # Generation 1 baseline model checkpoints
├── reports/            # Consolidated Stage 1-7 research reports & audits
├── results_v2/         # Active experimental outputs (ledger, stage4 CSV, JSONs)
├── results/            # Legacy Generation 1 figures and tables
├── research/           # 14 peer-reviewed research papers & dataset datasheet
├── paper/              # Academic conference paper package (LaTeX, Typst, PDF)
├── snuc/               # Academic capstone deliverables (Shiv Nadar University)
├── notebooks/          # Exploratory Jupyter research notebooks (01 to 08)
├── learning/           # Educational modules on IoT networking fundamentals
├── experiments/        # Standalone ablation scripts & parameter sweeps
├── scripts/            # Standalone demonstration tools & throughput benchmarks
├── docs/               # Engineering notes (INSIGHTS.md) & roadmap (PROGRESS.md)
├── PROJECT_STRUCTURE.md# Detailed guide to repository architecture & directory roles
└── requirements.txt    # Project dependencies
```

For full details on every directory, see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

---

## Quickstart & Pipeline Execution

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Data Partitions (Zero Feature Overlap)
```powershell
python -m src_v2.verify_partitions
```

### 3. Run Stage 4 Final Test Evaluation
Evaluates all 21 pre-trained Stage 3 checkpoints on the Internal Test set ($N=75,105$) and same-testbed Temporal Cross-Capture Test set ($N=57,280$) in strictly read-only mode:
```powershell
python -m src_v2.run_stage4_eval
```
Metrics are recorded to `results_v2/tables/stage4_test_results.csv` and `results_v2/metrics/stage4/*.json`.

### 4. Launch Security Operations Console (Streamlit SOC Dashboard)
```powershell
streamlit run scripts/demo_app.py
```

### 5. Benchmark Wire-Speed Latency & Throughput
```powershell
python scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```

---

## Academic Paper & University Submissions

- **Conference Paper**: Available in [`paper/`](paper/), including IEEE LaTeX sources ([`paper/main.tex`](paper/main.tex)), Typst typesetting sources ([`paper/paper.typ`](paper/paper.typ)), and compiled camera-ready PDF ([`paper/research_paper.pdf`](paper/research_paper.pdf)).
- **Capstone Deliverables**: Stored in [`snuc/`](snuc/) for Shiv Nadar University Chennai, including Reviews 1, 2, and 3 presentations and project reports.
- **Literature Repository**: Stored in [`research/`](research/) containing 14 peer-reviewed research studies and technical documentation.
