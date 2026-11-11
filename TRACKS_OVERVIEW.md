# Capstone Research Project Structure: Two-Track Organization

This repository is organized into **two distinct, complementary research tracks** built on the **CICIoT2023** benchmark dataset, alongside shared datasets, literature, educational material, and uncategorized exploratory notebooks.

> **Organization Notice**: The repository has been structured into two clean, self-contained tracks (`track1_dataset_evaluation/` and `track2_ids_engineering/`) with updated relative paths, Python package imports, and track-aligned directory names (`src/`, `results/`, `models/`). All historical Stage 1–9 audit reports and university deliverables (`snuc/`) preserve their original historical records.

---

## High-Level Two-Track Map

```text
capstone/
├── TRACKS_OVERVIEW.md                     # (This file) Master two-track directory guide
├── README.md / PROJECT_STRUCTURE.md       # Repository overview & directory architecture
├── REPRODUCIBILITY.md / requirements.txt  # End-to-end reproduction & dependency specs
│
├── track1_dataset_evaluation/             # TRACK 1: Dataset Integrity Audit & Research Paper
│   ├── src/                               # Atomic SHA-256 group-hash partitioning, training & Stage 4 eval
│   ├── data/partitions/                   # Strategy 1 & 2 hash partitions + temporal cross-capture sets
│   ├── models/                            # 21 verified Stage 3 model checkpoints + 2 preprocessors
│   ├── results/                           # Stage 4 test tables, per-class JSONs, centralized ledger
│   ├── reports/                           # Formal Stage 1–9 audit & synthesis reports
│   ├── paper/                             # IEEE LaTeX, Typst, references.bib, fig1–6, research_paper.pdf
│   ├── experiments/                       # Raw vs. balanced dataset distribution evaluations
│   └── scripts/                           # Paper figure generator (generate_paper_figures.py)
│
├── track2_ids_engineering/                # TRACK 2: Practical IDS Architecture & Hierarchical System
│   ├── src/                               # Initial 2-class, 8-class, 34-class & feature selection pipeline
│   ├── models/                            # Trained binary, 8-class, 34-class, and top-20 edge checkpoints
│   ├── results/                           # Confusion matrices, ROC/PR curves, benchmark & throughput tables
│   ├── experiments/                       # Weight calibration, tuning, feature pruning & 2-stage cascade
│   ├── scripts/                           # Archon Streamlit SOC console (demo_app.py) & 10G throughput suite
│   ├── docs/                              # Engineering insights ledger (INSIGHTS.md) & roadmap (PROGRESS.md)
│   ├── data/                              # Track 2 data spec (future Datasets A, B, C1, C2)
│   └── hierarchical_ids/                  # Implemented & evaluated 3-Stage Flood vs. Non-Flood & E4 IDS study
│
├── data/                                  # SHARED DATASET REPOSITORY (raw CSVs + curated stratified sample)
├── notebooks/                             # UNASSIGNED AT ROOT (01–08; kept at root pending track review)
├── research/                              # SHARED LITERATURE (14 papers, Dataset.pdf, critical survey)
├── learning/                              # SHARED EDUCATIONAL MODULES (Module 1 networking & CICIoT2023)
└── snuc/                                  # UNIVERSITY DELIVERABLES (untouched, external to git repo)
```

---

## Track Descriptions & Responsibilities

### Track 1 — Dataset Integrity, Leakage Audit & Academic Paper (`track1_dataset_evaluation/`)
- **Objective**: Evaluate the **CICIoT2023** dataset itself—quantifying identical-feature redundancy, conflicting label collisions, class-weighting trade-offs, and temporal cross-capture fragility—and publish the empirical findings.
- **Status**: **Completed (Stages 1–9 verified)**, culminating in [`track1_dataset_evaluation/paper/research_paper.pdf`](track1_dataset_evaluation/paper/research_paper.pdf).
- **Key Artifacts**:
  - Zero-leakage atomic group-hash partitioning (`src/`, `data/partitions/`).
  - 21 audited baseline and ablation checkpoints (`models/`).
  - Master evaluation tables & JSONs (`results/`).
  - Chronological research & verification reports (`reports/STAGE_1_...` through `STAGE_9_...`).
  - Camera-ready LaTeX/Typst manuscript and 300 DPI figures (`paper/`).
  - Uncapped natural-distribution dataset evaluations (`experiments/natural_distribution_evaluation.py`, `experiments/uncapped_pure34_benchmark.py`).

### Track 2 — Practical Intrusion Detection System Engineering (`track2_ids_engineering/`)
- **Objective**: Design, optimize, and demonstrate practical, operationally useful Intrusion Detection System architectures (multi-granularity detection, feature reduction, wire-speed throughput, SOC console, and specialized hierarchical routing).
- **Status**: **Active Presentation & Next-Phase System Design Track**.
- **Completed Foundation**:
  - Initial 2-class (`Benign` vs `Attack`), 8-class category, and 34-class fine-grained classification pipelines (`src/`, `models/`, `results/`).
  - Edge feature reduction (39 → 20 features) and 10 Gbps host-side throughput profiling (`scripts/benchmark_throughput.py`).
  - Interactive Archon Edge IDS Streamlit SOC Console (`scripts/demo_app.py`).
  - Standalone IDS optimization experiments (`experiments/hierarchical_cascade.py`, `feature_pruning.py`, `feature_decay_sweep.py`, `hyperparameter_tuning.py`, `weight_calibration.py`).
- **Completed & Evaluated Hierarchical IDS Study**:
  - **Three-Stage Hierarchical Cascade (Flood vs. Non-Flood Specialists) vs. Improved Flat 34-Class (E1–E4)**, implemented and evaluated on Track 1 Strategy 1 splits (`hierarchical_ids/README.md`, `hierarchical_ids/docs/COMPARISON_REPORT.md`, `notebooks/09_hierarchical_ids_evaluation.ipynb`).

---

## Shared & Root-Preserved Directories

| Directory | Role | Notes |
|---|---|---|
| [`data/`](data/) | Shared CICIoT2023 raw captures (`raw/Merged01–63.csv`) and curated stratified working sample (`processed/sample_stratified.csv`) | Read-only source data shared across both tracks; Track 1 hash partitions reside in [`track1_dataset_evaluation/data/partitions/`](track1_dataset_evaluation/data/partitions/). |
| [`notebooks/`](notebooks/) | Exploratory Jupyter notebooks `01` through `09` | Intentionally kept at repository root for now. Conceptually, `01`–`06`, `08`, and `09` cover Track 2 IDS engineering (`09`: hierarchical vs. improved flat evaluation), while `07` covers Track 1 uncapped dataset evaluation. |
| [`research/`](research/) | 14 peer-reviewed papers, `Dataset.pdf`, and `RESEARCH_PAPERS_CRITICAL_ANALYSIS.md` | Shared academic literature base supporting both tracks. |
| [`learning/`](learning/) | Educational Module 1 (`01_networking_and_ciciot2023_fundamentals.ipynb`, notes, builder script) | Shared foundational training materials. |
| [`snuc/`](snuc/) | Shiv Nadar University Chennai review presentations, reports, and LaTeX templates | Untouched and gitignored university deliverables. |

---

## Execution Convention

All Python modules across both tracks support direct package execution from the repository root:
- Track 1 example: `.\.venv\Scripts\python.exe -m track1_dataset_evaluation.src.verify_partitions`
- Track 2 example: `.\.venv\Scripts\python.exe -m track2_ids_engineering.src.train_binary`
- Legacy checkpoints serialized under `src_v2.*` load seamlessly via compatibility aliases in [`track1_dataset_evaluation/src/__init__.py`](track1_dataset_evaluation/src/__init__.py).
