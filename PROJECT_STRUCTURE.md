# Repository Architecture & Directory Structure Guide

This document provides a comprehensive structural guide to the CICIoT2023 Intrusion Detection System research repository.

---

## 1. High-Level Repository Architecture

```text
capstone/
├── .gitignore                         # Comprehensive Git exclusion rules protecting data & models
├── README.md                          # Executive project documentation & pipeline quickstart
├── PROJECT_STRUCTURE.md               # (This file) Repository architecture & directory guide
├── requirements.txt                   # Pinned runtime and experimental dependencies
│
├── src_v2/                            # [ACTIVE] Verified Stage 2–7 Leak-Free Research Pipeline
│   ├── config.py                      # Canonical paths, 39 features, threat tiers, class maps
│   ├── data_partitioning.py           # Stage 2 atomic hash-grouped dataset partitioning
│   ├── preprocessing.py               # LeakFreePreprocessor (fitted strictly on training splits)
│   ├── models.py                      # Model architectures (SGD, RF, LGBM, XGBoost, PyTorch MLP)
│   ├── run_stage3.py                  # Stage 3 training orchestrator for all 21 model checkpoints
│   ├── run_stage4_eval.py             # Stage 4 read-only dual-test evaluation runner
│   ├── ablation_strategy2.py          # Strategy 2 collision-purging partitioning script
│   ├── cross_capture.py               # Temporal cross-capture test partition builder
│   ├── evaluate.py                    # Multi-metric evaluation and threat-tier metric calculation
│   └── verify_partitions.py           # Mathematical verification script proving 0% feature overlap
│
├── src/                               # [LEGACY] Generation 1 Exploratory Pipeline (80/20 Random Splits)
│   ├── config.py                      # Legacy pipeline paths and configuration
│   ├── data_loader.py                 # Chunked dataset streaming loader
│   ├── preprocessing.py               # Legacy StandardScaler and train/test splitter
│   ├── models.py                      # Baseline models
│   ├── evaluate.py                    # Legacy metric evaluation
│   ├── feature_selection.py           # Top-20 feature pruning algorithm
│   └── train_*.py                     # Phase 1-3 baseline training scripts
│
├── data/                              # Datasets, Processed Samples, and Partition Maps
│   ├── raw/                           # 63 full-scale captures from CICIoT2023 (8.87 GB, gitignored)
│   ├── processed/                     # Curated working samples (sample_stratified.csv, 529k flows)
│   ├── partitions/                    # Strategy 1 leak-free partition maps & temporal test sets
│   ├── partitions/ablation_strategy2/ # Strategy 2 collision-purged partition maps
│   └── README.md                      # Data architecture and partition methodology documentation
│
├── models/                            # Serialized Model Weights & Preprocessors
│   ├── active/                        # 21 verified Stage 3 checkpoints + 2 preprocessors (gitignored)
│   ├── legacy/                        # Generation 1 baseline model checkpoints (gitignored)
│   └── README.md                      # Checkpoint manifest and loading specifications
│
├── reports/                           # Consolidated Master Research Reports (Stages 1–7)
│   ├── README.md                      # Master index linking and summarizing all 7 research stages
│   ├── STAGE_1_AUDIT_REPORT.md        # Stage 1: Forensic Dataset Audit & Collision Quantification
│   ├── STAGE_2_INVESTIGATION_REPORT.md# Stage 2: Feature-Hash Partitioning & Collision Purging
│   ├── STAGE_3_METHODOLOGY_REVIEW.md  # Stage 3: Preprocessing Parity & Class-Weighting Audit
│   ├── STAGE_3_CANARY_REPORT.md       # Stage 3: Canary Execution and Verification
│   ├── STAGE_4_EVALUATION_REPORT.md   # Stage 4: Master Test Evaluation (Dual-Partition)
│   ├── STAGE_5_ERROR_ANALYSIS.md      # Stage 5: Fine-Grained Error Analysis & Worst-10 Profiles
│   ├── STAGE_6_RESEARCH_SYNTHESIS.md  # Stage 6: Research Findings, Contributions & Synthesis
│   ├── STAGE_7_LITERATURE_REVIEW.md   # Stage 7: Systematic Literature Review (14 Studies)
│   ├── PROJECT_ORIENTATION_REPORT.md  # Pre-execution environment & codebase audit
│   └── PROJECT_STRUCTURE_AUDIT.md     # Pre-reorganization full repository audit
│
├── results_v2/                        # Active Experimental Outputs, Metrics, and Ledgers
│   ├── tables/                        # Master test CSV (stage4_test_results.csv) & centralized ledger
│   ├── metrics/                       # Per-class JSON metrics (21 models) & Stage 4 evaluation JSONs
│   ├── figures/                       # Destination directory for Generation 2 visualizations
│   └── README.md                      # Experimental data manifest
│
├── results/                           # [LEGACY] Generation 1 Evaluation Tables & Visualizations
│   ├── figures/                       # 30 PNG figures (confusion matrices, ROC curves, throughput charts)
│   └── tables/                        # 26 benchmark CSV/JSON tables from early exploration
│
├── research/                          # Academic Literature & Scientific Analysis
│   ├── *.pdf                          # 14 peer-reviewed research papers on CICIoT2023 & IoT IDS
│   ├── Dataset.pdf                    # Official CICIoT2023 dataset datasheet (Univ. of New Brunswick)
│   └── RESEARCH_PAPERS_CRITICAL_ANALYSIS.md # Detailed critical survey and comparative matrix
│
├── paper/                             # Academic Conference Paper Package
│   ├── main.tex                       # IEEE Transactions / Conference LaTeX manuscript
│   ├── paper.typ                      # Typst modern typesetting manuscript
│   ├── references.bib                 # Comprehensive BibTeX citation database
│   ├── research_paper.pdf             # Compiled camera-ready PDF deliverable
│   ├── figures/                       # Figures embedded within the academic manuscripts
│   └── README.md                      # Paper compilation and submission notes
│
├── snuc/                              # Shiv Nadar University Chennai Capstone Deliverables
│   ├── *.pptx / *.pdf                 # Reviews 1, 2, and 3 academic presentations & reports
│   └── *.tex                          # Project review LaTeX templates and humanized manuscripts
│
├── notebooks/                         # Exploratory Jupyter Notebooks (Phase 1–3, Benchmarking)
│   └── 01_*.ipynb - 08_*.ipynb        # Sequential exploratory research notebooks
│
├── learning/                          # Educational & Foundational Learning Modules
│   └── module_1/                      # Networking and CICIoT2023 fundamentals
│       ├── 01_networking_and_ciciot2023_fundamentals.ipynb
│       ├── MODULE_1_LEARNING_NOTES.md
│       └── scripts/build_notebook.py
│
├── experiments/                       # Standalone Ablations & Parameter Sweeps
│   ├── feature_decay_sweep.py         # Empirical latency vs. accuracy frontier
│   ├── feature_pruning.py             # Feature selection ablations
│   ├── hierarchical_cascade.py        # Multi-stage cascade pipeline
│   ├── hyperparameter_tuning.py       # Optuna / Grid search experiments
│   ├── natural_distribution_evaluation.py # Uncapped flood class testing
│   ├── uncapped_pure34_benchmark.py   # Raw traffic evaluation
│   └── weight_calibration.py          # Class weight balancing sweeps
│
├── scripts/                           # Utility Tools & SOC Dashboard
│   ├── demo_app.py                    # Streamlit Security Operations Center (SOC) Console
│   ├── benchmark_throughput.py        # Single-packet latency & throughput benchmarking tool
│   └── README.md                      # Script execution guide
│
├── docs/                              # Project Engineering Notes & Roadmaps
│   ├── INSIGHTS.md                    # Detailed technical ledger of findings & trade-offs
│   ├── PROGRESS.md                    # Historical capstone milestones and roadmap
│   └── README.md                      # Documentation overview
│
└── .venv/                             # Python 3.13 Virtual Environment (gitignored)
```

---

## 2. Directory Purpose & Functional Role Matrix

| Directory | Status | Primary Role & Contents |
|---|:---:|---|
| **`src_v2/`** | Active | Core engine for all Stage 2–7 research. Implements leak-free partitioning, strict training-only preprocessing, model instantiation, training runs, and read-only test evaluation. |
| **`src/`** | Legacy | Historical Generation 1 pipeline code used for early baselines and exploratory notebooks. Preserved for exact reproducibility of prior results. |
| **`data/`** | Active | Hierarchical dataset repository: `raw/` holds original CICIoT2023 CSVs; `processed/` holds the 529k stratified sample; `partitions/` holds atomic hash-grouped splits. |
| **`models/`** | Active | Central model store: `active/` holds the 21 audited Stage 3 checkpoints and 2 serialized preprocessors; `legacy/` preserves Generation 1 weights. |
| **`reports/`** | Active | Consolidated repository for all 7 research stage markdown reports, pre-execution orientation, and structural audits. |
| **`results_v2/`** | Active | Empirical outputs from the active pipeline: per-class JSON metrics, centralized experiment ledger, and master Stage 4 test results. |
| **`results/`** | Legacy | Generation 1 figures and tables from initial exploratory sweeps. |
| **`research/`** | Active | Curated academic library containing 14 peer-reviewed research papers and the CICIoT2023 dataset specification. |
| **`paper/`** | Active | Camera-ready academic conference paper package containing LaTeX, Typst, BibTeX, figures, and compiled PDF. |
| **`snuc/`** | University | Capstone project submission deliverables for Shiv Nadar University Chennai (Presentations, Project Reports, LaTeX templates). |
| **`notebooks/`** | Exploratory | Sequential Jupyter notebooks (01 to 08) chronicling the exploratory development phases. |
| **`learning/`** | Education | Educational notebooks and learning notes introducing IoT networking and dataset structure. |
| **`experiments/`**| Research | Standalone ablation scripts testing feature pruning, cascade architectures, and class weighting sweeps. |
| **`scripts/`** | Tools | Standalone execution utilities: Streamlit SOC dashboard (`demo_app.py`) and wire-speed throughput benchmarking (`benchmark_throughput.py`). |
| **`docs/`** | Reference | Engineering notebooks (`INSIGHTS.md`) and project roadmaps (`PROGRESS.md`). |

---

## 3. Data Flow Architecture

The following diagram illustrates how data flows through the active research pipeline:

```mermaid
flowchart TD
    subgraph Data Tier
        RAW["data/raw/ (Merged01-63.csv)"] --> SAMPLE["data/processed/sample_stratified.csv"]
        SAMPLE --> PART_S1["data/partitions/ (Strategy 1 Primary)"]
        SAMPLE --> PART_S2["data/partitions/ablation_strategy2/ (Strategy 2 Purged)"]
        RAW --> CC_TEST["data/partitions/cross_capture_test_clean.csv (Merged26-40)"]
    end

    subgraph Active Research Pipeline (src_v2)
        PART_S1 --> PRE_S1["LeakFreePreprocessor (Train N=350,821)"]
        PART_S2 --> PRE_S2["LeakFreePreprocessor (Train N=337,838)"]
        PRE_S1 --> TRAIN["src_v2/run_stage3.py"]
        PRE_S2 --> TRAIN
        MODELS["src_v2/models.py"] --> TRAIN
        TRAIN --> CKPTS["models/active/ (21 Checkpoints + 2 Preprocessors)"]
        CKPTS --> EVAL["src_v2/run_stage4_eval.py"]
        CC_TEST --> EVAL
    end

    subgraph Evaluation & Research Reports
        EVAL --> RES_TABLES["results_v2/tables/ (Ledger & Stage 4 CSV)"]
        EVAL --> RES_METRICS["results_v2/metrics/ (Per-Class & Stage 4 JSONs)"]
        RES_TABLES --> REP4["reports/STAGE_4_EVALUATION_REPORT.md"]
        RES_METRICS --> REP5["reports/STAGE_5_ERROR_ANALYSIS.md"]
        REP4 & REP5 --> REP6["reports/STAGE_6_RESEARCH_SYNTHESIS.md"]
        RESEARCH["research/ (14 Peer-Reviewed PDFs)"] --> REP7["reports/STAGE_7_LITERATURE_REVIEW.md"]
    end
```

---

## 4. Git Hygiene & Artifact Management Policy

- **Tracked in Git**: All Python source code (`src_v2/`, `src/`, `experiments/`, `scripts/`), research reports (`reports/`), documentation (`docs/`, `PROJECT_STRUCTURE.md`, `README.md`), research literature (`research/`), conference paper sources (`paper/`), exploratory notebooks (`notebooks/`, `learning/`), and structured summary result tables (`results/tables/*.csv`, `results_v2/tables/*.csv`).
- **Excluded via `.gitignore`**:
  - Raw and processed datasets (`data/raw/`, `data/processed/`, `data/partitions/`, `*.csv`).
  - Binary model checkpoints and serialized preprocessors (`models/`, `*.joblib`, `*.pt`, `*.pkl`).
  - University capstone submissions (`snuc/`).
  - Python virtual environment (`.venv/`), bytecode caches (`__pycache__/`, `*.pyc`).
