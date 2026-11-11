# Empirical Evaluation of Data Leakage, Class Imbalance, and Temporal Fragility in IoT Network Intrusion Detection on CICIoT2023

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/LightGBM-4.1.0-green.svg" alt="LightGBM" />
  <img src="https://img.shields.io/badge/XGBoost-1.7.6-orange.svg" alt="XGBoost" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4.0-blue.svg?logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Typst-0.12.0-239DAD.svg" alt="Typst" />
  <img src="https://img.shields.io/badge/CICIoT2023-Benchmark-red.svg" alt="CICIoT2023 Benchmark" />
</p>

---

## 1. Executive Summary

This repository hosts the complete empirical research codebase, evaluation artifacts, and academic manuscript package for our SEM-7 capstone project:
**"Empirical Evaluation of Data Leakage, Class Imbalance, and Temporal Fragility in IoT Network Intrusion Detection on CICIoT2023"**.

Recent literature on IoT Network Intrusion Detection Systems (NIDS) evaluating the **CICIoT2023** benchmark (Canadian Institute for Cybersecurity, 105 IoT devices, 33 attack profiles, 46+ million raw flows) frequently reports classification accuracies ranging from **98.5% to 99.9%** under standard evaluation protocols. 

This research project conducts an empirical data integrity audit and zero-leakage benchmark demonstrating that reported headline metrics can be distorted by unpartitioned identical-feature data leakage, conflicting label collisions, severe class imbalance, and an absence of cross-session temporal validation. 

By introducing an open-source **Atomic SHA-256 Group-Hash Partitioning Protocol** that enforces strictly zero identical-feature overlap (**Train feature hashes ∩ Test feature hashes = ∅**) alongside strict preprocessor isolation, we establish verified, leak-free diagnostic baselines across five model families and three diagnostic granularities.

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE SCIENTIFIC TRAJECTORY                                      |
+----------------------------------------------------------------------------------------------------+
|  1. BENCHMARK INTEGRITY VULNERABILITY                                                              |
|  - 11.29% feature redundancy (59,780 flows) in full stratified sample (N = 529,562).               |
|  - 8,740 conflicting-label collision hashes spanning 28,608 rows (5.40% of full sample).           |
|  - Naive random splits cause 81.39% of test collision instances to leak from train.                |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|  2. ZERO-LEAKAGE BENCHMARK METHODOLOGY                                                             |
|  - Atomic SHA-256 group-hash partitioning guarantees strictly 0 identical-feature overlap.         |
|  - Preprocessing parameters (mu, sigma^2, limits) fitted strictly on training data.                |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|  3. LEAK-FREE EVALUATION BASELINES                                                                 |
|  - Random Forest achieves 89.61% binary accuracy (Macro-F1: 0.7690).                                |
|  - LightGBM achieves 78.93% category accuracy (Macro-F1: 0.6894) & 72.65% fine-grained (0.6145).   |
|  - Purging collisions (Strategy 2) yields only a marginal +0.0022 clean validation Macro-F1 gain.  |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|  4. OPERATIONAL SECURITY & ROBUSTNESS TRADE-OFFS                                                   |
|  - Unweighted training raises headline accuracy (+1.8%) but causes 79% to 82% recall reduction     |
|    on stealthy web attacks (SQLi, XSS) and surges false negatives to Benign by +125.3%.            |
|  - Temporal cross-capture evaluation across distinct sessions reveals session non-stationarity.    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Central Research Question and Sub-RQs

### Central Research Question
> *When identical-feature data leakage and class imbalance artifacts are rigorously controlled through atomic hash-grouped partitioning and cost-sensitive optimization, what is the realistic baseline diagnostic capability, minority attack recall, and temporal stability of standard machine learning architectures on the CICIoT2023 benchmark?*

### Sub-Research Questions
- **RQ1 (Integrity)**: To what extent do identical feature vectors, exact duplicates, and conflicting label collisions exist in CICIoT2023, and what fraction of test instances overlap under random splitting?
- **RQ2 (Zero-Leakage Baseline)**: What are the measured diagnostic boundaries of linear, tree, and neural baselines across Binary (2-class), Category (8-class), and Fine-Grained (34-class) tasks when feature leakage is strictly eliminated?
- **RQ3 (Ambiguity Mechanics)**: How does the presence of conflicting label collisions affect tree and linear models, and does collision purging improve clean generalization?
- **RQ4 (Operational Trade-Offs)**: How does standard unweighted empirical loss minimization trade minority attack detection for overall headline accuracy?
- **RQ5 (Temporal Fragility)**: How robust are trained models when evaluated on an independent capture session recorded across subsequent hours on the physical IoT testbed?

---

## 3. Four Consolidated Research Contributions

1. **Forensic Data Integrity & Leakage Audit**: Granular census of 529,562 stratified CICIoT2023 flow records: identified 11.29% feature redundancy (59,780 flows) and 8,740 conflicting-label collision hashes spanning 28,608 rows (5.40% of full sample). Proved empirically that simulated 80/20 random stratified splitting allows 81.39% of test collision instances (and 10.14% of un-deduplicated test flows) to match identical training coordinates.
2. **Reproducible Zero-Leakage Group-Hash Partitioning Standard**: Open-source atomic SHA-256 group-hash partitioning protocol guaranteeing strictly zero identical-feature overlap (**Train feature hashes ∩ Test feature hashes = ∅**) alongside strict preprocessor isolation. Demonstrated consistent generalization (|Δ_gen| ≤ 0.0079) across all 21 models.
3. **Empirical Analysis of Ambiguity Filtering Mechanics**: Evaluated classifier sensitivity to conflicting label noise. Decomposed validation on 72,390 common-clean instances demonstrates that purging collision vectors yields a marginal +0.0022 Macro-F1 improvement (+0.22%), while classifiers achieve ~41.73% accuracy on isolated ambiguous vectors due to the non-injective nature of the stripped feature space.
4. **Quantitative Trade-Offs: Minority Recall Collapse & Temporal Non-Stationarity**: Quantified the operational trade-offs of class weighting: unweighted training raises headline accuracy by +1.80% but triggers a 79% to 82% relative recall reduction on minority web attacks (SQLi, XSS, PingSweep) and surges false negatives to Benign by +125.31%. Furthermore, temporal evaluation across distinct physical capture sessions reveals measurable degradation (binary Macro-F1 drops from 0.7637 to 0.7034).

---

## 4. Master Research Trajectory & Reports (Track 1)

All formal stage reports, consistency audits, verification documents, and blueprints are organized chronologically in [`track1_dataset_evaluation/reports/`](track1_dataset_evaluation/reports/) (see [`TRACKS_OVERVIEW.md`](TRACKS_OVERVIEW.md) for the two-track project layout):

| Stage | Report Document | Scope & Key Finding / Milestone |
|:---:|---|---|
| **Stage 1** | [`STAGE_1_AUDIT_REPORT.md`](track1_dataset_evaluation/reports/STAGE_1_AUDIT_REPORT.md) | Forensic dataset audit; identified 8,740 conflicting feature-label collision hashes. |
| **Stage 2** | [`STAGE_2_INVESTIGATION_REPORT.md`](track1_dataset_evaluation/reports/STAGE_2_INVESTIGATION_REPORT.md) | Zero-leakage atomic hash partitioning protocol & temporal cross-capture design. |
| **Stage 3** | [`STAGE_3_COMPLETE_REPORT.md`](track1_dataset_evaluation/reports/STAGE_3_COMPLETE_REPORT.md) | Training of 21 model checkpoints, preprocessor isolation, and decomposed validation. |
| **Stage 4** | [`STAGE_4_EVALUATION_REPORT.md`](track1_dataset_evaluation/reports/STAGE_4_EVALUATION_REPORT.md) | Read-only dual-test evaluation across Internal Test (N = 75,105) and Cross-Capture (N = 57,280). |
| **Stage 5** | [`STAGE_5_ERROR_ANALYSIS.md`](track1_dataset_evaluation/reports/STAGE_5_ERROR_ANALYSIS.md) | Fine-grained error analysis, confusion topology, and class weighting ablations. |
| **Stage 6** | [`STAGE_6_RESEARCH_SYNTHESIS.md`](track1_dataset_evaluation/reports/STAGE_6_RESEARCH_SYNTHESIS.md) | Calibrated scientific claims synthesis, trade-offs, and boundary definitions. |
| **Stage 7** | [`STAGE_7_LITERATURE_REVIEW.md`](track1_dataset_evaluation/reports/STAGE_7_LITERATURE_REVIEW.md) | Systematic review of 14 peer-reviewed IoT NIDS studies (2021–2026). |
| **Stage 8A** | [`STAGE_8A_RESEARCH_CONSOLIDATION.md`](track1_dataset_evaluation/reports/STAGE_8A_RESEARCH_CONSOLIDATION.md) | Consolidation across Stages 1–7, research problem formulation, and contribution mapping. |
| **Stage 8A.1**| [`STAGE_8A1_CONSISTENCY_AUDIT.md`](track1_dataset_evaluation/reports/STAGE_8A1_CONSISTENCY_AUDIT.md) | Independent consistency audit; uncovered 18 numerical and conceptual discrepancies. |
| **Stage 8A.2**| [`STAGE_8A2_AUDIT_VERIFICATION.md`](track1_dataset_evaluation/reports/STAGE_8A2_AUDIT_VERIFICATION.md) | Mathematical verification & certification of ground truth constants across the repository. |
| **Stage 8B** | [`STAGE_8B_MANUSCRIPT_PLAN.md`](track1_dataset_evaluation/reports/STAGE_8B_MANUSCRIPT_PLAN.md) | Comprehensive manuscript blueprint, 9-section architecture, and 3-tier claim ledger. |
| **Stage 8C** | [`STAGE_8C_DRAFTING_NOTES.md`](track1_dataset_evaluation/reports/STAGE_8C_DRAFTING_NOTES.md) | Working drafting inventory, visuals generation, and manuscript assembly notes. |
| **Stage 8D** | [`STAGE_8D_FINAL_REVIEW.md`](track1_dataset_evaluation/reports/STAGE_8D_FINAL_REVIEW.md) | Final manuscript audit, computational profiling reconciliation, and packaging. |

---

## 5. Master Benchmark Results Across 21 Models

All metrics are evaluated on the strictly isolated, zero-leakage **Internal Test Set (N = 75,105)**:

| Diagnostic Task | Model Architecture | Partition Strategy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test MCC | Test ROC-AUC | Test Macro-AP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Binary (2-Class)** | SGD Logistic Regression | Strategy 1 | 0.6803 | 81.73% | 0.6761 | 0.4849 | 0.9442 | 0.9951 |
| Binary | PyTorch Tabular MLP | Strategy 1 | 0.7143 | 85.11% | 0.7125 | 0.5315 | 0.9646 | 0.9969 |
| Binary | **Random Forest** | Strategy 1 | **0.7733** | **89.61%** | **0.7690** | **0.6057** | 0.9738 | 0.9977 |
| Binary | LightGBM | Strategy 1 | 0.7666 | 89.15% | 0.7637 | 0.6007 | **0.9751** | **0.9978** |
| Binary | XGBoost | Strategy 1 | 0.7501 | 87.79% | 0.7462 | 0.5785 | 0.9731 | 0.9977 |
| Binary | LightGBM (Ablation S2) | Strategy 2 | 0.7686 | 89.39% | 0.7668 | 0.6045 | 0.9751 | 0.9978 |
| **Category (8-Class)** | SGD Logistic Regression | Strategy 1 | 0.6006 | 74.04% | 0.6036 | 0.6723 | 0.9521 | 0.6390 |
| Category | PyTorch Tabular MLP | Strategy 1 | 0.6104 | 73.41% | 0.6130 | 0.6834 | 0.9663 | 0.7094 |
| Category | Random Forest | Strategy 1 | 0.6802 | 78.22% | 0.6805 | 0.7399 | 0.9765 | 0.7726 |
| Category | **LightGBM** | Strategy 1 | **0.6865** | **78.93%** | **0.6894** | **0.7468** | **0.9789** | **0.7875** |
| Category | XGBoost | Strategy 1 | 0.6615 | 76.08% | 0.6619 | 0.7180 | 0.9748 | 0.7599 |
| Category | LightGBM (Ablation S2) | Strategy 2 | 0.6925 | 79.12% | 0.6903 | 0.7482 | 0.9789 | 0.7878 |
| **Fine-Grained (34-Class)** | SGD Logistic Regression | Strategy 1 | 0.5260 | 64.29% | 0.5272 | 0.6308 | 0.9621 | 0.5357 |
| Fine-Grained | PyTorch Tabular MLP | Strategy 1 | 0.5211 | 62.42% | 0.5216 | 0.6137 | 0.9710 | 0.5828 |
| Fine-Grained | Random Forest | Strategy 1 | 0.5956 | 70.58% | 0.5966 | 0.6972 | 0.9781 | 0.6431 |
| Fine-Grained | **LightGBM** | Strategy 1 | **0.6142** | **72.65%** | **0.6145** | **0.7170** | **0.9811** | **0.6619** |
| Fine-Grained | XGBoost | Strategy 1 | 0.5984 | 71.04% | 0.6001 | 0.7012 | 0.9790 | 0.6475 |
| Fine-Grained | LightGBM (Ablation S2) | Strategy 2 | 0.6218 | 72.69% | 0.6140 | 0.7175 | 0.9811 | 0.6623 |
| Fine-Grained | XGBoost (Ablation S2) | Strategy 2 | 0.6034 | 70.97% | 0.5993 | 0.7005 | 0.9791 | 0.6475 |
| Fine-Grained | LightGBM (Unweighted) | Strategy 1 | 0.6239 | 74.45% | 0.6220 | 0.7346 | 0.9712 | 0.6431 |
| Fine-Grained | XGBoost (Unweighted) | Strategy 1 | 0.6139 | 74.03% | 0.6148 | 0.7306 | 0.9806 | 0.6550 |

---

## 6. Host-Side Computational Profiling

Measured on host CPU hardware (**Intel Core i5-1135G7 @ 2.40GHz**, single-thread inference):

| Task Granularity | Model Architecture | P50 Latency (µs) | P95 Latency (µs) | Batch Throughput (flows/s) | Peak RSS (MB) | On-Disk Model Size |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Binary (2-Class)** | SGD Logistic Regression | **362.1** | 518.2 | **8,325,000** | 481.7 | <0.01 MB (1.3 KB) |
| Binary | PyTorch Tabular MLP | 247.1 | 557.3 | 163,500 | 650.9 | 0.06 MB (63.3 KB) |
| Binary | **LightGBM** | 1,983.3 | 2,227.0 | 938,400 | 494.7 | 0.36 MB (360.9 KB) |
| Binary | XGBoost | 1,575.3 | 2,130.2 | 1,424,300 | 497.5 | 0.26 MB (256.6 KB) |
| Binary | Random Forest | 70,883.1 | 123,496.3 | 62,400 | 498.9 | 12.35 MB |
| **Category (8-Class)**| SGD Logistic Regression | **143.2** | 247.5 | **8,007,700** | 484.7 | <0.01 MB (2.5 KB) |
| Category | PyTorch Tabular MLP | 613.8 | 1,372.0 | 123,600 | 638.6 | 0.06 MB (64.9 KB) |
| Category | LightGBM | 1,409.8 | 1,852.1 | 92,400 | 484.8 | 2.81 MB |
| Category | XGBoost | 2,244.0 | 3,235.0 | 264,800 | 483.6 | 2.11 MB |
| Category | Random Forest | 22,467.5 | 41,428.5 | 330,300 | 526.9 | 36.47 MB |
| **Fine-Grained (34-Class)**| SGD Logistic Regression | **189.7** | 495.9 | **2,901,500** | 482.8 | <0.01 MB (7.1 KB) |
| Fine-Grained | PyTorch Tabular MLP | 942.3 | 1,405.9 | 74,700 | 638.9 | 0.07 MB (71.8 KB) |
| Fine-Grained | **LightGBM** | 3,504.4 | 4,882.8 | 10,990 | 493.8 | 11.88 MB |
| Fine-Grained | XGBoost | 3,732.5 | 6,092.7 | 47,560 | 501.8 | 8.67 MB |
| Fine-Grained | Random Forest | 31,296.1 | 46,911.1 | 157,490 | 572.6 | 95.65 MB |

> [!NOTE]
> **Measurement Distinctions & Operational Scope**: P50/P95 latencies measure sequential unbatched single-flow evaluation (including Python/framework dispatch overhead for isolated flows). Batch throughput is evaluated over pre-allocated in-memory NumPy arrays, which amortizes dispatch and allocation overhead. Batch throughput does not represent single-flow latency (which cannot be obtained by inverting batch throughput) nor does it represent physical network interface line rate (which involves packet capture, link-layer parsing, flow reassembly, and feature extraction upstream of the ML classifier). All profiling reflects host-side Intel Core i5-1135G7 CPU execution.

---

## 7. Repository Directory Overview

```text
capstone/
├── TRACKS_OVERVIEW.md                 # Two-track research organization guide
├── track1_dataset_evaluation/         # TRACK 1: Dataset integrity audit, leak-free benchmark & paper
│   ├── src/                           # Stage 2–4 zero-leakage partitioning, training & evaluation
│   ├── data/partitions/               # Strategy 1 & Strategy 2 partition maps + cross-capture test sets
│   ├── models/                        # 21 verified Stage 3 checkpoints + 2 preprocessors (gitignored)
│   ├── results/                       # Stage 4 CSVs, centralized ledger, per-class & stage4 JSONs
│   ├── reports/                       # Stage 1–9 formal research reports & audits
│   ├── paper/                         # Academic paper package (LaTeX, Typst, PDF, references.bib, figures)
│   ├── experiments/                   # Natural vs. uncapped dataset distribution scripts
│   └── scripts/                       # 300 DPI publication figure generator
├── track2_ids_engineering/            # TRACK 2: Practical IDS architectures, baselines & hierarchical IDS
│   ├── src/                           # 2-class, 8-class, 34-class baselines & feature selection
│   ├── models/                        # Baseline binary, 8-class, 34-class & top-20 checkpoints (gitignored)
│   ├── results/                       # Baseline confusion matrices, ROC/PR curves & benchmark tables
│   ├── experiments/                   # Weight calibration, tuning, pruning & 2-stage cascade ablations
│   ├── scripts/                       # Streamlit SOC console (demo_app.py) & 10G throughput benchmark
│   ├── docs/                          # Engineering notes (INSIGHTS.md) & roadmap (PROGRESS.md)
│   ├── data/                          # Spec for future hierarchical IDS Datasets A, B, C1, C2
│   └── hierarchical_ids/              # Proposed 3-Stage Flood vs. Non-Flood IDS architecture spec
├── data/                              # Shared raw CICIoT2023 captures & curated stratified sample
│   ├── raw/                           # 63 full raw captures from CICIoT2023 (gitignored)
│   └── processed/                     # Curated working sample (sample_stratified.csv, 529k flows)
├── notebooks/                         # Exploratory Jupyter research notebooks (01 to 08)
├── research/                          # 14 peer-reviewed research papers & dataset datasheet
├── learning/                          # Educational modules on IoT networking fundamentals
├── snuc/                              # Academic capstone deliverables (Shiv Nadar University Chennai)
├── PROJECT_STRUCTURE.md               # Detailed guide to repository architecture & directory roles
└── requirements.txt                   # Project dependencies
```

---

## 8. Quickstart & Reproduction Instructions

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Data Partitions (Zero Feature Overlap — Track 1)
```powershell
.\.venv\Scripts\python.exe -m track1_dataset_evaluation.src.verify_partitions
```

### 3. Re-run Stage 4 Read-Only Test Evaluation (Track 1)
Evaluates all 21 pre-trained Stage 3 checkpoints on the Internal Test set (N = 75,105) and Cross-Capture Test set (N = 57,280) in strictly read-only mode:
```powershell
.\.venv\Scripts\python.exe -m track1_dataset_evaluation.src.run_stage4_eval
```

### 4. Regenerate Publication Figures (300 DPI — Track 1)
```powershell
.\.venv\Scripts\python.exe track1_dataset_evaluation/scripts/generate_paper_figures.py
```

### 5. Re-compile Academic Manuscript (Typst — Track 1)
```powershell
.\.venv\Scripts\python.exe -c "import typst; typst.compile('track1_dataset_evaluation/paper/paper.typ', output='track1_dataset_evaluation/paper/research_paper.pdf')"
```

### 6. Launch Security Operations Console (Streamlit Dashboard — Track 2)
```powershell
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py
```

---

## 9. Academic Manuscript Package (Track 1)

- **Title**: *Empirical Evaluation of Data Leakage, Class Imbalance, and Temporal Fragility in IoT Network Intrusion Detection on CICIoT2023*
- **Authors**: Priyajit Biswal, Rohit K Manoj, Rishab Rajeev (Dept. of Computer Science & Engineering, Shiv Nadar University Chennai)
- **Faculty In Charge**: Dr. Vegesna S M Srinivasavarma
- **LaTeX Source**: [`track1_dataset_evaluation/paper/main.tex`](track1_dataset_evaluation/paper/main.tex) (IEEEtran / Elsevier layout, 100% verified syntax and citations)
- **Typst Source**: [`track1_dataset_evaluation/paper/paper.typ`](track1_dataset_evaluation/paper/paper.typ) (Modern Typst markup)
- **Compiled PDF**: [`track1_dataset_evaluation/paper/research_paper.pdf`](track1_dataset_evaluation/paper/research_paper.pdf) (publication-ready PDF, Stage 9 verified)
- **Bibliography**: [`track1_dataset_evaluation/paper/references.bib`](track1_dataset_evaluation/paper/references.bib) (23 BibTeX entries including all 14 audited peer-reviewed literature studies)
- **Visuals**: [`track1_dataset_evaluation/paper/figures/`](track1_dataset_evaluation/paper/figures/) (Figures 1 to 6)

---

## 10. Explicit Scientific Limitations

1. **Dataset Scope**: Benchmarks are evaluated on the CICIoT2023 physical testbed dataset. Cross-dataset transferability to external datasets (such as ToN_IoT or BoT-IoT) was not evaluated.
2. **Packet Payload Inaccessibility**: Tabular summary features exclude packet payloads, inherently bounding the detection of application-layer web exploits like SQL Injection.
3. **Neural Compute Budget**: The PyTorch MLP baseline was trained under a 5-epoch CPU budget. Extended training or specialized tabular architectures (e.g., TabNet) may yield different neural baselines.
4. **Host Hardware Profiling**: Compute metrics reflect single-thread host CPU execution (Intel Core i5-1135G7); physical microcontroller evaluation (ARM Cortex-M) remains future work.
5. **Literature Scope**: Methodological comparisons are bounded strictly to the 14 surveyed peer-reviewed studies.
