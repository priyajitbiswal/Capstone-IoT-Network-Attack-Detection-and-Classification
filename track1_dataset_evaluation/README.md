# Track 1: Dataset Integrity Evaluation & Academic Manuscript (`track1_dataset_evaluation/`)

This track contains all research artifacts dedicated to **evaluating the CICIoT2023 benchmark dataset itself**—quantifying identical-feature redundancy, conflicting label collisions, class-imbalance recall collapse, temporal cross-capture fragility, and natural-vs-balanced distribution discrepancies—alongside the completed academic paper.

---

## Directory Contents

| Path | Role & Contents |
|---|---|
| [`src/`](src/) | Verified Stage 2–4 leak-free research pipeline: atomic SHA-256 group-hash partitioning (`data_partitioning.py`), collision purging (`ablation_strategy2.py`), cross-capture extraction (`cross_capture.py`), partition overlap verifier (`verify_partitions.py`), strict training-only preprocessing (`preprocessing.py`), model training (`run_stage3.py`), and dual-test evaluation (`run_stage4_eval.py`). |
| [`data/partitions/`](data/partitions/) | Strategy 1 primary leak-free partitions (`N = 501,098`), Strategy 2 collision-purged ablation partitions (`N = 482,682`), and clean temporal cross-capture test partition (`N = 57,280`). *(Partition CSVs are gitignored; metadata JSONs are tracked.)* |
| [`models/`](models/README.md) | 21 verified Stage 3 checkpoints (5 models × 3 granularities + 4 Strategy 2 ablations + 2 unweighted ablations) and 2 fitted `LeakFreePreprocessor` objects. *(Gitignored binaries; see [`models/README.md`](models/README.md).)* |
| [`results/`](results/README.md) | Stage 4 master evaluation table (`tables/stage4_test_results.csv`), centralized experiment ledger (`tables/centralized_experiment_ledger.csv`), distribution reconciliation tables (`exp5_*`, `exp7_*`), per-class JSON reports, Stage 4 JSONs, and DNN loss logs. |
| [`reports/`](reports/README.md) | Complete chronological archive of research audits and stage reports (`STAGE_1_AUDIT_REPORT.md` through `STAGE_9_FINAL_PUBLICATION_VERIFICATION.md`), plus repository orientation/structure audits. |
| [`paper/`](paper/README.md) | Camera-ready academic manuscript package: [`paper/research_paper.pdf`](paper/research_paper.pdf), IEEEtran LaTeX (`main.tex`), Typst (`paper.typ`), BibTeX (`references.bib`), and 300 DPI publication figures (`fig1`–`fig6`). |
| [`experiments/`](experiments/README.md) | Dataset-distribution experiments evaluating raw uncapped traffic vs. stratified capping (`natural_distribution_evaluation.py`, `uncapped_pure34_benchmark.py`). |
| [`scripts/`](scripts/README.md) | Publication visualization utility [`generate_paper_figures.py`](scripts/generate_paper_figures.py) generating Figures 1–6 from `results/tables/`. |

---

## Core Findings Summary (Completed Phase)

1. **Forensic Data Integrity Audit**: Identified 11.29% feature redundancy (59,780 flows) and 8,740 conflicting-label collision hashes across 529,562 stratified flows; simulated 80/20 random splits allow 81.39% of test collision instances to match training coordinates.
2. **Zero-Leakage Partitioning Protocol**: Enforced `Train feature hashes ∩ Test feature hashes = ∅` across internal and cross-capture splits via atomic SHA-256 group-hash partitioning and training-only preprocessor fitting.
3. **Class Imbalance & Minority Collapse**: Unweighted empirical loss boosts headline 34-class accuracy by +1.80% while triggering 79%–82% relative recall collapse on stealthy minority attacks (SQL Injection, XSS) and increasing Benign false negatives by +125.31%.
4. **Temporal Non-Stationarity**: Evaluating models on independent capture sessions (`Merged26`–`Merged40`) reveals measurable macro-F1 degradation across diagnostic granularities.

---

## Cross-References

- Shared raw & stratified datasets: [`../data/`](../data/)
- Related notebook still at root: [`../notebooks/07_uncapped_high_accuracy_benchmark.ipynb`](../notebooks/07_uncapped_high_accuracy_benchmark.ipynb)
- Overall two-track guide: [`../TRACKS_OVERVIEW.md`](../TRACKS_OVERVIEW.md)
