# Track 1: CICIoT2023 Dataset Integrity Audit & Academic Paper (`track1_dataset_evaluation/`)

Frozen Track 1 research suite evaluating **CICIoT2023** integrity—identical-feature redundancy, conflicting label collisions, unweighted minority-recall collapse, and same-testbed cross-capture fragility—alongside the compiled IEEE-format paper ([`paper/research_paper.pdf`](paper/research_paper.pdf)).

---

## Directory Structure & Important Notes

| Path | Contents & Operational Notes |
|---|---|
| [`src/`](src/) | Leak-free pipeline: group-hash partitioner (`data_partitioning.py`), Strategy 2 collision purger (`ablation_strategy2.py`), cross-capture extractor (`cross_capture.py`), overlap verifier (`verify_partitions.py`), fit-on-train preprocessor (`preprocessing.py`), model factory (`models.py`), trainer (`run_stage3.py`), and dual-test evaluator (`run_stage4_eval.py`). *(Includes `src_v2.*` compatibility aliases in `src/__init__.py` for serialized checkpoints.)* |
| [`data/partitions/`](data/partitions/) | Strategy 1 primary splits (`N=501,098`: Train `350,821`, Val `75,172`, Test `75,105`), Strategy 2 purged ablation (`N=482,682`), and clean cross-capture holdout (`cross_capture_test_clean.csv`, `N=57,280`). *(Partition CSVs gitignored; metadata JSONs tracked.)* |
| [`models/`](models/) | `preprocessor_strategy1.joblib`, `preprocessor_strategy2.joblib`, and 21 audited Stage 3 checkpoints (`binary_*`, `category_*`, `fine_grained_*`, `ablation_s2_*`, `ablation_unweighted_*`) *(gitignored binaries)*. Preprocessors fit finite column bounds + `StandardScaler` strictly on training rows (`scale_features=True`). |
| [`experiments/`](experiments/) | Raw vs. balanced dataset scripts: `natural_distribution_evaluation.py` (outputs `exp5_distribution_reconciliation.csv`) and `uncapped_pure34_benchmark.py` (outputs `exp7_uncapped_pure34.csv`). Reads `../data/processed/sample_stratified.csv` & `../data/raw/`. |
| [`results/`](results/) | `tables/` (`centralized_experiment_ledger.csv`, `stage4_test_results.csv`, `exp5_*.csv`, `exp7_*.csv`), `metrics/per_class/` (21 JSONs), `metrics/stage4/` (21 JSONs), DNN loss JSONs, and `figures/`. |
| [`scripts/`](scripts/) | `generate_paper_figures.py` — deterministically regenerates Figures 1–6 at 300 DPI from `results/tables/` into `paper/figures/`. |
| [`paper/`](paper/) | Compiled 11-page PDF (`research_paper.pdf`), Typst source (`paper.typ`, primary local build), IEEEtran LaTeX source (`main.tex`), BibTeX (`references.bib`, 23 entries), and 300 DPI figures (`figures/fig1–fig6`). |
| [`reports/`](reports/) | One concise markdown report per stage: [`STAGE_1_AUDIT_REPORT.md`](reports/STAGE_1_AUDIT_REPORT.md) through [`STAGE_9_FINAL_PUBLICATION_VERIFICATION.md`](reports/STAGE_9_FINAL_PUBLICATION_VERIFICATION.md). |

---

## Stage 1–9 Reports Index

| Stage | Report | Core Output / Milestone |
|:---:|---|---|
| **1** | [`STAGE_1_AUDIT_REPORT.md`](reports/STAGE_1_AUDIT_REPORT.md) | Forensic repo + schema audit (`src/` vs. `src_v2/`, 46-vs-39 features, XGBoost weighting bug, 10.14% pre-dedup leakage, array-vs-wire fallacy). |
| **2** | [`STAGE_2_INVESTIGATION_REPORT.md`](reports/STAGE_2_INVESTIGATION_REPORT.md) | Duplicate/collision census (`28,464` exact dups, `8,740` collision hashes), atomic 70/15/15 hash partitioning, `Merged26–40` cross-capture extraction. |
| **3** | [`STAGE_3_COMPLETE_REPORT.md`](reports/STAGE_3_COMPLETE_REPORT.md) | Canary check, 21-model training/validation benchmark, decomposed common-clean (`72,390`, `Δ=+0.0022`) vs. ambiguous (`41.73%`), latency/RSS telemetry. |
| **4** | [`STAGE_4_EVALUATION_REPORT.md`](reports/STAGE_4_EVALUATION_REPORT.md) | Sealed read-only evaluation on Internal Test (`75,105`) & Cross-Capture (`57,280`) across all 21 models (`|Δ_gen| <= 0.0079`). |
| **5** | [`STAGE_5_ERROR_ANALYSIS.md`](reports/STAGE_5_ERROR_ANALYSIS.md) | Worst-10 classes, top off-diagonal confusions (`83.3%` intra-tier), Benign FNs (`RECON-OSSCAN` 21.06%), balanced vs. unweighted & S1 vs. S2 error patterns. |
| **6** | [`STAGE_6_RESEARCH_SYNTHESIS.md`](reports/STAGE_6_RESEARCH_SYNTHESIS.md) | Research questions, `StandardScaler` tree parity audit, synthesized contributions, explicit limitations, and calibrated claims taxonomy. |
| **7** | [`STAGE_7_LITERATURE_REVIEW.md`](reports/STAGE_7_LITERATURE_REVIEW.md) | Systematic audit & comparison matrix of 14 peer-reviewed IoT NIDS studies (`research/`) and four empirical literature gaps. |
| **8** | [`STAGE_8_CONSOLIDATION_AND_MANUSCRIPT.md`](reports/STAGE_8_CONSOLIDATION_AND_MANUSCRIPT.md) | Certified Stage 8A–8D reconciliation (pre/post-dedup arithmetic proof, `82.58% ± 1.05%` 5-seed leakage, cross-capture prevalence shift proof, profiling fixes, paper plan). |
| **9** | [`STAGE_9_FINAL_PUBLICATION_VERIFICATION.md`](reports/STAGE_9_FINAL_PUBLICATION_VERIFICATION.md) | Final 11-page PDF visual & numerical audit resolving 10 manuscript review checks and certifying submission readiness. |

---

## Commands (Run From Repository Root)

```powershell
# Verify 0.00% feature-vector overlap across Train, Val, Test & Cross-Capture
.\.venv\Scripts\python.exe -u track1_dataset_evaluation\src\verify_partitions.py

# Re-run read-only Stage 4 evaluation on sealed Internal Test & Cross-Capture holdout
.\.venv\Scripts\python.exe -m track1_dataset_evaluation.src.run_stage4_eval

# Regenerate Figures 1-6 (300 DPI) and compile paper/research_paper.pdf via Typst
.\.venv\Scripts\python.exe track1_dataset_evaluation/scripts/generate_paper_figures.py
.\.venv\Scripts\python.exe -c "import typst; typst.compile('track1_dataset_evaluation/paper/paper.typ', output='track1_dataset_evaluation/paper/research_paper.pdf')"
```

*(For LaTeX compilation via Overleaf or local TeXLive, upload `paper/` and run `pdflatex main.tex` -> `bibtex main` -> `pdflatex main.tex` twice. Related notebooks & Track 2 IDS work are indexed from [`../README.md`](../README.md) and [`../notebooks/07_uncapped_high_accuracy_benchmark.ipynb`](../notebooks/07_uncapped_high_accuracy_benchmark.ipynb).)*
