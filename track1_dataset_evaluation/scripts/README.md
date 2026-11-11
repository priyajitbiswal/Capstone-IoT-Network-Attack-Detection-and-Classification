# Track 1 Publication Scripts (`track1_dataset_evaluation/scripts/`)

This directory houses Track 1 visualization utilities for the academic manuscript. *(Operational IDS tools `demo_app.py` and `benchmark_throughput.py` reside in [`../../track2_ids_engineering/scripts/`](../../track2_ids_engineering/scripts/).)*

---

## Publication Figure Generator: `generate_paper_figures.py`
Standalone, deterministic Python script that reads experimental metrics from [`../results/tables/`](../results/tables/) and generates high-resolution (300 DPI) publication figures embedded in [`../paper/main.tex`](../paper/main.tex) and [`../paper/paper.typ`](../paper/paper.typ):
- `fig1_partitioning_protocol.png`: Schematic of the Zero-Leakage Atomic Hash Partitioning Protocol.
- `fig2_accuracy_vs_macro_f1.png`: In-distribution test generalization scatter across 21 models.
- `fig3_minority_recall_collapse.png`: The Accuracy Paradox: minority attack recall disparity under unweighted training.
- `fig4_threat_tier_cross_capture.png`: Temporal cross-capture generalization across threat tiers.
- `fig5_latency_pareto_frontier.png`: Reconciled host-side P50 latency vs. Macro-F1 Pareto curve.
- `fig6_error_topology_matrix.png`: Fine-grained 34-class confusion matrix highlighting error clusters.

**Run Command (from repository root)**:
```powershell
.\.venv\Scripts\python.exe track1_dataset_evaluation/scripts/generate_paper_figures.py
```
