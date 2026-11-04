# Utility & Demonstration Scripts (`scripts/`)

This directory houses executive demonstration tools, performance benchmarking utilities, and publication figure generators.

---

## Scripts Overview

### 1. Publication Figure Generator: `generate_paper_figures.py`
Standalone, deterministic Python script that reads experimental metrics from [`results_v2/tables/`](../results_v2/tables/) and generates high-resolution (300 DPI) publication figures embedded in [`paper/main.tex`](../paper/main.tex) and [`paper/paper.typ`](../paper/paper.typ):
- `fig1_partitioning_protocol.png`: Schematic of the Zero-Leakage Atomic Hash Partitioning Protocol.
- `fig2_accuracy_vs_macro_f1.png`: In-distribution test generalization scatter across 21 models.
- `fig3_minority_recall_collapse.png`: The Accuracy Paradox: minority attack recall disparity under unweighted training.
- `fig4_threat_tier_cross_capture.png`: Temporal cross-capture generalization across threat tiers.
- `fig5_latency_pareto_frontier.png`: Reconciled host-side P50 latency vs. Macro-F1 Pareto curve.
- `fig6_error_topology_matrix.png`: Fine-grained 34-class confusion matrix highlighting error clusters.

**Run Command**:
```powershell
.\.venv\Scripts\python.exe scripts/generate_paper_figures.py
```

---

### 2. Security Operations Center (SOC) Console: `demo_app.py`
An interactive Streamlit-based dashboard simulating real-time packet inspection, tiered threat escalation, and multi-model packet triage on the CICIoT2023 dataset.

**Run Command**:
```powershell
.\.venv\Scripts\streamlit.exe run scripts/demo_app.py
```

---

### 3. Host-Side Throughput & Latency Benchmark: `benchmark_throughput.py`
Measures single-packet latency (µs) and batch packet processing throughput (Packets Per Second) for edge models on host CPU hardware (Intel Core i5-1135G7).

**Run Command**:
```powershell
.\.venv\Scripts\python.exe scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```
Outputs are written to `results/tables/throughput_benchmark.csv` and `results/figures/throughput_benchmark.png`.
