# Utility & Demonstration Scripts (`scripts/`)

This directory houses executive demonstration tools and performance benchmarking scripts.

---

## Scripts Overview

### 1. Security Operations Center (SOC) Console: `demo_app.py`
An interactive Streamlit-based dashboard simulating real-time packet inspection, tiered threat escalation, and wire-speed packet triage on the CICIoT2023 dataset.

**Run Command**:
```powershell
streamlit run scripts/demo_app.py
```

### 2. Empirical Throughput Benchmark: `benchmark_throughput.py`
Measures single-packet latency ($\mu\text{s}$) and sustained packet processing throughput (Packets Per Second) for edge models (LightGBM, XGBoost, Random Forest, Logistic Regression, PyTorch Tabular DNN) on host CPU hardware.

**Run Command**:
```powershell
python scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```
Outputs are written to `results/tables/throughput_benchmark.csv` and `results/figures/throughput_benchmark.png`.
