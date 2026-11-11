# Track 2 IDS Deployment & Profiling Scripts (`track2_ids_engineering/scripts/`)

This directory holds the operational IDS demonstration and hardware profiling scripts:

- **`demo_app.py`**: Interactive Streamlit Security Operations Center (SOC) console simulating real-time packet/flow ingress, multi-tier threat escalation, and automated mitigation rules using the [`../models/`](../models/README.md) Tier 1–3 models.
- **`benchmark_throughput.py`**: Profiles single-packet wire latency, DMA burst P99 latency, and sustained throughput (Packets Per Second) across Tier 1, Tier 2 (Top-20), Tier 3 (34-class), and PyTorch DNN models (`../results/tables/throughput_benchmark.csv`, `../results/figures/throughput_benchmark.png`).

**Run Commands (from repository root)**:
```powershell
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py
.\.venv\Scripts\python.exe track2_ids_engineering/scripts/benchmark_throughput.py --samples 50000 --warmup 1000
```
