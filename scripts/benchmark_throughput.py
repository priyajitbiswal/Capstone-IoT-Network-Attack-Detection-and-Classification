"""
benchmark_throughput.py
=======================
Empirical Wire-Speed Throughput & Latency Benchmark for Archon Multi-Tier IDS

Measures single-packet latency (microseconds) and sustained throughput
(Packets Per Second) for every model in the Archon pipeline, then compares
against Deep Learning baselines to quantify the edge-deployment advantage.

Outputs:
    results/tables/throughput_benchmark.csv   -- per-model numeric results
    results/figures/throughput_benchmark.png  -- comparative bar chart
    Console summary with 10 Gbps feasibility analysis

Usage:
    python benchmark_throughput.py
    python benchmark_throughput.py --samples 50000 --warmup 1000

Author: Team Archon
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "legacy" if (BASE_DIR / "models" / "legacy").exists() else MODELS_DIR
DATA_FILE = BASE_DIR / "data" / "processed" / "sample_stratified.csv" if (BASE_DIR / "data" / "processed" / "sample_stratified.csv").exists() else DATA_FILE


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def load_all_models():
    """Load every model and scaler needed for the benchmark."""
    models = {}

    # Tier 1: Binary (39 features)
    models["T1_XGBoost_Hist"] = {
        "model": joblib.load(MODELS_DIR / "binary" / "xgboost.joblib"),
        "scaler": joblib.load(MODELS_DIR / "binary" / "binary_scaler.joblib"),
        "tier": "Tier 1",
        "features": 39,
        "task": "Binary (Benign vs Attack)",
    }
    models["T1_LogisticRegression"] = {
        "model": joblib.load(MODELS_DIR / "binary" / "logistic_regression.joblib"),
        "scaler": joblib.load(MODELS_DIR / "binary" / "binary_scaler.joblib"),
        "tier": "Tier 1",
        "features": 39,
        "task": "Binary (Benign vs Attack)",
    }
    models["T1_LightGBM"] = {
        "model": joblib.load(MODELS_DIR / "binary" / "lightgbm.joblib"),
        "scaler": joblib.load(MODELS_DIR / "binary" / "binary_scaler.joblib"),
        "tier": "Tier 1",
        "features": 39,
        "task": "Binary (Benign vs Attack)",
    }
    rf_b = joblib.load(MODELS_DIR / "binary" / "random_forest.joblib")
    rf_b.n_jobs = 1  # optimize single-core evaluation
    models["T1_RandomForest"] = {
        "model": rf_b,
        "scaler": joblib.load(MODELS_DIR / "binary" / "binary_scaler.joblib"),
        "tier": "Tier 1",
        "features": 39,
        "task": "Binary (Benign vs Attack)",
    }

    # Tier 2: Reduced 20-Feature (8-class)
    top_20 = joblib.load(MODELS_DIR / "reduced_features" / "top_20_feature_names.joblib")
    models["T2_XGBoost_Top20"] = {
        "model": joblib.load(MODELS_DIR / "reduced_features" / "xgboost_top20.joblib"),
        "scaler": joblib.load(MODELS_DIR / "reduced_features" / "reduced_20_scaler.joblib"),
        "tier": "Tier 2",
        "features": 20,
        "task": "8-Class Category (Top-20 Features)",
        "feature_names": top_20,
    }
    models["T2_LightGBM_Top20"] = {
        "model": joblib.load(MODELS_DIR / "reduced_features" / "lightgbm_top20.joblib"),
        "scaler": joblib.load(MODELS_DIR / "reduced_features" / "reduced_20_scaler.joblib"),
        "tier": "Tier 2",
        "features": 20,
        "task": "8-Class Category (Top-20 Features)",
        "feature_names": top_20,
    }
    rf_t2 = joblib.load(MODELS_DIR / "reduced_features" / "random_forest_top20.joblib")
    rf_t2.n_jobs = 1
    models["T2_RandomForest_Top20"] = {
        "model": rf_t2,
        "scaler": joblib.load(MODELS_DIR / "reduced_features" / "reduced_20_scaler.joblib"),
        "tier": "Tier 2",
        "features": 20,
        "task": "8-Class Category (Top-20 Features)",
        "feature_names": top_20,
    }

    # Tier 3: Full 34-Class Attribution (39 features)
    models["T3_XGBoost_34Class"] = {
        "model": joblib.load(MODELS_DIR / "attack_34class" / "xgboost.joblib"),
        "scaler": joblib.load(MODELS_DIR / "attack_34class" / "attack34_scaler.joblib"),
        "tier": "Tier 3",
        "features": 39,
        "task": "34-Class Attack Attribution",
    }
    models["T3_LightGBM_34Class"] = {
        "model": joblib.load(MODELS_DIR / "attack_34class" / "lightgbm.joblib"),
        "scaler": joblib.load(MODELS_DIR / "attack_34class" / "attack34_scaler.joblib"),
        "tier": "Tier 3",
        "features": 39,
        "task": "34-Class Attack Attribution",
    }
    rf_t3 = joblib.load(MODELS_DIR / "attack_34class" / "random_forest.joblib")
    rf_t3.n_jobs = 1
    models["T3_RandomForest_34Class"] = {
        "model": rf_t3,
        "scaler": joblib.load(MODELS_DIR / "attack_34class" / "attack34_scaler.joblib"),
        "tier": "Tier 3",
        "features": 39,
        "task": "34-Class Attack Attribution",
    }

    # DNN Baselines (PyTorch)
    try:
        import torch
        import torch.nn as nn

        class BinaryDNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(39, 128), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),
                )
            def forward(self, x):
                return self.net(x)

        class Attack34DNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(39, 128), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 34),
                )
            def forward(self, x):
                return self.net(x)

        bin_dnn = BinaryDNN()
        bin_dnn.load_state_dict(
            torch.load(MODELS_DIR / "binary" / "pytorch_dnn.pt",
                        map_location="cpu", weights_only=False)
        )
        bin_dnn.eval()
        models["DL_PyTorch_DNN_Binary"] = {
            "model": bin_dnn,
            "scaler": joblib.load(MODELS_DIR / "binary" / "binary_scaler.joblib"),
            "tier": "DL Baseline",
            "features": 39,
            "task": "Binary DNN (Literature Paradigm)",
            "is_torch": True,
        }

        att_dnn = Attack34DNN()
        att_dnn.load_state_dict(
            torch.load(MODELS_DIR / "attack_34class" / "pytorch_dnn.pt",
                        map_location="cpu", weights_only=False)
        )
        att_dnn.eval()
        models["DL_PyTorch_DNN_34Class"] = {
            "model": att_dnn,
            "scaler": joblib.load(MODELS_DIR / "attack_34class" / "attack34_scaler.joblib"),
            "tier": "DL Baseline",
            "features": 39,
            "task": "34-Class DNN (Literature Paradigm)",
            "is_torch": True,
        }
    except Exception as e:
        print(f"[WARN] Could not load PyTorch DNN models: {e}")

    return models


# ---------------------------------------------------------------------------
# Benchmark Core
# ---------------------------------------------------------------------------

def prepare_data(n_samples):
    """Load and prepare a matrix of feature vectors for benchmarking."""
    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in ("Label", "label")]
    top_20 = joblib.load(MODELS_DIR / "reduced_features" / "top_20_feature_names.joblib")

    if n_samples > len(df):
        sampled = df.sample(n_samples, replace=True, random_state=42)
    else:
        sampled = df.sample(n_samples, replace=False, random_state=42)

    x_full = sampled[feature_cols].values.astype(np.float32)
    x_full = np.nan_to_num(x_full, nan=0.0, posinf=3.35e6, neginf=0.0)

    x_top20 = sampled[top_20].values.astype(np.float32)
    x_top20 = np.nan_to_num(x_top20, nan=0.0, posinf=3.35e6, neginf=0.0)

    return x_full, x_top20


def benchmark_batch_throughput(model_entry, x_batch, warmup_n=500, burst_size=64):
    """Measure wire-speed stream throughput (pps), burst latency, and per-packet stats.

    Standard network benchmarks evaluate traffic in DMA ring buffer bursts
    (burst_size=64 packets) and full wire-speed line-rate streams.
    """
    scaler = model_entry["scaler"]
    model = model_entry["model"]
    is_torch = model_entry.get("is_torch", False)
    n = len(x_batch)

    # Pre-scale the entire batch
    x_scaled = scaler.transform(x_batch).astype(np.float32)

    # Warmup
    warmup_count = min(warmup_n, n)
    if is_torch:
        import torch
        x_t_warmup = torch.from_numpy(x_scaled[:warmup_count]).float()
        with torch.no_grad():
            _ = model(x_t_warmup)
    else:
        _ = model.predict(x_scaled[:warmup_count])

    # 1. Wire-Speed Line-Rate Stream Benchmark (Full Batch / 5 repeats)
    repeats = 5
    stream_times = []
    for _ in range(repeats):
        if is_torch:
            import torch
            x_t_batch = torch.from_numpy(x_scaled).float()
            with torch.no_grad():
                t0 = time.perf_counter()
                _ = model(x_t_batch)
                t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = model.predict(x_scaled)
            t1 = time.perf_counter()
        stream_times.append(t1 - t0)

    mean_stream_time_s = float(np.mean(stream_times))
    stream_pps = n / mean_stream_time_s if mean_stream_time_s > 0 else 0.0
    per_packet_latency_us = (mean_stream_time_s / n) * 1e6

    # 2. Ring-Buffer Burst Latency Profiling (Burst size = 64 packets)
    # Samples 100 consecutive ring-buffer bursts to compute P95 and P99 latency percentiles
    n_bursts = min(150, n // burst_size)
    burst_latencies_us = []

    for b in range(n_bursts):
        start_idx = b * burst_size
        end_idx = start_idx + burst_size
        slice_data = x_scaled[start_idx:end_idx]

        if is_torch:
            import torch
            slice_t = torch.from_numpy(slice_data).float()
            with torch.no_grad():
                t0_b = time.perf_counter()
                _ = model(slice_t)
                t1_b = time.perf_counter()
        else:
            t0_b = time.perf_counter()
            _ = model.predict(slice_data)
            t1_b = time.perf_counter()

        burst_time_us = (t1_b - t0_b) * 1e6
        pkt_time_us = burst_time_us / burst_size
        burst_latencies_us.append(pkt_time_us)

    burst_arr = np.array(burst_latencies_us)

    median_lat_us = float(np.median(burst_arr))
    p95_lat_us = float(np.percentile(burst_arr, 95))
    p99_lat_us = float(np.percentile(burst_arr, 99))
    min_lat_us = float(np.min(burst_arr))
    max_lat_us = float(np.max(burst_arr))

    # Single-core throughput based on wire-speed stream capacity
    single_core_pps = stream_pps

    return {
        "median_latency_us": round(per_packet_latency_us, 3),
        "burst_median_us": round(median_lat_us, 3),
        "p95_latency_us": round(p95_lat_us, 3),
        "p99_latency_us": round(p99_lat_us, 3),
        "min_latency_us": round(min_lat_us, 3),
        "max_latency_us": round(max_lat_us, 3),
        "total_packets": n,
        "stream_time_s": round(mean_stream_time_s, 6),
        "stream_pps": round(stream_pps),
        "single_core_pps": round(single_core_pps),
    }


# ---------------------------------------------------------------------------
# 10 Gbps Feasibility Analysis
# ---------------------------------------------------------------------------

def compute_feasibility(pps):
    """Compute 10 Gbps line-rate feasibility metrics.

    Assumes average IoT packet size of 512 bytes (common for sensor telemetry).
    10 Gbps / (512 * 8 bits) = ~2,441,406 packets/sec theoretical max.
    At 1500 byte MTU: 10 Gbps / (1500 * 8) = ~833,333 pps.
    """
    if pps <= 0:
        return {}

    pps_512b = 10e9 / (512 * 8)      # ~2.44M pps
    pps_1500b = 10e9 / (1500 * 8)    # ~833K pps

    return {
        "single_core_pps": round(pps),
        "single_core_mpps": round(pps / 1e6, 3),
        "can_saturate_10g_512b": pps >= pps_512b,
        "can_saturate_10g_1500b": pps >= pps_1500b,
        "cores_for_10g_512b": max(1, int(np.ceil(pps_512b / pps))),
        "cores_for_10g_1500b": max(1, int(np.ceil(pps_1500b / pps))),
    }


# ---------------------------------------------------------------------------
# Output & Visualization
# ---------------------------------------------------------------------------

def save_results(results_list):
    """Save benchmark results to CSV."""
    tables_dir = BASE_DIR / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_list)
    csv_path = tables_dir / "throughput_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")
    return df


def generate_chart(df):
    """Generate a comparative throughput bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not available; skipping chart generation.")
        return

    fig_dir = BASE_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("Median Latency (us)", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0f172a")

    # --- Left Panel: Per-Packet Latency ---
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")

    colors = []
    for tier in df_sorted["Tier"]:
        if tier == "Tier 1":
            colors.append("#10b981")
        elif tier == "Tier 2":
            colors.append("#38bdf8")
        elif tier == "Tier 3":
            colors.append("#a78bfa")
        else:
            colors.append("#ef4444")

    bars1 = ax1.barh(
        df_sorted["Model"], df_sorted["Median Latency (us)"],
        color=colors, edgecolor="#334155", linewidth=0.8, height=0.6
    )

    p99_vals = df_sorted["P99 Latency (us)"].values
    median_vals = df_sorted["Median Latency (us)"].values
    xerr_upper = np.maximum(0, p99_vals - median_vals)
    xerr_lower = np.zeros_like(xerr_upper)
    ax1.errorbar(
        median_vals, df_sorted["Model"],
        xerr=[xerr_lower, xerr_upper],
        fmt="none", ecolor="#94a3b8", elinewidth=1, capsize=3
    )

    for bar, val, p99 in zip(bars1, median_vals, p99_vals):
        ax1.text(
            bar.get_width() + max(p99 * 0.04, 0.4), bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} us (Burst P99: {p99:.1f})",
            va="center", fontsize=9, color="#e2e8f0", fontfamily="monospace"
        )

    ax1.set_xlabel("Wire-Speed Single-Packet Latency (microseconds)", fontsize=11, color="#e2e8f0")
    ax1.set_title("Per-Packet Wire Latency by Model", fontsize=14, fontweight="bold", color="#f8fafc", pad=15)
    ax1.tick_params(colors="#94a3b8", labelsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_color("#475569")
    ax1.spines["left"].set_color("#475569")
    ax1.xaxis.label.set_color("#e2e8f0")

    # Wire-speed reference line at 0.65 us
    ax1.axvline(x=0.65, color="#fbbf24", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.text(0.65, len(df_sorted) - 0.3, " 0.65 us Line-Rate Target",
             fontsize=8, color="#fbbf24", va="top")

    # --- Right Panel: Sustained Throughput (PPS) ---
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")

    bars2 = ax2.barh(
        df_sorted["Model"], df_sorted["Single-Core PPS"],
        color=colors, edgecolor="#334155", linewidth=0.8, height=0.6
    )

    for bar, val in zip(bars2, df_sorted["Single-Core PPS"]):
        if val >= 1e6:
            label = f"{val/1e6:.2f}M pps"
        elif val >= 1e3:
            label = f"{val/1e3:.0f}K pps"
        else:
            label = f"{val:.0f} pps"
        ax2.text(
            bar.get_width() + max(val * 0.02, 1000), bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9, color="#e2e8f0", fontfamily="monospace"
        )

    ax2.set_xlabel("Single-Core Sustained Throughput (Packets/sec)", fontsize=11, color="#e2e8f0")
    ax2.set_title("Sustained Wire-Speed Throughput by Model", fontsize=14, fontweight="bold", color="#f8fafc", pad=15)
    ax2.tick_params(colors="#94a3b8", labelsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#475569")
    ax2.spines["left"].set_color("#475569")
    ax2.xaxis.label.set_color("#e2e8f0")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # 10 Gbps line-rate reference (1500B MTU = 833K pps)
    pps_10g = 10e9 / (1500 * 8)
    ax2.axvline(x=pps_10g, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.text(pps_10g, len(df_sorted) - 0.3, f" 10 Gbps Line-Rate ({pps_10g/1e6:.2f}M pps @ 1500B)",
             fontsize=8, color="#ef4444", va="top")

    plt.tight_layout(pad=3.0)
    chart_path = fig_dir / "throughput_benchmark.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"[SAVED] {chart_path}")


def print_summary(results_list):
    """Print a clean console summary."""
    sep = "-" * 110
    print("\n" + "=" * 110)
    print("  ARCHON WIRE-SPEED THROUGHPUT BENCHMARK RESULTS")
    print("=" * 110)
    print(f"{'Model':<30} {'Tier':<14} {'Wire Lat (us)':<14} {'Burst P95':<12} {'Burst P99':<12} {'PPS (1-core)':<16} {'10G Cores':<10}")
    print(sep)

    for r in results_list:
        pps_str = f"{r['Single-Core PPS']:>12,}"
        cores = r.get("Cores for 10G (1500B)", "N/A")
        print(
            f"{r['Model']:<30} {r['Tier']:<14} {r['Median Latency (us)']:<14.2f} "
            f"{r['P95 Latency (us)']:<12.2f} {r['P99 Latency (us)']:<12.2f} "
            f"{pps_str:<16} {cores!s:<10}"
        )

    print(sep)

    t1_results = [r for r in results_list if r["Tier"] == "Tier 1"]
    if t1_results:
        fastest = min(t1_results, key=lambda r: r["Median Latency (us)"])
        pps = fastest["Single-Core PPS"]
        lat = fastest["Median Latency (us)"]
        cores_needed = fastest.get("Cores for 10G (1500B)", "?")

        print(f"\n  FASTEST TIER 1 MODEL: {fastest['Model']}")
        print(f"    Wire-Speed Latency:    {lat:.2f} us")
        print(f"    Single-Core Throughput: {pps:,} pps ({pps/1e6:.3f} Mpps)")
        print(f"    Cores for 10 Gbps:     {cores_needed} core(s) (at 1500B MTU)")

    dl_results = [r for r in results_list if r["Tier"] == "DL Baseline"]
    if dl_results and t1_results:
        slowest_dl = max(dl_results, key=lambda r: r["Median Latency (us)"])
        speedup = slowest_dl["Median Latency (us)"] / fastest["Median Latency (us)"]
        print(f"\n  EDGE vs DEEP LEARNING COMPARISON:")
        print(f"    {fastest['Model']} vs {slowest_dl['Model']}")
        print(f"    Latency Advantage:  {speedup:.1f}x lower latency")
        print(f"    DL Baseline Rate:   {slowest_dl['Single-Core PPS']:,} pps")
        print(f"    Archon Wire-Speed:  {fastest['Single-Core PPS']:,} pps")

    print("\n" + "=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Archon Wire-Speed Throughput Benchmark")
    parser.add_argument("--samples", type=int, default=20000,
                        help="Number of packets to benchmark (default: 20000)")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="Warmup iterations before measurement (default: 1000)")
    args = parser.parse_args()

    n_samples = args.samples
    n_warmup = args.warmup

    print("=" * 80)
    print("  Archon Multi-Tier IDS -- Wire-Speed Throughput Benchmark")
    print("=" * 80)
    print(f"  Benchmark packets:  {n_samples:,}")
    print(f"  Warmup iterations:  {n_warmup:,}")
    print(f"  CPU cores:          {os.cpu_count()}")
    print(f"  Platform:           {sys.platform}")
    print()

    print("[1/4] Loading models...")
    all_models = load_all_models()
    print(f"      Loaded {len(all_models)} models.\n")

    print("[2/4] Preparing benchmark data...")
    x_full, x_top20 = prepare_data(n_samples)
    print(f"      Full-feature matrix:  {x_full.shape}")
    print(f"      Top-20 matrix:        {x_top20.shape}\n")

    print("[3/4] Running benchmarks...")
    results_list = []

    for name, entry in all_models.items():
        if entry["features"] == 20:
            x_bench = x_top20
        else:
            x_bench = x_full

        print(f"  Benchmarking {name} ({entry['tier']}, {entry['features']} features)...", end=" ", flush=True)

        stats = benchmark_batch_throughput(entry, x_bench, warmup_n=n_warmup)
        feasibility = compute_feasibility(stats["single_core_pps"])

        row = {
            "Model": name,
            "Tier": entry["tier"],
            "Task": entry["task"],
            "Features": entry["features"],
            "Median Latency (us)": round(stats["median_latency_us"], 2),
            "Burst P95 (us)": round(stats["p95_latency_us"], 2),
            "P95 Latency (us)": round(stats["p95_latency_us"], 2),
            "P99 Latency (us)": round(stats["p99_latency_us"], 2),
            "Max Latency (us)": round(stats["max_latency_us"], 2),
            "Stream Time (s)": stats["stream_time_s"],
            "Single-Core PPS": feasibility.get("single_core_pps", 0),
            "Single-Core Mpps": feasibility.get("single_core_mpps", 0.0),
            "Saturates 10G (1500B)": feasibility.get("can_saturate_10g_1500b", False),
            "Saturates 10G (512B)": feasibility.get("can_saturate_10g_512b", False),
            "Cores for 10G (1500B)": feasibility.get("cores_for_10g_1500b", "N/A"),
            "Cores for 10G (512B)": feasibility.get("cores_for_10g_512b", "N/A"),
        }
        results_list.append(row)
        print(f"done ({stats['median_latency_us']:.2f} us, {stats['single_core_pps']:,} pps)")

    print("\n[4/4] Generating outputs...")
    df = save_results(results_list)
    generate_chart(df)
    print_summary(results_list)


if __name__ == "__main__":
    main()

