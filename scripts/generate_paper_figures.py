#!/usr/bin/env python3
"""
generate_paper_figures.py

Reproducible script to generate publication-quality figures for the research manuscript:
"Unmasking Data Leakage and Temporal Fragility in IoT Network Intrusion Detection:
An Empirical Audit and Benchmark on CICIoT2023"

Stage 8D Audited and Verified Version.
All coordinates, latencies, throughputs, and metric values reconciled against:
- results_v2/tables/centralized_experiment_ledger.csv
- results_v2/tables/stage4_test_results.csv
- reports/STAGE_8A2_AUDIT_VERIFICATION.md
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_figure_1():
    """Figure 1: Schematic Diagram of the Atomic Hash-Grouped Partitioning Protocol."""
    print("Generating Figure 1: Partitioning Protocol Architecture...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')

    c_blue = '#1f77b4'
    c_orange = '#ff7f0e'
    c_green = '#2ca02c'
    c_red = '#d62728'
    c_purple = '#9467bd'
    c_gray = '#7f7f7f'

    def draw_box(x, y, w, h, text, color, bg='#f0f4f8', alpha=0.9, title_fontsize=10, body_fontsize=8):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03",
            ec=color, fc=bg, lw=1.8, alpha=alpha
        )
        ax.add_patch(box)
        lines = text.split('\n')
        title = lines[0]
        body = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        ax.text(x + w / 2, y + h * 0.72 if body else y + h / 2, title,
                ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color='#1a1a1a')
        if body:
            ax.text(x + w / 2, y + h * 0.35, body,
                    ha='center', va='center', fontsize=body_fontsize, color='#333333')

    # Step 1: Raw Stratified Sample
    draw_box(0.05, 0.75, 0.38, 0.18,
             "Full Stratified Sample (N = 529,562)\n39 Continuous/Discrete Features + Class Label\nContains 28,464 Exact Duplicate Rows (5.38%)",
             c_blue, bg='#e8f1f5')

    # Step 2: Unique Stratum
    draw_box(0.57, 0.75, 0.38, 0.18,
             "Unique Feature-Label Stratum (N = 501,098)\n491,422 Unique 39-Feature Coordinates\n8,740 Conflicting Hashes (18,416 flows, 3.68%)",
             c_orange, bg='#fef3e7')

    # Arrow 1: Deduplication
    ax.annotate('', xy=(0.56, 0.84), xytext=(0.44, 0.84),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2, headwidth=8))
    ax.text(0.50, 0.87, "Exact-Row\nDeduplication", ha='center', va='bottom', fontsize=8, color=c_gray)

    # Step 3: Atomic Hash Grouping
    draw_box(0.20, 0.46, 0.60, 0.18,
             "Atomic SHA-256 Group-Hash Partitioning Standard\nUnique 39-Feature Coordinate Hash -> Atomic Disjoint Partition Assignment\nPreprocessor Isolation: Scaler (mu, sigma) & Limits Fitted Strictly on Train",
             c_purple, bg='#f4eef9')

    # Arrow 2
    ax.annotate('', xy=(0.50, 0.65), xytext=(0.76, 0.74),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, width=2, headwidth=8))

    # Step 4A: Strategy 1
    draw_box(0.05, 0.10, 0.40, 0.26,
             "Strategy 1: Primary Partition (N = 501,098)\nRetains Collisions (Real-World Network Ambiguity)\n- Train (70%): 350,821 flows\n- Val (15%): 75,172 flows (2,782 ambiguous)\n- Test (15%): 75,105 flows (Zero Overlap)",
             c_red, bg='#fdf0f0')

    # Step 4B: Strategy 2
    draw_box(0.55, 0.10, 0.40, 0.26,
             "Strategy 2: Collision-Purged (N = 482,682)\nPurges All 8,740 Conflicting Feature Hashes\n- Train (70%): 337,838 flows\n- Val (15%): 72,390 flows (Common-Clean)\n- Test (15%): 72,454 flows (Zero Overlap)",
             c_green, bg='#eafaf1')

    # Arrows 3 & 4
    ax.annotate('', xy=(0.25, 0.37), xytext=(0.38, 0.45),
                arrowprops=dict(facecolor=c_red, edgecolor=c_red, width=2, headwidth=7))
    ax.annotate('', xy=(0.75, 0.37), xytext=(0.62, 0.45),
                arrowprops=dict(facecolor=c_green, edgecolor=c_green, width=2, headwidth=7))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig1_partitioning_protocol.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_2():
    """Figure 2: In-Distribution Test Generalization: Macro-F1 vs Accuracy."""
    print("Generating Figure 2: Macro-F1 vs Accuracy Scatter...")
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results_v2', 'tables', 'stage4_test_results.csv')
    df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    task_colors = {
        'binary': '#1f77b4',
        'category': '#2ca02c',
        'fine_grained': '#d62728'
    }
    task_labels = {
        'binary': 'Binary (2-Class)',
        'category': 'Category (8-Class)',
        'fine_grained': 'Fine-Grained (34-Class)'
    }
    model_markers = {
        'random_forest': 's',
        'lightgbm': 'o',
        'xgboost': '^',
        'logistic_regression': 'v',
        'pytorch_dnn': 'D'
    }
    model_display = {
        'random_forest': 'Random Forest',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'logistic_regression': 'SGD Linear',
        'pytorch_dnn': 'PyTorch MLP'
    }

    # Plot points for Strategy 1 primary baselines
    s1_df = df[df['strategy'] == 'Strategy 1 (Primary)']
    for _, row in s1_df.iterrows():
        task = row['target_type']
        model = row['model_name']
        acc = row['test_int_accuracy'] * 100
        f1 = row['test_int_macro_f1']
        color = task_colors.get(task, '#333333')
        marker = model_markers.get(model, 'o')

        ax.scatter(acc, f1, color=color, marker=marker, s=80, alpha=0.85, edgecolors='black', linewidth=0.8)

        # Annotations for key anchors
        if model == 'random_forest' and task == 'binary':
            ax.annotate("RF Binary (89.6%, 0.769)",
                        xy=(acc, f1), xytext=(acc - 5.5, f1 + 0.015),
                        fontsize=7.5, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='#555555'))
        elif model == 'lightgbm' and task == 'binary':
            ax.annotate("LGBM Binary (89.2%, 0.764)",
                        xy=(acc, f1), xytext=(acc - 5.8, f1 - 0.022),
                        fontsize=7.5, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='#555555'))
        elif model == 'lightgbm' and task == 'category':
            ax.annotate("LGBM Category (78.9%, 0.689)",
                        xy=(acc, f1), xytext=(acc - 6.2, f1 + 0.018),
                        fontsize=7.5, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='#555555'))
        elif model == 'lightgbm' and task == 'fine_grained':
            ax.annotate("LGBM Fine (72.7%, 0.615)",
                        xy=(acc, f1), xytext=(acc + 0.8, f1 - 0.018),
                        fontsize=7.5, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='#555555'))

    ax.set_xlabel("Internal Test Accuracy (%)", fontweight='bold')
    ax.set_ylabel("Internal Test Macro-F1 Score", fontweight='bold')
    ax.set_title("In-Distribution Test Generalization: Macro-F1 vs. Accuracy Across 21 Models", fontweight='bold', pad=12)
    ax.set_xlim(60, 93)
    ax.set_ylim(0.48, 0.82)

    from matplotlib.lines import Line2D
    task_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=8, label=lab)
                     for t, lab in task_labels.items() for k, col in task_colors.items() if t == k]
    model_elements = [Line2D([0], [0], marker=m, color='w', markerfacecolor='#555555', markersize=8, label=model_display[k])
                      for k, m in model_markers.items()]

    l1 = ax.legend(handles=task_elements, title="Task Granularity", loc='lower right', frameon=True)
    ax.add_artist(l1)
    ax.legend(handles=model_elements, title="Model Architecture", loc='upper left', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_accuracy_vs_macro_f1.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_3():
    """Figure 3: Minority Attack Recall Disparity under Unweighted Loss Optimization."""
    print("Generating Figure 3: Minority Attack Recall Collapse...")
    attacks = [
        'SQL Injection',
        'Recon-PingSweep',
        'XSS',
        'Backdoor Malware',
        'Browser Hijacking',
        'Command Injection',
        'Dictionary BruteForce',
        'Uploading Attack'
    ]
    balanced_recall = [35.83, 26.40, 20.22, 25.18, 53.06, 32.16, 52.62, 23.33]
    unweighted_recall = [7.48, 4.80, 3.83, 12.23, 31.29, 21.96, 34.18, 6.67]
    rel_drops = [(u - b) / b * 100 for b, u in zip(balanced_recall, unweighted_recall)]

    y = np.arange(len(attacks))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.barh(y + height / 2, balanced_recall, height, label='Balanced (Cost-Sensitive Weighting)', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.barh(y - height / 2, unweighted_recall, height, label='Unweighted (Standard Cross-Entropy)', color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.6)

    for i in range(len(attacks)):
        ax.text(balanced_recall[i] + 1.2, y[i] + height / 2, f"{balanced_recall[i]:.1f}%", va='center', fontsize=8, color='#1b5e20', fontweight='bold')
        ax.text(unweighted_recall[i] + 1.2, y[i] - height / 2, f"{unweighted_recall[i]:.1f}% ({rel_drops[i]:.1f}%)", va='center', fontsize=8, color='#b71c1c', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(attacks, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Internal Test Recall (%)", fontweight='bold')
    ax.set_title("The Accuracy Paradox: Minority Attack Recall Disparity Under Unweighted Loss (LightGBM 34-Class)", fontweight='bold', pad=12)
    ax.set_xlim(0, 68)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_minority_recall_collapse.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_4():
    """Figure 4: Temporal Cross-Capture Generalization across Threat Tiers."""
    print("Generating Figure 4: Threat Tier Cross-Capture Evaluation...")
    tiers = [
        'Tier 1: Volumetric Floods',
        'Tier 2: IoT Botnets (Mirai)',
        'Tier 3: Recon & Spoofing',
        'Tier 4: Web & Exploits',
        'Non-Attack: Benign'
    ]
    int_f1 = [0.7943, 0.9953, 0.4696, 0.1748, 0.6884]
    cross_f1 = [0.8084, 0.9931, 0.4620, 0.3253, 0.6693]

    x = np.arange(len(tiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    ax.bar(x - width / 2, int_f1, width, label='Internal Test (Same Sessions, N=75,105)', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.bar(x + width / 2, cross_f1, width, label='Cross-Capture Holdout (Later Sessions, N=57,280)', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6)

    for i in range(len(tiers)):
        ax.text(x[i] - width / 2, int_f1[i] + 0.02, f"{int_f1[i]:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0d47a1')
        ax.text(x[i] + width / 2, cross_f1[i] + 0.02, f"{cross_f1[i]:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e65100')

    ax.set_ylabel("Threat Tier Macro-F1 Score", fontweight='bold')
    ax.set_title("Temporal Cross-Capture Generalization Across Threat Tiers (Fine-Grained LightGBM)", fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontweight='bold', rotation=12, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig4_threat_tier_cross_capture.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_5():
    """Figure 5: Host-Side Latency vs Detection Macro-F1 Pareto Frontier (Stage 8D Reconciled)."""
    print("Generating Figure 5: Host-Side Latency vs Macro-F1 Pareto Curve...")
    # Exact certified numbers from centralized_experiment_ledger.csv and stage4_test_results.csv
    models = [
        # (name, task, p50_us, macro_f1)
        ('SGD Linear', 'Binary', 362.10, 0.6761),
        ('SGD Linear', 'Category', 143.20, 0.6036),
        ('SGD Linear', 'Fine-Grained', 189.70, 0.5272),
        ('PyTorch MLP', 'Binary', 247.10, 0.7125),
        ('PyTorch MLP', 'Category', 613.80, 0.6130),
        ('PyTorch MLP', 'Fine-Grained', 942.30, 0.5216),
        ('LightGBM', 'Binary', 1983.30, 0.7637),
        ('LightGBM', 'Category', 1409.80, 0.6894),
        ('LightGBM', 'Fine-Grained', 3504.40, 0.6145),
        ('XGBoost', 'Binary', 1575.30, 0.7462),
        ('XGBoost', 'Category', 2244.00, 0.6619),
        ('XGBoost', 'Fine-Grained', 3732.50, 0.6001),
        ('Random Forest', 'Binary', 70883.05, 0.7690),
        ('Random Forest', 'Category', 22467.45, 0.6805),
        ('Random Forest', 'Fine-Grained', 31296.10, 0.5966),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    task_colors = {'Binary': '#1f77b4', 'Category': '#2ca02c', 'Fine-Grained': '#d62728'}
    arch_markers = {'SGD Linear': 'v', 'PyTorch MLP': 'D', 'LightGBM': 'o', 'XGBoost': '^', 'Random Forest': 's'}

    for name, task, lat, f1 in models:
        ax.scatter(lat, f1, color=task_colors[task], marker=arch_markers[name], s=90, alpha=0.85, edgecolors='black', linewidth=0.8)

    # Binary Pareto Frontier:
    # (PyTorch MLP: 247.1 us, 0.7125) -> (XGBoost: 1575.3 us, 0.7462) -> (LightGBM: 1983.3 us, 0.7637) -> (RF: 70883 us, 0.7690)
    pareto_x = [247.10, 1575.30, 1983.30, 70883.05]
    pareto_y = [0.7125, 0.7462, 0.7637, 0.7690]
    ax.plot(pareto_x, pareto_y, 'b--', lw=1.5, alpha=0.7, label='Binary Pareto Frontier')

    # Annotations
    ax.annotate("LightGBM Binary\n(1.98 ms, F1=0.764)", xy=(1983.30, 0.7637), xytext=(2600, 0.730),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='#1f77b4'))
    ax.annotate("Random Forest Binary\n(70.9 ms, F1=0.769)", xy=(70883.05, 0.7690), xytext=(25000, 0.795),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='#1f77b4'))
    ax.annotate("LightGBM Fine\n(3.50 ms, F1=0.615)", xy=(3504.40, 0.6145), xytext=(4800, 0.585),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='#d62728'))
    ax.annotate("PyTorch MLP Binary\n(0.25 ms, F1=0.713)", xy=(247.10, 0.7125), xytext=(120, 0.750),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='#1f77b4'))
    ax.annotate("SGD Linear Category\n(0.14 ms, F1=0.604)", xy=(143.20, 0.6036), xytext=(80, 0.560),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='#2ca02c'))

    ax.set_xscale('log')
    ax.set_xlabel("Single-Flow Host CPU P50 Latency (microseconds, log scale)", fontweight='bold')
    ax.set_ylabel("Internal Test Macro-F1 Score", fontweight='bold')
    ax.set_title("Host-Side Inference Latency vs. Diagnostic Macro-F1 Pareto Frontier", fontweight='bold', pad=12)
    ax.set_ylim(0.48, 0.82)
    ax.set_xlim(50, 120000)

    from matplotlib.lines import Line2D
    task_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=8, label=t)
                     for t, col in task_colors.items()]
    arch_elements = [Line2D([0], [0], marker=m, color='w', markerfacecolor='#555555', markersize=8, label=n)
                     for n, m in arch_markers.items()]

    l1 = ax.legend(handles=task_elements, title="Task Granularity", loc='lower left', frameon=True)
    ax.add_artist(l1)
    ax.legend(handles=arch_elements, title="Model Architecture", loc='upper right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig5_latency_pareto_frontier.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_6():
    """Figure 6: Fine-Grained 34-Class Error Topology Matrix."""
    print("Generating Figure 6: Confusion Matrix / Error Topology...")
    src_cm = os.path.join(OUTPUT_DIR, 'attack34_cm_xgboost.png')
    out_path = os.path.join(OUTPUT_DIR, 'fig6_error_topology_matrix.png')
    if os.path.exists(src_cm):
        shutil.copyfile(src_cm, out_path)
        print(f"Copied {src_cm} to {out_path}")
    else:
        print(f"Warning: {src_cm} not found.")


def main():
    print("Starting generation of publication-quality figures (Stage 8D Audited)...")
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print("All figures successfully regenerated in:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
