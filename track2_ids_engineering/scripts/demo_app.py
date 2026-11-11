"""
Track 2 SOC Defense Console (Streamlit) — Final Presentation Version.

Primary IDS Architecture:
  - E4_LGBM_Leaves63_Sqrt (Single-pass 34-class LightGBM, 75.37% Test Acc, 0.7440 MCC, 4.15 ms Test P50)

Comparative Hierarchical IDS Architecture:
  - E2_Cascade_ValTuned_LGBM (3-Stage Hierarchy using Track 1 binary_lightgbm.joblib in Stage 1,
    tau_att=0.15, Stage 2 malicious-only 7-cat router f_tau=0.40, Stages 3A/3B specialists;
    75.13% Test Acc, 0.6357 Macro-F1, 8.44 ms Test P50)

Strictly reuses Track 1's training-fitted LeakFreePreprocessor (`preprocessor_strategy1.joblib`),
exact canonical 39 `FEATURE_COLUMNS`, and draws unseen test traffic directly from Track 1's
zero-overlap Internal Test partition (`partition == 'test'`, N = 75,105) when available.
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401
from track1_dataset_evaluation.src.config import (
    FEATURE_COLUMNS,
    LABEL_MAPPING_8CLASSES,
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    ORIGINAL_SAMPLE_FILE,
    PARTITION_ASSIGNMENTS_FILE,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import (
    Flat34BaselineArchitecture,
    HierarchicalCascade3Stage,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CATEGORIES,
    HIER_MODELS_DIR,
    HIER_RESULTS_DIR,
)

# Color mappings (preserved SOC console palette)
STATUS_COLORS: Dict[str, str] = {
    "BENIGN_PASSED": "#10B981",       # Emerald
    "FLOOD_DROPPED_EDGE": "#EF4444",  # Crimson
    "BOTNET_QUARANTINED": "#F59E0B",  # Amber
    "EXPLOIT_ESCALATED": "#8B5CF6",   # Violet
}

# Verified 34 -> 29 Operational Mitigation Flood Twin Groups (identical L4/L7 protocol remediation)
DEFAULT_OP29_MERGE_GROUPS: Dict[str, List[str]] = {
    "FLOOD_SYN": ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD", "DDOS-SYNONYMOUSIP_FLOOD"],
    "FLOOD_TCP": ["DDOS-TCP_FLOOD", "DOS-TCP_FLOOD"],
    "FLOOD_UDP": ["DDOS-UDP_FLOOD", "DOS-UDP_FLOOD"],
    "FLOOD_HTTP": ["DDOS-HTTP_FLOOD", "DOS-HTTP_FLOOD"],
}

ARCH_E4_KEY = "E4_Primary_Flat34"
ARCH_E2_KEY = "E2_Fast_Cascade"
ARCH_DISPLAY_OPTIONS: Dict[str, str] = {
    ARCH_E4_KEY: "E4: Improved Flat LightGBM (Primary Recommended — 4.15 ms P50)",
    ARCH_E2_KEY: "E2: 3-Stage Fast Cascade (LightGBM Stage 1 Comparison — 8.44 ms P50)",
}

CUSTOM_CSS = """
<style>
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 96% !important;
}
h1, h2, h3, h4 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155 !important;
    border-radius: 8px;
    padding: 18px 24px;
    margin-top: 0.4rem;
    margin-bottom: 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
}
.header-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.header-sub {
    color: #94a3b8;
    font-size: 13.5px;
    margin-top: 6px;
    margin-bottom: 0;
}
.header-status {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    font-family: monospace;
    letter-spacing: 0.05em;
    background: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0b1120 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    margin-bottom: 16px !important;
}
button[kind="primary"], .stButton > button[kind="primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
    margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 14px;
    color: #94a3b8;
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}
.control-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.badge {
    font-family: monospace;
    font-weight: 700;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    color: #ffffff;
    letter-spacing: 0.04em;
}
</style>
"""


def _build_op29_map() -> Dict[str, str]:
    """Builds exact 34 -> 29 class lookup mapping using frozen JSON artifact when available."""
    map_path = HIER_RESULTS_DIR / "merged_flood_taxonomy_map.json"
    if map_path.exists():
        with open(map_path, "r") as f:
            data = json.load(f)
        if "old_to_new_34_to_29" in data:
            return dict(data["old_to_new_34_to_29"])

    mapping: Dict[str, str] = {}
    for grp_name, members in DEFAULT_OP29_MERGE_GROUPS.items():
        for m in members:
            mapping[m] = grp_name
    for cls_name in THIRTY_FOUR_CLASSES:
        mapping.setdefault(cls_name, cls_name)
    return mapping


def load_soc_resources() -> Dict[str, Any]:
    """Loads Track 1 Strategy 1 preprocessor, E4 primary model, and Fast E2 cascade."""
    prep_path = TRACK1_MODELS_DIR / "preprocessor_strategy1.joblib"
    e4_path = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
    s1_path = TRACK1_MODELS_DIR / "binary_lightgbm.joblib"
    s2_path = HIER_MODELS_DIR / "stage2_cat7_lgbm_sqrt_balanced.joblib"
    s3a_path = HIER_MODELS_DIR / "stage3a_flood_lgbm_d63_sqrt.joblib"
    s3b_path = HIER_MODELS_DIR / "stage3b_nonflood_lgbm_d63_sqrt.joblib"

    required = [prep_path, e4_path, s1_path, s2_path, s3a_path, s3b_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model/preprocessor checkpoint(s): {missing}")

    preprocessor = joblib.load(prep_path)
    e4_arch = Flat34BaselineArchitecture(
        name="E4_LGBM_Leaves63_Sqrt",
        model=joblib.load(e4_path),
        checkpoint_path=e4_path,
    )
    e2_arch = HierarchicalCascade3Stage(
        name="E2_Cascade_ValTuned_LGBM",
        stage1_model=joblib.load(s1_path),
        stage2_model=joblib.load(s2_path),
        flood_model=joblib.load(s3a_path),
        non_flood_model=joblib.load(s3b_path),
        stage2_mode="cat7_attack",
        attack_threshold=0.15,
        flood_route_threshold=0.40,
        checkpoint_paths=[s1_path, s2_path, s3a_path, s3b_path],
    )

    return {
        "preprocessor": preprocessor,
        "architectures": {
            ARCH_E4_KEY: e4_arch,
            ARCH_E2_KEY: e2_arch,
        },
        "op29_map": _build_op29_map(),
        "feature_columns": list(FEATURE_COLUMNS),
    }


def load_evaluation_traffic() -> Tuple[pd.DataFrame, str]:
    """
    Loads real CICIoT2023 flows mapped through Track 1's Strategy 1 partition index so
    dashboard demos replay strictly unseen Internal Test flows (`partition == 'test'`).
    """
    if not ORIGINAL_SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Missing stratified sample dataset: {ORIGINAL_SAMPLE_FILE}")

    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)
    if PARTITION_ASSIGNMENTS_FILE.exists():
        df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
        df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
        df_clean["partition"] = df_assign["partition"].values
        df_test = df_clean[df_clean["partition"] == "test"].drop(columns=["partition"]).reset_index(drop=True)
        df_test["LabelUpper"] = df_test[TARGET_COLUMN].astype(str).str.strip().str.upper()
        return df_test, f"Track 1 Held-Out Internal Test (N = {len(df_test):,}, Zero Overlap)"

    df_raw["LabelUpper"] = df_raw[TARGET_COLUMN].astype(str).str.strip().str.upper()
    return df_raw, f"CICIoT2023 Stratified Sample (N = {len(df_raw):,})"


def preprocess_flows(df_rows: pd.DataFrame, preprocessor: Any) -> np.ndarray:
    """Applies exact Track 1 leak-free cleaning (+inf/-inf/NaN + 39-col order) and StandardScaler."""
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_rows.columns]
    if missing_cols:
        raise ValueError(f"Input frame missing canonical CICIoT2023 features: {missing_cols}")

    ordered_df = df_rows[FEATURE_COLUMNS]
    X_clean = preprocessor.clean_transform(ordered_df)
    X_scaled = preprocessor.transform_scaler(X_clean)
    return X_scaled


def map_verdict_playbook(pred_label_34: str, op29_label: str) -> Dict[str, str]:
    """Maps a predicted class label into an actionable SOC gateway mitigation playbook."""
    if pred_label_34 == "BENIGN":
        return {
            "verdict": "BENIGN_PASSED",
            "tier": "Perimeter Pass",
            "action": "Forward cleanly to internal IoT VLAN",
            "severity": "None",
            "color": STATUS_COLORS["BENIGN_PASSED"],
        }

    cat8 = LABEL_MAPPING_8CLASSES.get(pred_label_34, "Web")
    if cat8 in FLOOD_CATEGORIES:
        if op29_label == "FLOOD_SYN":
            playbook = "SYN Cookies + Embryonic Half-Open Rate-Limit (plus L3 src==dst ACL if spoofed)"
        elif op29_label == "FLOOD_TCP":
            playbook = "Edge TCP Volumetric FlowSpec Drop & State Scrubbing"
        elif op29_label == "FLOOD_UDP":
            playbook = "Upstream UDP Rate-Limit & QoS Ingress Policing"
        elif op29_label == "FLOOD_HTTP":
            playbook = "L7 Reverse-Proxy HTTP Challenge & Connection Throttle"
        else:
            playbook = f"Line-Rate Edge Flood Drop ({op29_label})"
        return {
            "verdict": "FLOOD_DROPPED_EDGE",
            "tier": f"Volumetric Flood Mitigation ({op29_label})",
            "action": playbook,
            "severity": "High (Volumetric Flood)",
            "color": STATUS_COLORS["FLOOD_DROPPED_EDGE"],
        }

    if cat8 == "Mirai":
        return {
            "verdict": "BOTNET_QUARANTINED",
            "tier": "Edge Botnet Quarantine",
            "action": f"Isolate Device Port & Reset Credentials ({pred_label_34})",
            "severity": "Critical (Botnet Compromise)",
            "color": STATUS_COLORS["BOTNET_QUARANTINED"],
        }

    if cat8 in {"Recon", "Spoofing"}:
        action = (
            f"Dynamic ARP / DNSSEC Guard ({pred_label_34})"
            if cat8 == "Spoofing"
            else f"Stealth Probe Block & Firewall Hardening ({pred_label_34})"
        )
        return {
            "verdict": "EXPLOIT_ESCALATED",
            "tier": f"SOC Escalation ({cat8})",
            "action": action,
            "severity": "Elevated (Recon / Spoofing)",
            "color": STATUS_COLORS["EXPLOIT_ESCALATED"],
        }

    return {
        "verdict": "EXPLOIT_ESCALATED",
        "tier": "SOC WAF / Host IR",
        "action": f"Trigger Dynamic WAF Signature & Host Alert ({pred_label_34})",
        "severity": "Severe (Application Exploit)",
        "color": STATUS_COLORS["EXPLOIT_ESCALATED"],
    }


def infer_batch(
    df_rows: pd.DataFrame,
    arch_key: str,
    resources: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Preprocesses and classifies a batch of rows cleanly with timed model inference."""
    arch = resources["architectures"][arch_key]
    preprocessor = resources["preprocessor"]
    op29_map = resources["op29_map"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0_prep = time.perf_counter_ns()
        X_scaled = preprocess_flows(df_rows, preprocessor)
        prep_total_us = (time.perf_counter_ns() - t0_prep) / 1000.0

        t0_inf = time.perf_counter_ns()
        trace = arch.predict_with_trace(X_scaled)
        inf_total_us = (time.perf_counter_ns() - t0_inf) / 1000.0

    y_pred = trace["y_pred"]
    y_prob = trace["y_prob"]
    n_rows = len(df_rows)
    per_flow_prep_us = prep_total_us / max(1, n_rows)
    per_flow_inf_us = inf_total_us / max(1, n_rows)

    flood_indices = [THIRTY_FOUR_TO_IDX[c] for c in THIRTY_FOUR_CLASSES if LABEL_MAPPING_8CLASSES[c] in FLOOD_CATEGORIES]
    non_flood_indices = [
        THIRTY_FOUR_TO_IDX[c]
        for c in THIRTY_FOUR_CLASSES
        if c != "BENIGN" and LABEL_MAPPING_8CLASSES[c] not in FLOOD_CATEGORIES
    ]
    benign_idx = THIRTY_FOUR_TO_IDX["BENIGN"]

    results: List[Dict[str, Any]] = []
    for i in range(n_rows):
        row = df_rows.iloc[i]
        true_raw = str(row[TARGET_COLUMN]).strip().upper()
        true_op29 = op29_map.get(true_raw, true_raw)
        pred_idx = int(y_pred[i])
        pred_34 = THIRTY_FOUR_CLASSES[pred_idx]
        pred_op29 = op29_map.get(pred_34, pred_34)

        probs = y_prob[i]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(THIRTY_FOUR_CLASSES[int(j)], float(probs[int(j)])) for j in top3_idx]

        stage_info: Dict[str, Any] = {
            "benign_prob": float(probs[benign_idx]),
            "flood_prob": float(np.sum(probs[flood_indices])),
            "non_flood_prob": float(np.sum(probs[non_flood_indices])),
            "executions": int(trace["executions_per_flow"][i]),
        }
        if "stage1_prob_attack" in trace:
            stage_info.update({
                "mode": "E2_CASCADE",
                "s1_prob_attack": float(trace["stage1_prob_attack"][i]),
                "s1_passed_attack": bool(trace["stage1_pred_attack"][i]),
                "s2_cat": str(trace["stage2_pred_cat"][i]),
                "route": str(trace["specialist_route"][i]),
            })
        else:
            stage_info["mode"] = "E4_FLAT34"

        results.append({
            "true_34": true_raw,
            "true_op29": true_op29,
            "pred_34": pred_34,
            "pred_op29": pred_op29,
            "confidence": float(probs[pred_idx]),
            "top3": top3,
            "exact_34_match": (true_raw == pred_34),
            "op29_match": (true_op29 == pred_op29),
            "prep_us": round(per_flow_prep_us, 1),
            "inf_us": round(per_flow_inf_us, 1),
            "stage": stage_info,
            "action": map_verdict_playbook(pred_34, pred_op29),
            "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
        })

    return results


def infer_single_flow(
    row: pd.Series,
    arch_key: str,
    resources: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs timed single-row end-to-end preprocessing + inference for the live inspector."""
    df_one = pd.DataFrame([row[FEATURE_COLUMNS + [TARGET_COLUMN]]])
    return infer_batch(df_one, arch_key=arch_key, resources=resources)[0]


@st.cache_resource
def get_cached_resources() -> Dict[str, Any]:
    return load_soc_resources()


@st.cache_data
def get_cached_traffic() -> Tuple[pd.DataFrame, str]:
    return load_evaluation_traffic()


@st.cache_data
def load_benchmark_tables() -> Dict[str, Optional[pd.DataFrame]]:
    """Loads verified Track 2 presentation CSVs read-only for the Architecture Specs tab."""
    paths = {
        "final_test": HIER_RESULTS_DIR / "final_comparison_test.csv",
        "e2_fast": HIER_RESULTS_DIR / "e2_fast_lgbm_evaluation.csv",
        "merged": HIER_RESULTS_DIR / "merged_flood_e4_summary.csv",
    }
    return {k: (pd.read_csv(p) if p.exists() else None) for k, p in paths.items()}


def _sample_campaign_df(data_df: pd.DataFrame, campaign_choice: str, pkt_count: int, seed: int) -> pd.DataFrame:
    """Samples multi-stage scenario traffic cleanly without assuming missing label variants."""
    rng = np.random.default_rng(seed)

    def _draw(mask: pd.Series, n: int) -> pd.DataFrame:
        sub = data_df[mask]
        if sub.empty or n <= 0:
            return data_df.iloc[0:0]
        return sub.sample(n=n, replace=(len(sub) < n), random_state=int(rng.integers(1, 100000)))

    lbl = data_df["LabelUpper"]
    if "APT Multi-Stage" in campaign_choice:
        n_flood = int(pkt_count * 0.40)
        n_bf = max(2, int(pkt_count * 0.15))
        n_web = max(2, int(pkt_count * 0.15))
        n_mirai = max(2, int(pkt_count * 0.10))
        n_benign = max(1, pkt_count - (n_flood + n_bf + n_web + n_mirai))
        parts = [
            _draw(lbl.str.startswith(("DDOS-", "DOS-")), n_flood),
            _draw(lbl == "DICTIONARYBRUTEFORCE", n_bf),
            _draw(lbl.isin(["SQLINJECTION", "COMMANDINJECTION", "XSS", "BACKDOOR_MALWARE", "UPLOADING_ATTACK"]), n_web),
            _draw(lbl.str.startswith("MIRAI-"), n_mirai),
            _draw(lbl == "BENIGN", n_benign),
        ]
    elif "Volumetric" in campaign_choice:
        n_flood = int(pkt_count * 0.90)
        n_benign = max(1, pkt_count - n_flood)
        parts = [
            _draw(lbl.str.startswith(("DDOS-", "DOS-")), n_flood),
            _draw(lbl == "BENIGN", n_benign),
        ]
    elif "Stealthy Web" in campaign_choice:
        n_web = int(pkt_count * 0.60)
        n_bf = int(pkt_count * 0.20)
        n_benign = max(1, pkt_count - (n_web + n_bf))
        parts = [
            _draw(lbl.isin(["SQLINJECTION", "COMMANDINJECTION", "XSS", "BACKDOOR_MALWARE", "UPLOADING_ATTACK"]), n_web),
            _draw(lbl == "DICTIONARYBRUTEFORCE", n_bf),
            _draw(lbl == "BENIGN", n_benign),
        ]
    else:
        n_mirai = int(pkt_count * 0.60)
        n_recon = int(pkt_count * 0.20)
        n_benign = max(1, pkt_count - (n_mirai + n_recon))
        parts = [
            _draw(lbl.str.startswith("MIRAI-"), n_mirai),
            _draw(lbl.str.contains("RECON|VULNERABILITYSCAN", na=False), n_recon),
            _draw(lbl == "BENIGN", n_benign),
        ]

    combined = pd.concat(parts, ignore_index=True)
    return combined.sample(frac=1.0, random_state=int(rng.integers(1, 100000))).reset_index(drop=True)


def run_ui() -> None:
    st.set_page_config(
        page_title="Archon Track 2 IDS - Security Operations Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    resources = get_cached_resources()
    data_df, data_source_desc = get_cached_traffic()
    bench_tables = load_benchmark_tables()

    # ----------------- Sidebar Controls -----------------
    with st.sidebar:
        st.markdown("### Inference Engine Control")
        arch_key = st.radio(
            "Select Active IDS Model:",
            options=[ARCH_E4_KEY, ARCH_E2_KEY],
            format_func=lambda k: ARCH_DISPLAY_OPTIONS[k],
            index=0,
        )
        show_op29 = st.checkbox(
            "Enable 29-Class Operational Mitigation Grouping (Flood Twins)",
            value=False,
            help=(
                "Groups mitigation-equivalent IP-less protocol twins (SYN+SynIP, TCP, UDP, HTTP). "
                "Distinct from the 34-class academic benchmark."
            ),
        )
        st.divider()
        if arch_key == ARCH_E4_KEY:
            st.info(
                "**E4 (`E4_LGBM_Leaves63_Sqrt`)**\n"
                "- Role: **Primary Recommended IDS**\n"
                "- Executions / Flow: `1.0` single pass\n"
                "- Internal Test (34-cls): **75.37% Acc**, `0.6352` Macro-F1, **`0.7440` MCC**\n"
                "- Test P50 Latency: **`4.15 ms`** | Disk: `33.90 MB`"
            )
        else:
            st.info(
                "**E2 Fast (`E2_Cascade_ValTuned_LGBM`)**\n"
                "- Role: **Comparative 3-Stage Hierarchy**\n"
                "- Stage 1: Track 1 `binary_lightgbm` (`tau_att=0.15`)\n"
                "- Stage 2: 7-Class Attack Router (`f_tau=0.40`)\n"
                "- Internal Test (34-cls): `75.13%` Acc, **`0.6357`** Macro-F1, `0.7416` MCC\n"
                "- Test P50 Latency: **`8.44 ms`** *(6.31x faster vs legacy 53.30 ms RF)*"
            )
        st.caption(f"Traffic Pool: {data_source_desc}")
        st.caption("Preprocessor: Track 1 `preprocessor_strategy1.joblib` (39 features)")

    # ----------------- Header Banner -----------------
    active_badge = "PRIMARY: E4 FLAT L63" if arch_key == ARCH_E4_KEY else "COMPARATIVE: E2 FAST CASCADE"
    st.markdown(
        f"""
        <div class="header-card">
            <div class="header-title">
                ARCHON GATEWAY DEFENSE CONSOLE — TRACK 2 IDS
                <span class="header-status">{active_badge}</span>
            </div>
            <p class="header-sub">
                CICIoT2023 Real-Time Inference Engine | Track 1 Leak-Free Strategy 1 Preprocessing (39 Features) |
                34-Class Benchmark + Operational Mitigation Mapping
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_live, tab_single, tab_batch, tab_specs = st.tabs([
        "Live Attack Campaign Replay",
        "Targeted Flow Inspector",
        "Batch Flow Telemetry",
        "Architecture & Benchmark Specifications",
    ])

    # ==============================================================================
    # TAB 1: LIVE ATTACK CAMPAIGN REPLAY
    # ==============================================================================
    with tab_live:
        st.markdown("### Live Ingress Campaign Replay")
        st.caption("Stream unseen CICIoT2023 Internal Test flows through Track 1 preprocessing and the selected Track 2 IDS engine.")

        with st.container(border=True):
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                campaign_choice = st.selectbox(
                    "Campaign Attack Profile:",
                    [
                        "APT Multi-Stage Attack (Volumetric Flood + Brute-Force + Web Exploits + Mirai + Benign)",
                        "Volumetric Flood Tsunami (SYN, SynonymousIP, TCP, UDP, and HTTP Floods)",
                        "Stealthy Web Application Exploits (SQLi, Command Injection, XSS, Backdoor)",
                        "Mirai Botnet & Reconnaissance Stream",
                    ],
                )
            with col_c2:
                pkt_count = st.slider("Ingress Flow Count:", min_value=15, max_value=60, value=25, step=5)

            run_btn = st.button("Start Ingress Simulation", type="primary", width="stretch")

        if run_btn:
            combined = _sample_campaign_df(data_df, campaign_choice, pkt_count, seed=random.randint(1, 999999))
            kpi_holder = st.empty()
            feed_holder = st.empty()

            history: List[Dict[str, Any]] = []
            c_benign = c_flood = c_botnet = c_exploit = c_34_ok = c_29_ok = 0

            for idx, (_, row) in enumerate(combined.iterrows()):
                res = infer_single_flow(row, arch_key=arch_key, resources=resources)
                history.append(res)

                verdict = res["action"]["verdict"]
                if verdict == "BENIGN_PASSED":
                    c_benign += 1
                elif verdict == "FLOOD_DROPPED_EDGE":
                    c_flood += 1
                elif verdict == "BOTNET_QUARANTINED":
                    c_botnet += 1
                else:
                    c_exploit += 1

                if res["exact_34_match"]:
                    c_34_ok += 1
                if res["op29_match"]:
                    c_29_ok += 1

                total_seen = idx + 1
                with kpi_holder.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Flows Processed", f"{total_seen} / {len(combined)}")
                    m2.metric("Flood Drops", f"{c_flood}")
                    m3.metric("Botnet / Exploit / Recon", f"{c_botnet + c_exploit}")
                    m4.metric("Benign Passed", f"{c_benign}")
                    if show_op29:
                        m5.metric("29-Class Mitigation Match", f"{(c_29_ok / total_seen) * 100:.1f}%", f"34-cls exact: {(c_34_ok / total_seen) * 100:.1f}%")
                    else:
                        m5.metric("Exact 34-Class Match", f"{(c_34_ok / total_seen) * 100:.1f}%")

                act = res["action"]
                display_pred = res["pred_op29"] if show_op29 else res["pred_34"]
                stage_tag = (
                    f"Single-Pass Flat34 | Conf: {res['confidence']*100:.1f}%"
                    if res["stage"]["mode"] == "E4_FLAT34"
                    else f"S1 Att: {res['stage']['s1_prob_attack']*100:.1f}% -> {res['stage']['route']}"
                )
                with feed_holder.container():
                    st.markdown(
                        f"""
                        <div style="background: #0f172a; border: 1px solid #1e293b; border-left: 5px solid {act['color']}; padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                                <div>
                                    <span class="badge" style="background-color: {act['color']};">{act['verdict']}</span>
                                    <strong style="color: #f1f5f9; margin-left: 10px; font-size: 14px;">
                                        Flow #{total_seen}: True={res['true_34']} | Pred={display_pred}
                                    </strong>
                                </div>
                                <div style="color: #64748b; font-size: 12px; font-family: monospace;">
                                    {stage_tag} | Infer: {res['inf_us']:.0f} us
                                </div>
                            </div>
                            <div style="color: #94a3b8; font-size: 13px; margin-top: 6px;">
                                <strong>Mitigation:</strong> {act['action']} (34-Class: {res['pred_34']} | Op Group: {res['pred_op29']})
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                time.sleep(0.04)

            st.markdown("#### Incident Telemetry & Defense Log")
            log_rows = []
            for i, h in enumerate(history):
                stg = h["stage"]
                route_desc = (
                    f"Single-Pass ({stg['executions']}x)"
                    if stg["mode"] == "E4_FLAT34"
                    else f"S1={stg['s1_prob_attack']:.2f} -> {stg['s2_cat']} -> {stg['route']}"
                )
                log_rows.append({
                    "Seq": i + 1,
                    "True Label (34-Cls)": h["true_34"],
                    "Predicted (34-Cls)": h["pred_34"],
                    "Op Mitigation Class (29-Cls)": h["pred_op29"],
                    "Confidence": f"{h['confidence']*100:.1f}%",
                    "Pipeline Trace": route_desc,
                    "Inf Latency (us)": h["inf_us"],
                    "Gateway Action": h["action"]["verdict"],
                    "Mitigation Playbook": h["action"]["action"],
                })
            st.dataframe(pd.DataFrame(log_rows), width="stretch")

            st.markdown(
                f"""
                <div style="background: #0f172a; border: 1px solid #334155; padding: 14px 18px; border-radius: 8px; margin-top: 10px;">
                    <h4 style="color: #38bdf8; margin: 0 0 6px 0; font-size: 15px;">Replay Diagnostics</h4>
                    <p style="color: #cbd5e1; font-size: 13.5px; margin: 0; line-height: 1.55;">
                        Exact 34-class attribution matched <strong>{c_34_ok}/{len(history)} ({(c_34_ok/len(history))*100:.1f}%)</strong> flows.
                        Under the 29-class IP-less Flood-twin mitigation taxonomy (merging SYN+SynonymousIP, TCP, UDP, and HTTP protocol twins),
                        operational remediation aligned on <strong>{c_29_ok}/{len(history)} ({(c_29_ok/len(history))*100:.1f}%)</strong> flows.
                        Notice minority Web/Recon classes remain inherently challenging (~0.22 Minority F1 on Test) due to overlapping 39-feature representations without packet payloads.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==============================================================================
    # TAB 2: TARGETED FLOW INSPECTOR
    # ==============================================================================
    with tab_single:
        st.markdown("### Targeted Flow Inspector")
        st.caption("Inspect an individual unseen flow across Track 1 preprocessing, model probability outputs, and automated gateway mitigation.")

        with st.container(border=True):
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                target_choice = st.selectbox(
                    "Select Ground-Truth Class to Sample from Internal Test:",
                    [
                        "BENIGN (Normal IoT Sensor Telemetry)",
                        "DDOS-SYN_FLOOD (Flood Twin Test: Often swaps w/ DOS-SYN & SYNONYMOUSIP)",
                        "DOS-SYN_FLOOD (Flood Twin Test: Protocol-identical TCP SYN)",
                        "DDOS-TCP_FLOOD (Flood Twin Test: TCP Volumetric)",
                        "DDOS-UDP_FLOOD (Flood Twin Test: UDP Volumetric)",
                        "DDOS-HTTP_FLOOD (Flood Twin Test: HTTP Volumetric)",
                        "SQLINJECTION (Minority Web Exploit)",
                        "COMMANDINJECTION (Minority Web Exploit)",
                        "DICTIONARYBRUTEFORCE (Credential Brute Force)",
                        "MIRAI-GREETH_FLOOD (IoT Botnet Traffic)",
                        "RECON-PORTSCAN (Network Reconnaissance)",
                        "DNS_SPOOFING (DNS Cache Poisoning)",
                        "BROWSERHIJACKING (Web Hijack Attack)",
                    ],
                )
            with col_t2:
                st.write("")
                st.write("")
                resample_btn = st.button("Sample & Inspect Flow", type="primary", width="stretch")

        if "inspector_seed" not in st.session_state:
            st.session_state["inspector_seed"] = 42
        if resample_btn:
            st.session_state["inspector_seed"] = random.randint(1, 1000000)

        tag_clean = target_choice.split(" ")[0].strip().upper()
        matches = data_df[data_df["LabelUpper"] == tag_clean]
        selected_row = (
            matches.sample(1, random_state=st.session_state["inspector_seed"]).iloc[0]
            if not matches.empty
            else data_df.iloc[0]
        )

        res = infer_single_flow(selected_row, arch_key=arch_key, resources=resources)
        stg = res["stage"]

        col1, col2, col3 = st.columns(3)
        if stg["mode"] == "E4_FLAT34":
            with col1:
                st.markdown("#### Pass 1: Preprocessing + Flat E4")
                st.caption("Single-pass LightGBM (`num_leaves=63, sqrt_balanced`)")
                st.write(f"Benign Probability: **{stg['benign_prob']*100:.2f}%**")
                st.write(f"Volumetric Flood Sum: **{stg['flood_prob']*100:.2f}%**")
                st.write(f"Non-Flood Attack Sum: **{stg['non_flood_prob']*100:.2f}%**")
                st.write(f"Executions: **{stg['executions']} model pass** ({res['inf_us']} us)")
            with col2:
                st.markdown("#### Pass 2: Top-3 34-Class Attribution")
                st.caption("Fine-grained 34-class posterior ranking")
                for rank_i, (cname, cprob) in enumerate(res["top3"], start=1):
                    st.write(f"#{rank_i} **{cname}**: `{cprob*100:.2f}%`")
                st.write(f"Preprocess Time: **{res['prep_us']} us**")
            with col3:
                st.markdown("#### Pass 3: Operational Policy View")
                st.caption("Original 34-class vs. 29-class Flood group")
                st.write(f"Ground Truth (34-Cls): **{res['true_34']}**")
                st.write(f"Predicted (34-Cls): **{res['pred_34']}** ({'MATCH' if res['exact_34_match'] else 'DIFF'})")
                st.write(f"Operational Group (29-Cls): **{res['pred_op29']}** ({'MATCH' if res['op29_match'] else 'DIFF'})")
        else:
            with col1:
                st.markdown("#### Stage 1: Binary LightGBM Filter")
                st.caption("Track 1 `binary_lightgbm.joblib` (`tau_att = 0.15`)")
                st.write(f"P(Attack): **{stg['s1_prob_attack']*100:.2f}%**")
                st.write(f"Gate Decision: **{'ESCALATED TO S2' if stg['s1_passed_attack'] else 'EXIT BENIGN'}**")
                st.write(f"Total Pipeline Inf: **{res['inf_us']} us** ({stg['executions']} execs)")
            with col2:
                st.markdown("#### Stage 2: 7-Category Attack Router")
                st.caption("Malicious-only LightGBM (`f_tau = 0.40`)")
                st.write(f"Stage 2 Category: **{stg['s2_cat']}**")
                st.write(f"Routed Branch: **{stg['route']}**")
                st.write(f"Flood Group Sum: **{stg['flood_prob']*100:.2f}%**")
            with col3:
                st.markdown("#### Stage 3: Specialist Fine Attribution")
                st.caption("Stage 3A (16 Flood) / Stage 3B (17 Non-Flood)")
                for rank_i, (cname, cprob) in enumerate(res["top3"], start=1):
                    st.write(f"#{rank_i} **{cname}**: `{cprob*100:.2f}%`")
                st.write(f"Op Group: **{res['pred_op29']}** (True: {res['true_34']})")

        st.divider()
        act = res["action"]
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-left: 5px solid {act['color']}; padding: 16px 20px; border-radius: 8px;">
                <h4 style="color: {act['color']}; margin: 0 0 8px 0;">Gateway Decision: {act['verdict']}</h4>
                <p style="color: #f1f5f9; margin: 0; font-size: 14.5px;">
                    <strong>34-Class Prediction:</strong> {res['pred_34']} ({res['confidence']*100:.2f}%) |
                    <strong>29-Class Mitigation Group:</strong> {res['pred_op29']} |
                    <strong>Ground Truth:</strong> {res['true_34']}
                </p>
                <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 14px;">
                    <strong>Playbook Action ({act['severity']}):</strong> {act['action']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Examine Exact 39 CICIoT2023 Input Features (`FEATURE_COLUMNS`)"):
            feat_df = pd.DataFrame([
                {"Feature": c, "Raw Value": round(float(res["features"][c]), 5)}
                for c in FEATURE_COLUMNS
            ])
            st.dataframe(feat_df, width="stretch")

    # ==============================================================================
    # TAB 3: BATCH FLOW TELEMETRY
    # ==============================================================================
    with tab_batch:
        st.markdown("### Batch Flow Telemetry")
        st.caption("Randomly sample unseen Internal Test flows across all 34 classes and profile vectorized batch execution.")

        with st.container(border=True):
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                b_size = st.slider("Select Batch Size:", min_value=10, max_value=100, value=25, step=5)
            with col_b2:
                st.write("")
                st.write("")
                run_batch_btn = st.button("Execute New Batch Sample", type="primary", width="stretch")

        if "batch_seed" not in st.session_state:
            st.session_state["batch_seed"] = 101
        if run_batch_btn:
            st.session_state["batch_seed"] = random.randint(1, 1000000)

        sample_batch = data_df.sample(n=min(b_size, len(data_df)), random_state=st.session_state["batch_seed"]).reset_index(drop=True)
        batch_res = infer_batch(sample_batch, arch_key=arch_key, resources=resources)

        acc_34 = np.mean([r["exact_34_match"] for r in batch_res]) * 100.0
        acc_29 = np.mean([r["op29_match"] for r in batch_res]) * 100.0
        mean_inf_us = float(np.mean([r["inf_us"] for r in batch_res]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Batch Flows Evaluated", len(batch_res))
        m2.metric("Exact 34-Class Match", f"{acc_34:.1f}%")
        m3.metric("29-Class Operational Match", f"{acc_29:.1f}%", "Flood twins unified")
        m4.metric("Mean Amortized Infer Time", f"{mean_inf_us:.1f} us/flow")

        df_batch_view = pd.DataFrame([
            {
                "True Label": r["true_34"],
                "Pred (34-Cls)": r["pred_34"],
                "Match (34)": "YES" if r["exact_34_match"] else "NO",
                "Op Mitigation Group": r["pred_op29"],
                "Match (Op 29)": "YES" if r["op29_match"] else "NO",
                "Conf": f"{r['confidence']*100:.1f}%",
                "Action": r["action"]["verdict"],
            }
            for r in batch_res
        ])
        st.dataframe(df_batch_view, width="stretch")

    # ==============================================================================
    # TAB 4: ARCHITECTURE & BENCHMARK SPECIFICATIONS
    # ==============================================================================
    with tab_specs:
        st.markdown("### Track 2 Verified Experimental Results & Deployment Trade-Offs")
        st.caption(
            "All figures reflect strict Track 1 Strategy 1 zero-overlap evaluation on Internal Test (N = 75,105) "
            "and Cross-Capture Clean (N = 57,280). Operational 29-class numbers are a separate mitigation view — not a gain on the 34-class task."
        )

        df_test_orig = bench_tables.get("final_test")
        df_e2_fast = bench_tables.get("e2_fast")
        df_merged = bench_tables.get("merged")

        if df_test_orig is not None:
            st.markdown("#### 1. Original 34-Class Internal Test Benchmark (`N = 75,105`, Including Updated Fast E2)")
            cols_keep = [
                "architecture", "accuracy", "macro_f1", "mcc",
                "flood_macro_f1", "non_flood_macro_f1", "minority_macro_recall",
                "benign_recall", "single_latency_p50_us", "batch_throughput_fps", "model_size_mb",
            ]
            combined_rows = df_test_orig[cols_keep].copy()
            if df_e2_fast is not None:
                e2_row = df_e2_fast[df_e2_fast["split"] == "test"][cols_keep]
                combined_rows = pd.concat([combined_rows, e2_row], ignore_index=True)
            st.dataframe(combined_rows, width="stretch")

        if df_merged is not None:
            st.markdown("#### 2. E4 Original 34-Class vs. Operational 29-Class Flood-Merged View (`N = 75,105` Test)")
            st.dataframe(
                df_merged[df_merged["split"] == "test"][[
                    "taxonomy", "inference_mode", "n_classes", "accuracy", "macro_f1",
                    "mcc", "flood_macro_f1", "non_flood_macro_f1", "minority_macro_f1",
                ]],
                width="stretch",
            )

        fig_tradeoff = HIER_RESULTS_DIR / "figures" / "e1_e4_architecture_tradeoffs.png"
        fig_merged = HIER_RESULTS_DIR / "figures" / "e4_34class_vs_merged_flood.png"
        c_fig1, c_fig2 = st.columns(2)
        if fig_tradeoff.exists():
            with c_fig1:
                st.image(str(fig_tradeoff), caption="E1-E4 Performance & E2 Latency Fix (53.30 ms -> 8.44 ms)")
        if fig_merged.exists():
            with c_fig2:
                st.image(str(fig_merged), caption="Distinct Tasks: Standard 34-Class vs. 29-Class Operational Mitigation")

        st.markdown(
            """
            <div class="control-card">
                <h4 style="color: #38bdf8; margin: 0 0 8px 0;">Deployment Trade-Offs & Dataset Limitations</h4>
                <ul style="color: #cbd5e1; font-size: 13.5px; line-height: 1.65; margin: 0; padding-left: 20px;">
                    <li><strong>Primary Recommendation (E4):</strong> Achieves top Internal Test Accuracy (<code>75.37%</code>) and MCC (<code>0.7440</code>) at <code>4.15 ms</code> single-flow P50 with 1 model execution, proving capacity + <code>sqrt_balanced</code> weighting match 3-stage cascades without pipeline complexity.</li>
                    <li><strong>Comparative Fast Hierarchy (E2 LGBM):</strong> Replacing Track 1's <code>n_jobs=-1</code> Random Forest Stage 1 with <code>binary_lightgbm.joblib</code> cuts Test P50 latency by <strong>6.31x</strong> (<code>53.30 ms -> 8.44 ms</code>) while matching or improving Test Accuracy (<code>75.13%</code>) and MCC (<code>0.7416</code>). Use when Flood vs. Non-Flood policy stages must be tuned or swapped independently.</li>
                    <li><strong>IP-Stripped Dataset Limit (7,840 Flood Twin Errors):</strong> Raw CICIoT2023 CSVs export 39 packet/flow statistics with zero IP, port, or attacker-cardinality columns. Four protocol twins (<code>SYN/SynonymousIP</code>, <code>TCP</code>, <code>UDP</code>, <code>HTTP</code>) account for <strong>42.4%</strong> of E4's Test errors while sharing identical protocol remediation.</li>
                    <li><strong>Minority & Generalization Limits:</strong> Rare Web/Recon attacks (~2.58% of Test) plateau around ~0.22 Minority Macro-F1 across all 39-feature tabular models; Cross-Capture Holdout (<code>Merged26-40</code>) comes from the same CIC lab testbed with 5.67x capped minority prevalence.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run_ui()
