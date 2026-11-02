import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Archon Edge IDS | Live 3-Tier Defense Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Color mappings
STATUS_COLORS = {
    "BENIGN_PASSED": "#10B981",       # Green
    "FLOOD_DROPPED_EDGE": "#EF4444",   # Red
    "BOTNET_QUARANTINED": "#F59E0B",  # Orange
    "EXPLOIT_ESCALATED": "#8B5CF6",   # Purple
}

# Label encoders and categories
LABEL_MAPPING_8CLASSES = {
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "BenignTraffic": "Benign",
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web",
    "CommandInjection": "Web", "SqlInjection": "Web", "Uploading_Attack": "Web",
    "DictionaryBruteForce": "BruteForce",
}

@st.cache_resource
def load_models():
    """Load pre-trained models and scalers for all 3 tiers."""
    t1_model = joblib.load(BASE_DIR / "models_saved" / "binary" / "logistic_regression.joblib")
    t1_scaler = joblib.load(BASE_DIR / "models_saved" / "binary" / "binary_scaler.joblib")
    
    t2_model = joblib.load(BASE_DIR / "models_saved" / "reduced_features" / "xgboost_top20.joblib")
    t2_scaler = joblib.load(BASE_DIR / "models_saved" / "reduced_features" / "reduced_20_scaler.joblib")
    top_20_feats = joblib.load(BASE_DIR / "models_saved" / "reduced_features" / "top_20_feature_names.joblib")
    
    t3_model = joblib.load(BASE_DIR / "models_saved" / "attack_34class" / "xgboost.joblib")
    t3_scaler = joblib.load(BASE_DIR / "models_saved" / "attack_34class" / "attack34_scaler.joblib")
    
    return {
        "t1_model": t1_model, "t1_scaler": t1_scaler,
        "t2_model": t2_model, "t2_scaler": t2_scaler, "top_20_feats": top_20_feats,
        "t3_model": t3_model, "t3_scaler": t3_scaler
    }

@st.cache_data
def load_sample_dataset():
    """Load stratified sample dataset to draw realistic traffic flows."""
    sample_file = BASE_DIR / "data" / "sample_stratified.csv"
    df = pd.read_csv(sample_file)
    return df

# Initialize models and data
models = load_models()
data_df = load_sample_dataset()

# Clean feature columns
feature_cols = [c for c in data_df.columns if c not in ["Label", "label"]]

# ----------------- UI Header -----------------
st.title("🛡️ Archon Multi-Tier IoT Edge IDS: Real-Time Software Demo")
st.markdown(
    """
    **Autonomous 3-Tier Network Intrusion Detection Engine** evaluated on **CICIoT2023**  
    *Tier 1 Wire-Speed Filter (0.20 µs) ➔ Tier 2 Edge Gateway Triage (12.15 µs, 20 Features) ➔ Tier 3 Cloud Forensic Attribution*
    """
)
st.divider()

# ----------------- Sidebar Controls -----------------
st.sidebar.header("⚙️ Simulation Controls")

sim_mode = st.sidebar.radio(
    "Traffic Ingestion Mode:",
    ["Simulate Targeted Attack / Packet", "Stream Random Network Flow Batch"]
)

if sim_mode == "Simulate Targeted Attack / Packet":
    target_options = [
        "BENIGN (Normal IoT Sensor Traffic)",
        "SQLINJECTION (Web Database Exploit)",
        "COMMANDINJECTION (Gateway Shell Exploit)",
        "DICTIONARYBRUTEFORCE (Telnet/SSH Credential Attack)",
        "MIRAI-GREETH_FLOOD (IoT Botnet Command & Control)",
        "DDOS-SYN_FLOOD (Volumetric Bandwidth Flood)",
        "RECON-PORTSCAN (Attacker Network Survey)",
        "DNS_SPOOFING (Cache Poisoning Attack)",
        "BROWSERHIJACKING (Web Portal Attack)",
    ]
    chosen_attack = st.sidebar.selectbox("Choose Traffic Flow Profile:", target_options)
    send_btn = st.sidebar.button("🚀 Ingest Flow Into Gateway", type="primary")
else:
    batch_size = st.sidebar.slider("Flow Batch Size:", min_value=5, max_value=30, value=10)
    stream_btn = st.sidebar.button("⚡ Run Live Batch Stream", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Hardware & Edge Profile")
st.sidebar.info(
    """
    - **Target Device**: ARM Cortex-A72 / Raspberry Pi 4
    - **Tier 1 Line-Rate**: 5,017,561 flows/sec
    - **Tier 2 Edge Memory**: 80 bytes/flow (48.7% RAM savings)
    - **Tier 2 Edge Latency**: 12.15 µs/flow (82,299 flows/sec)
    """
)

# ----------------- Processing Logic -----------------
def process_single_flow(row):
    """Processes a flow through Tier 1 -> Tier 2 -> Tier 3."""
    flow_features = row[feature_cols].to_dict()
    true_label = str(row["Label"]).strip()
    
    # 1. Tier 1: Wire-Speed Binary Screen (SGD Logistic Regression)
    x_full = np.array([row[feature_cols].values], dtype=np.float32)
    # Handle infinite/NaN
    x_full = np.nan_to_num(x_full, nan=0.0, posinf=3.35e6, neginf=0.0)
    
    t0 = time.perf_counter_ns()
    x_t1_scaled = models["t1_scaler"].transform(x_full)
    t1_pred = models["t1_model"].predict(x_t1_scaled)[0]
    t1_prob = models["t1_model"].predict_proba(x_t1_scaled)[0]
    t1_time_us = (time.perf_counter_ns() - t0) / 1000.0
    
    # t1_pred: 0 = Benign, 1 = Attack (or vice versa based on training)
    # Check probability of benign
    # If binary model classified as Benign (0) with high confidence:
    is_benign = (t1_pred == 0) and (t1_prob[0] > 0.85)
    
    tier1_result = {
        "prediction": "Normal (Benign)" if is_benign else "Suspicious (Anomaly)",
        "confidence": float(max(t1_prob)),
        "latency_us": round(t1_time_us, 2),
        "status": "PASSED" if is_benign else "ESCALATED",
    }
    
    tier2_result = None
    tier3_result = None
    final_action = None
    
    if is_benign:
        final_action = {
            "verdict": "BENIGN_PASSED",
            "tier_resolved": "Tier 1 (Border Wire-Filter)",
            "action": "Immediate Egress to IoT LAN",
            "threat_level": "None",
            "color": STATUS_COLORS["BENIGN_PASSED"]
        }
    else:
        # 2. Tier 2: Edge Gateway Triage (20-Feature Reduced XGBoost)
        top_20 = models["top_20_feats"]
        x_top20 = np.array([row[top_20].values], dtype=np.float32)
        x_top20 = np.nan_to_num(x_top20, nan=0.0, posinf=3.35e6, neginf=0.0)
        
        t0 = time.perf_counter_ns()
        x_t2_scaled = models["t2_scaler"].transform(x_top20)
        t2_pred_idx = models["t2_model"].predict(x_t2_scaled)[0]
        t2_time_us = (time.perf_counter_ns() - t0) / 1000.0
        
        # Check class index or label
        tier2_label = str(t2_pred_idx)
        
        tier2_result = {
            "category": tier2_label,
            "features_used": 20,
            "latency_us": round(t2_time_us, 2),
            "ram_savings": "48.7%",
        }
        
        # Action routing
        if "DDoS" in tier2_label or "DoS" in tier2_label or "FLOOD" in tier2_label.upper():
            final_action = {
                "verdict": "FLOOD_DROPPED_EDGE",
                "tier_resolved": "Tier 2 (Edge Gateway)",
                "action": "Dropped at Gateway Boundary (Bandwidth Preserved)",
                "threat_level": "High (Volumetric Flood)",
                "color": STATUS_COLORS["FLOOD_DROPPED_EDGE"]
            }
        elif "Mirai" in tier2_label or "Spoof" in tier2_label:
            final_action = {
                "verdict": "BOTNET_QUARANTINED",
                "tier_resolved": "Tier 2 (Edge Gateway)",
                "action": "Dynamic MAC Isolation & Port Quarantine",
                "threat_level": "Critical (Compromised Node)",
                "color": STATUS_COLORS["BOTNET_QUARANTINED"]
            }
        else:
            # 3. Tier 3: Cloud Forensic Attribution (34-Class Full XGBoost)
            t0 = time.perf_counter_ns()
            x_t3_scaled = models["t3_scaler"].transform(x_full)
            t3_pred = models["t3_model"].predict(x_t3_scaled)[0]
            t3_time_us = (time.perf_counter_ns() - t0) / 1000.0
            
            tier3_result = {
                "attributed_attack": str(t3_pred),
                "features_used": 39,
                "latency_us": round(t3_time_us, 2),
                "mitigation_rule": "Auto-Generated WAF Regex & Upstream BGP Null-Route"
            }
            
            final_action = {
                "verdict": "EXPLOIT_ESCALATED",
                "tier_resolved": "Tier 3 (Cloud SIEM / SOC)",
                "action": f"Forensic Attribution: {t3_pred} ➔ Pushed WAF Blocking Rule",
                "threat_level": "Severe (Stealth Application Exploit)",
                "color": STATUS_COLORS["EXPLOIT_ESCALATED"]
            }
            
    return {
        "true_label": true_label,
        "features": flow_features,
        "tier1": tier1_result,
        "tier2": tier2_result,
        "tier3": tier3_result,
        "final_action": final_action
    }

# ----------------- Execution Display -----------------

if sim_mode == "Simulate Targeted Attack / Packet":
    # Find matching row from dataset
    tag_clean = chosen_attack.split(" ")[0].strip()
    matches = data_df[data_df["Label"].str.upper() == tag_clean.upper()]
    if matches.empty:
        matches = data_df[data_df["Label"].str.contains(tag_clean, case=False, na=False)]
    if matches.empty:
        selected_row = data_df.sample(1, random_state=42).iloc[0]
    else:
        selected_row = matches.sample(1, random_state=random.randint(1, 10000)).iloc[0]
        
    st.subheader(f"Analyzing Ingress Flow Profile: `{selected_row['Label']}`")
    
    # Run pipeline
    res = process_single_flow(selected_row)
    
    # 3-Tier Swimlane Visualizer
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟦 Tier 1: Wire-Filter")
        st.caption("Border Switch Ingress (10 Gbps Wire-Speed)")
        t1 = res["tier1"]
        if t1["status"] == "PASSED":
            st.success(f"**Status: {t1['prediction']}**")
        else:
            st.warning(f"**Status: {t1['prediction']}**")
        st.write(f"⏱️ **Latency**: `{t1['latency_us']} µs`")
        st.write(f"🎯 **Confidence**: `{t1['confidence']*100:.1f}%`")
        st.write(f"📋 **Next Stage**: `{t1['status']}`")
        
    with col2:
        st.markdown("### 🟨 Tier 2: Edge Gateway")
        st.caption("Embedded Router (Top 20 Features)")
        if res["tier2"]:
            t2 = res["tier2"]
            st.info(f"**Category: {t2['category']}**")
            st.write(f"⏱️ **Latency**: `{t2['latency_us']} µs`")
            st.write(f"💾 **Memory Cut**: `{t2['ram_savings']}`")
            st.write(f"⚡ **Throughput**: `82,299 flows/s`")
        else:
            st.markdown("*[Bypassed: Flow verified benign at Tier 1]*")
            
    with col3:
        st.markdown("### 🟪 Tier 3: Cloud Forensics")
        st.caption("Centralized SOC / SIEM (34-Class Full Attribution)")
        if res["tier3"]:
            t3 = res["tier3"]
            st.error(f"**Identified: {t3['attributed_attack']}**")
            st.write(f"⏱️ **Latency**: `{t3['latency_us']} µs`")
            st.write(f"🛡️ **Playbook**: `{t3['mitigation_rule']}`")
        else:
            st.markdown("*[Not Needed: Resolved at Edge Tier]*")
            
    st.divider()
    
    # Final Action Banner
    action = res["final_action"]
    st.markdown(
        f"""
        <div style="background-color: {action['color']}22; border-left: 6px solid {action['color']}; padding: 16px; border-radius: 8px;">
            <h3 style="color: {action['color']}; margin: 0 0 8px 0;">🛡️ Gateway Action: {action['verdict']}</h3>
            <p style="margin: 0; font-size: 16px;"><strong>Resolved At:</strong> {action['tier_resolved']} | <strong>Threat Severity:</strong> {action['threat_level']}</p>
            <p style="margin: 4px 0 0 0; font-size: 15px;"><strong>Automated Mitigation:</strong> {action['action']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Flow Vector Inspection Accordion
    with st.expander("🔍 Inspect Extracted 39 Flow Features (Vector Values)"):
        st.json({k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in selected_row[feature_cols].items()})

else:
    # Batch stream mode
    st.subheader(f"⚡ Live Packet Stream Simulation ({batch_size} Flows)")
    sample_batch = data_df.sample(batch_size, random_state=random.randint(1, 100000))
    
    records = []
    progress_bar = st.progress(0)
    
    for idx, (_, row) in enumerate(sample_batch.iterrows()):
        res = process_single_flow(row)
        records.append({
            "True Label": res["true_label"],
            "Tier 1 Status": res["tier1"]["status"],
            "Tier 1 Latency (µs)": res["tier1"]["latency_us"],
            "Resolved At": res["final_action"]["tier_resolved"],
            "Final Action": res["final_action"]["action"],
            "Threat Level": res["final_action"]["threat_level"],
        })
        progress_bar.progress((idx + 1) / batch_size)
        
    df_results = pd.DataFrame(records)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Flows Processed", f"{len(df_results)}")
    with col_m2:
        benign_count = (df_results["Tier 1 Status"] == "PASSED").sum()
        st.metric("Benign Line-Passed", f"{benign_count} flows", f"{(benign_count/batch_size)*100:.0f}%")
    with col_m3:
        edge_dropped = df_results["Resolved At"].str.contains("Tier 2").sum()
        st.metric("Edge Gateway Dropped", f"{edge_dropped} flows")
    with col_m4:
        cloud_escalated = df_results["Resolved At"].str.contains("Tier 3").sum()
        st.metric("Cloud SOC Attributed", f"{cloud_escalated} flows")
        
    st.dataframe(df_results, use_container_width=True)

