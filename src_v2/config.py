"""
Configuration and metadata for the revised (v2) CICIoT2023 Intrusion Detection Pipeline.
Preserves original settings while introducing leak-free partition paths, revised models,
and separate output directories.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "MERGED_CSV"
DATA_DIR = PROJECT_ROOT / "data"

# Revised Pipeline Specific Paths (preserves original models_saved and results)
PARTITIONS_DIR = DATA_DIR / "partitions"
MODELS_V2_DIR = PROJECT_ROOT / "models_saved_v2"
RESULTS_V2_DIR = PROJECT_ROOT / "results_v2"
FIGURES_V2_DIR = RESULTS_V2_DIR / "figures"
TABLES_V2_DIR = RESULTS_V2_DIR / "tables"

# Ensure output directories exist
for p in [PARTITIONS_DIR, MODELS_V2_DIR, RESULTS_V2_DIR, FIGURES_V2_DIR, TABLES_V2_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Original dataset file (strictly read-only)
ORIGINAL_SAMPLE_FILE = DATA_DIR / "sample_stratified.csv"

# Partition assignment file and metadata
PARTITION_ASSIGNMENTS_FILE = PARTITIONS_DIR / "partition_assignments.csv"
PARTITION_METADATA_FILE = PARTITIONS_DIR / "partition_metadata.json"

# Verified 39 input feature columns
FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

TARGET_COLUMN = "Label"

# Label Mappings (Binary)
LABEL_MAPPING_BINARY = {
    "BENIGN": "Benign",
    "DDOS-RSTFINFLOOD": "Attack",
    "DDOS-PSHACK_FLOOD": "Attack",
    "DDOS-SYN_FLOOD": "Attack",
    "DDOS-UDP_FLOOD": "Attack",
    "DDOS-TCP_FLOOD": "Attack",
    "DDOS-ICMP_FLOOD": "Attack",
    "DDOS-SYNONYMOUSIP_FLOOD": "Attack",
    "DDOS-ACK_FRAGMENTATION": "Attack",
    "DDOS-UDP_FRAGMENTATION": "Attack",
    "DDOS-ICMP_FRAGMENTATION": "Attack",
    "DDOS-SLOWLORIS": "Attack",
    "DDOS-HTTP_FLOOD": "Attack",
    "DOS-UDP_FLOOD": "Attack",
    "DOS-SYN_FLOOD": "Attack",
    "DOS-TCP_FLOOD": "Attack",
    "DOS-HTTP_FLOOD": "Attack",
    "MIRAI-GREETH_FLOOD": "Attack",
    "MIRAI-GREIP_FLOOD": "Attack",
    "MIRAI-UDPPLAIN": "Attack",
    "RECON-PINGSWEEP": "Attack",
    "RECON-OSSCAN": "Attack",
    "RECON-PORTSCAN": "Attack",
    "VULNERABILITYSCAN": "Attack",
    "RECON-HOSTDISCOVERY": "Attack",
    "DNS_SPOOFING": "Attack",
    "MITM-ARPSPOOFING": "Attack",
    "BROWSERHIJACKING": "Attack",
    "BACKDOOR_MALWARE": "Attack",
    "XSS": "Attack",
    "UPLOADING_ATTACK": "Attack",
    "SQLINJECTION": "Attack",
    "COMMANDINJECTION": "Attack",
    "DICTIONARYBRUTEFORCE": "Attack",
}

BINARY_CLASSES = ["Benign", "Attack"]
BINARY_TO_IDX = {"Benign": 0, "Attack": 1}
IDX_TO_BINARY = {0: "Benign", 1: "Attack"}

# Category Mapping (8 Classes)
LABEL_MAPPING_8CLASSES = {
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DOS-UDP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "RECON-PINGSWEEP": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "RECON-HOSTDISCOVERY": "Recon",
    "DNS_SPOOFING": "Spoofing",
    "MITM-ARPSPOOFING": "Spoofing",
    "BROWSERHIJACKING": "Web",
    "BACKDOOR_MALWARE": "Web",
    "XSS": "Web",
    "UPLOADING_ATTACK": "Web",
    "SQLINJECTION": "Web",
    "COMMANDINJECTION": "Web",
    "DICTIONARYBRUTEFORCE": "BruteForce",
    "BENIGN": "Benign",
}

EIGHT_CLASSES = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
    "BruteForce",
]
EIGHT_TO_IDX = {cls_name: i for i, cls_name in enumerate(EIGHT_CLASSES)}
IDX_TO_EIGHT = {i: cls_name for i, cls_name in enumerate(EIGHT_CLASSES)}

# 34-Class Mapping
THIRTY_FOUR_CLASSES = sorted(list(LABEL_MAPPING_BINARY.keys()))
THIRTY_FOUR_TO_IDX = {cls_name: i for i, cls_name in enumerate(THIRTY_FOUR_CLASSES)}
IDX_TO_THIRTY_FOUR = {i: cls_name for i, cls_name in enumerate(THIRTY_FOUR_CLASSES)}

# Functional Threat Tiers (Mutually exclusive grouping of all 33 attacks; Benign handled separately)
FUNCTIONAL_THREAT_TIERS = {
    "Tier 1: Volumetric Floods": [
        "DDOS-ACK_FRAGMENTATION", "DDOS-HTTP_FLOOD", "DDOS-ICMP_FLOOD",
        "DDOS-ICMP_FRAGMENTATION", "DDOS-PSHACK_FLOOD", "DDOS-RSTFINFLOOD",
        "DDOS-SLOWLORIS", "DDOS-SYNONYMOUSIP_FLOOD", "DDOS-SYN_FLOOD",
        "DDOS-TCP_FLOOD", "DDOS-UDP_FLOOD", "DDOS-UDP_FRAGMENTATION",
        "DOS-HTTP_FLOOD", "DOS-SYN_FLOOD", "DOS-TCP_FLOOD", "DOS-UDP_FLOOD",
    ],
    "Tier 2: IoT Botnets": [
        "MIRAI-GREETH_FLOOD", "MIRAI-GREIP_FLOOD", "MIRAI-UDPPLAIN",
    ],
    "Tier 3: Recon & Spoofing": [
        "RECON-HOSTDISCOVERY", "RECON-OSSCAN", "RECON-PINGSWEEP",
        "RECON-PORTSCAN", "VULNERABILITYSCAN", "DNS_SPOOFING", "MITM-ARPSPOOFING",
    ],
    "Tier 4: Web & Exploit Attacks": [
        "BACKDOOR_MALWARE", "BROWSERHIJACKING", "COMMANDINJECTION",
        "DICTIONARYBRUTEFORCE", "SQLINJECTION", "UPLOADING_ATTACK", "XSS",
    ],
}

# Reproducible random seed
RANDOM_SEED = 42

