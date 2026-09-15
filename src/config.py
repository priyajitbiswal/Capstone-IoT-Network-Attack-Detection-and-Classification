"""
Configuration and metadata for CICIoT2023 Intrusion Detection Pipeline.
Contains dataset paths, verified 39 input features, and label mappings for 2, 8, and 34 classes.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "MERGED_CSV"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Ensure output directories exist
for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Sample file path for stratified subset
SAMPLE_FILE = DATA_DIR / "sample_stratified.csv"

# Verified 39 input feature columns from the CICIoT2023 MERGED_CSV files
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

# Binary Mapping (Phase 1: 2 Classes)
# Maps each of the 34 raw uppercase labels to either 'Attack' or 'Benign'
LABEL_MAPPING_BINARY = {
    # Benign
    "BENIGN": "Benign",
    # DDoS
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
    # DoS
    "DOS-UDP_FLOOD": "Attack",
    "DOS-SYN_FLOOD": "Attack",
    "DOS-TCP_FLOOD": "Attack",
    "DOS-HTTP_FLOOD": "Attack",
    # Mirai
    "MIRAI-GREETH_FLOOD": "Attack",
    "MIRAI-GREIP_FLOOD": "Attack",
    "MIRAI-UDPPLAIN": "Attack",
    # Reconnaissance
    "RECON-PINGSWEEP": "Attack",
    "RECON-OSSCAN": "Attack",
    "RECON-PORTSCAN": "Attack",
    "VULNERABILITYSCAN": "Attack",
    "RECON-HOSTDISCOVERY": "Attack",
    # Spoofing
    "DNS_SPOOFING": "Attack",
    "MITM-ARPSPOOFING": "Attack",
    # Web-based
    "BROWSERHIJACKING": "Attack",
    "BACKDOOR_MALWARE": "Attack",
    "XSS": "Attack",
    "UPLOADING_ATTACK": "Attack",
    "SQLINJECTION": "Attack",
    "COMMANDINJECTION": "Attack",
    # Brute Force
    "DICTIONARYBRUTEFORCE": "Attack",
}

BINARY_CLASSES = ["Benign", "Attack"]
BINARY_TO_IDX = {"Benign": 0, "Attack": 1}
IDX_TO_BINARY = {0: "Benign", 1: "Attack"}

# Category Mapping (Phase 2: 8 Classes)
# Maps each of the 34 raw uppercase labels to its 8-class functional category
LABEL_MAPPING_8CLASSES = {
    # DDoS
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
    # DoS
    "DOS-UDP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",
    # Mirai
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    # Reconnaissance
    "RECON-PINGSWEEP": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "RECON-HOSTDISCOVERY": "Recon",
    # Spoofing
    "DNS_SPOOFING": "Spoofing",
    "MITM-ARPSPOOFING": "Spoofing",
    # Web-based
    "BROWSERHIJACKING": "Web",
    "BACKDOOR_MALWARE": "Web",
    "XSS": "Web",
    "UPLOADING_ATTACK": "Web",
    "SQLINJECTION": "Web",
    "COMMANDINJECTION": "Web",
    # Brute Force
    "DICTIONARYBRUTEFORCE": "BruteForce",
    # Benign
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

# 34-Class Mapping (Phase 3: 34 Fine-Grained Classes)
THIRTY_FOUR_CLASSES = sorted(list(LABEL_MAPPING_BINARY.keys()))
THIRTY_FOUR_TO_IDX = {cls_name: i for i, cls_name in enumerate(THIRTY_FOUR_CLASSES)}
IDX_TO_THIRTY_FOUR = {i: cls_name for i, cls_name in enumerate(THIRTY_FOUR_CLASSES)}
