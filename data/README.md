# Shared Data Repository (`data/`)

This directory houses the shared raw CICIoT2023 network captures and curated stratified working sample used across both research tracks. *(Track 1 leak-free hash partitions live in [`../track1_dataset_evaluation/data/partitions/`](../track1_dataset_evaluation/data/partitions/).)*

---

## Directory Organization

```text
data/
├── raw/                              # 63 full-scale CICIoT2023 captures (Merged01.csv - Merged63.csv, 8.87 GB)
└── processed/                        # Curated working sample
    └── sample_stratified.csv         # 529,562 flows (capped floods at 20k, 100% rare attack retention)
```

---

## Dataset Acquisition & Regeneration

### 1. Primary Acquisition (UNB CIC Portal)
The raw dataset is distributed by the Canadian Institute for Cybersecurity:
- **URL**: [https://www.unb.ca/cic/datasets/iot-dataset-2023.html](https://www.unb.ca/cic/datasets/iot-dataset-2023.html)
- **License**: CC BY-SA 4.0
- Place `Merged01.csv` through `Merged25.csv` into `data/raw/` to regenerate the stratified sample.
- Place `Merged26.csv` through `Merged40.csv` into `data/raw/` to regenerate the Track 1 cross-capture holdout.

### 2. Guaranteed Local Regeneration
Run the streaming sampler to generate `data/processed/sample_stratified.csv`:
```powershell
.\.venv\Scripts\python.exe -m track2_ids_engineering.src.create_sample
```
- **Output Record Count**: Exactly 529,562 flows (34 classes).
- **Verified SHA-256 Checksum**: `39370152e9b2cf95e64e1df650d3bc677241f811cd38983894502fd1f8baa227`

*(Note: Raw benchmark files and the 106.43 MB sample are excluded from Git to maintain a lean repository and comply with GitHub file size restrictions).*

---

## Partitioning Integrity & Zero Leakage (Track 1)

Track 1 partition maps are stored in [`../track1_dataset_evaluation/data/partitions/`](../track1_dataset_evaluation/data/partitions/) and generated deterministically via [`../track1_dataset_evaluation/src/data_partitioning.py`](../track1_dataset_evaluation/src/data_partitioning.py) and [`../track1_dataset_evaluation/src/cross_capture.py`](../track1_dataset_evaluation/src/cross_capture.py):
1. **Strategy 1 (Primary Benchmark, N = 501,098)**: Feature hashes are grouped atomically after exact duplicate row removal (28,464 rows). All flows sharing an identical feature vector reside in exactly one partition (Train: 350,821; Val: 75,172; Test: 75,105). Strictly zero identical-feature coordinate overlap exists between splits. Strategy 1 retains real-world network ambiguity (including 8,740 conflicting label collision hashes).
2. **Strategy 2 (Collision-Purged Ablation, N = 482,682)**: All 8,740 unique feature hashes associated with conflicting labels across classes (18,416 flows in the unique stratum; 28,608 rows in the full sample) are completely purged from the dataset (Train: 337,838; Val: 72,390; Test: 72,454).
3. **Cross-Capture Clean Test (N = 57,280)**: Sampled from subsequent recording sessions on the same physical IoT testbed (`Merged26.csv` through `Merged40.csv`), and filtered to remove any feature hashes present in the training set (0.00% feature overlap).
