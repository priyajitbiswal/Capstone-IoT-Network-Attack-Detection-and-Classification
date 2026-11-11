# Track 2 Data Layout (`track2_ids_engineering/data/`)

Track 2 reads the shared CICIoT2023 raw captures and curated stratified working dataset from [`../../data/`](../../data/):
- `../../data/raw/` (`Merged01.csv`–`Merged63.csv`)
- `../../data/processed/sample_stratified.csv` (529,562 flows across 34 classes)

## Hierarchical IDS Data Handling ([`hierarchical_ids/`](../hierarchical_ids/))

Rather than duplicating CSV copies on disk for Datasets A (Binary), B (7/8 Attack Categories), C1 (16 Flood attacks), and C2 (17 Non-Flood attacks), [`../hierarchical_ids/src/data_utils.py`](../hierarchical_ids/src/data_utils.py) dynamically indexes Track 1's Strategy 1 leak-free partitions (`Train = 350,821`, `Validation = 75,172`, `Internal Test = 75,105`) and `Cross-Capture Clean = 57,280` strictly in memory via [`../../track1_dataset_evaluation/data/partitions/`](../../track1_dataset_evaluation/data/partitions/) with verified zero feature-vector overlap.
