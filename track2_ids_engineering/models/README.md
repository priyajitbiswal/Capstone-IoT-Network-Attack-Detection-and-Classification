# Track 2 Model Checkpoints (`track2_ids_engineering/models/`)

This directory stores serialized model weights and scalers for Track 2 IDS engineering.

---

## Directory Organization

```text
models/
├── README.md                         # (This file)
├── binary/                           # 5 Binary models (LR, RF, LGBM, XGBoost, PyTorch DNN) + binary_scaler.joblib
├── category_8class/                  # 5 Category 8-class models + category8_scaler.joblib
├── attack_34class/                   # 5 Fine-grained 34-class models + attack34_scaler.joblib
└── reduced_features/                 # 3 Top-20 edge models (RF, LGBM, XGBoost) + reduced_20_scaler.joblib + top_20_feature_names.joblib
```

*(Future trained checkpoints for the proposed Three-Stage Hierarchical IDS — Stage 1 Binary, Stage 2 8-Class Category, Stage 3A Flood Specialist, Stage 3B Non-Flood Specialist — can be organized under `models/hierarchical/` when implemented.)*

> [!IMPORTANT]
> Model binaries (`*.joblib`, `*.pt`, `*.pkl`) are gitignored to preserve repository cleanliness.
