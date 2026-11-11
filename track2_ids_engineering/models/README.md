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

*(Trained checkpoints for the implemented E1–E4 Three-Stage Hierarchical vs. Improved Flat IDS study live under [`../hierarchical_ids/models/`](../hierarchical_ids/models/README.md).)*

> [!IMPORTANT]
> Model binaries (`*.joblib`, `*.pt`, `*.pkl`) are gitignored to preserve repository cleanliness.
