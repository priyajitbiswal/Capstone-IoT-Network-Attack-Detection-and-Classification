# Model Checkpoints & Preprocessors (`models/`)

This directory stores serialized model weights, PyTorch neural network checkpoints, and stateful preprocessor objects.

---

## Directory Organization

```text
models/
├── active/                           # Generation 2 Verified Stage 3 Suite (21 Models + 2 Preprocessors)
│   ├── preprocessor_strategy1.joblib # Strategy 1 LeakFreePreprocessor (fitted strictly on Train N=350,821)
│   ├── preprocessor_strategy2.joblib # Strategy 2 LeakFreePreprocessor (fitted strictly on Train N=337,838)
│   ├── binary_*                      # Phase 1: 5 Binary models (LR, RF, LGBM, XGBoost, PyTorch MLP)
│   ├── category_*                    # Phase 2: 5 Category 8-class models
│   ├── fine_grained_*                # Phase 3: 5 Fine-grained 34-class models
│   ├── ablation_s2_*                 # Phase 4: 4 Strategy 2 models (Binary LGBM, Category LGBM, 34-class LGBM & XGB)
│   └── ablation_unweighted_*         # Phase 5: 2 Unweighted ablation models (34-class LGBM & XGBoost)
└── legacy/                           # Generation 1 Baseline Checkpoints (80/20 Random Splits)
    ├── binary/                       # 5 Binary models + binary_scaler.joblib
    ├── category_8class/              # 5 Category models + category8_scaler.joblib
    ├── attack_34class/               # 5 Fine-grained models + attack34_scaler.joblib
    └── reduced_features/             # 3 Top-20 feature models + reduced_20_scaler.joblib + top_20_feature_names.joblib
```

---

## Preprocessing & Caching Rules

1. **Stateful `LeakFreePreprocessor`**:
   - Fitted strictly on the training partition.
   - Learns training finite maximums for clipping infinite values.
   - Learns StandardScaler mean and standard deviation for scaled models (Logistic Regression / SGD and PyTorch MLP).
   - Preserves unscaled feature values for tree models (Random Forest, LightGBM, XGBoost).
2. **Inference Pipeline**:
   - `clean_transform(df)` &rarr; replaces infinities with training maximums, fills NaNs with zeros.
   - `transform_scaler(X)` &rarr; applies StandardScaler z-score transformation (for scaled models only).

> [!IMPORTANT]
> Model binaries (`*.joblib`, `*.pt`, `*.pkl`) are gitignored to preserve bandwidth and repository cleanliness.
