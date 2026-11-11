# Stage 3 Complete Report: Model Training, Canary Check, Decomposed Validation & Profiling

> Authoritative Stage 3 training & validation report (`track1_dataset_evaluation/src/run_stage3.py`). Test (`75,105`) and Cross-Capture (`57,280`) partitions remained strictly sealed throughout Stage 3.

---

## 1. Canary Parity & Suite Execution Overview

- **Infrastructure Canary (`binary_logistic_regression`)**: Executed first in isolation (`train = 350,821`, `val = 75,172`) verifying baseline RSS (`315.2 MB -> 469.7 MB` peak, `+154.5 MB` delta), training time (`9.22 s`), P50 single-flow latency (`370.5 us`), vectorized batch throughput (`7.09M flows/s`), and zero test-partition access.
- **21-Model Suite (`5 phases`)**:
  - Phases 1–3: 5 model families (**SGD LogReg**, **Random Forest**, **LightGBM**, **BalancedXGBoost**, **PyTorch Tabular MLP**) × 3 tasks (**Binary**, **8-Class Category**, **34-Class Fine-Grained**) = **15 Strategy 1 models**.
  - Phase 4: **4 Strategy 2 collision-purged ablations** (Binary LGBM, Category LGBM, Fine-Grained LGBM & XGB).
  - Phase 5: **2 Unweighted 34-class ablations** (`class_weight=None` on LightGBM & XGBoost) added to isolate cost-sensitive weighting impact (`15 + 4 + 2 = 21` checkpoints + `preprocessor_strategy1.joblib` & `preprocessor_strategy2.joblib`).
- **Unified Feature Schema & Preprocessing**: All 39 features (`FEATURE_COLUMNS`: frame/header 3, `Rate` 1, TCP flag indicators 7, cumulative flag counters 4, protocol flags 14, packet window statistics 10 — note 7 features from Neto et al.'s 46-feature list are omitted in the source CSVs). `LeakFreePreprocessor` fit finite bounds + `StandardScaler` strictly on `Train` (`scale_features=True`).

### Standardized Model Hyperparameters (`random_state=42`)
- **SGD LogReg**: `SGDClassifier(loss="log_loss", max_iter=100, tol=1e-3, class_weight="balanced")`
- **Random Forest**: `RandomForestClassifier(n_estimators=50, max_depth=15, criterion="gini", class_weight="balanced", n_jobs=-1)`
- **LightGBM**: `LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, class_weight="balanced", n_jobs=-1)`
- **BalancedXGBoost**: `XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, tree_method="hist")` + `compute_sample_weight("balanced", y)`
- **PyTorch MLP**: `39 -> Linear(128)-BN-ReLU-Drop(0.2) -> Linear(64)-BN-ReLU-Drop(0.2) -> K`, `AdamW(lr=0.002, wd=1e-4)`, `batch_size=2048`, **5 epochs** (`860` updates), inverse-frequency `CrossEntropyLoss` weights.

---

## 2. Complete Validation Benchmark Across All 21 Models (`Val = 75,172` S1 / `72,390` S2)

| Exp ID | Task | Model | Strategy | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val MCC | Val Macro-AP | Val ROC-AUC |
|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `binary_logistic_regression` | Binary | SGD LogReg | S1 | 0.8204 | 0.6803 | 0.8577 | 0.4899 | 0.9951 | 0.9446 |
| `binary_random_forest` | Binary | Random Forest | S1 | **0.8982** | **0.7733** | **0.9143** | **0.6119** | 0.9977 | 0.9740 |
| `binary_lightgbm` | Binary | LightGBM | S1 | 0.8928 | 0.7665 | 0.9104 | 0.6046 | **0.9978** | **0.9750** |
| `binary_xgboost` | Binary | XGBoost | S1 | 0.8800 | 0.7501 | 0.9011 | 0.5841 | 0.9976 | 0.9732 |
| `binary_pytorch_dnn` | Binary | PyTorch MLP | S1 | 0.8515 | 0.7143 | 0.8802 | 0.5341 | 0.9969 | 0.9646 |
| `category_logistic_regression` | 8-Class | SGD LogReg | S1 | 0.7425 | 0.6006 | 0.7532 | 0.6743 | 0.6351 | 0.9510 |
| `category_random_forest` | 8-Class | Random Forest | S1 | 0.7819 | 0.6802 | 0.7983 | 0.7398 | 0.7670 | 0.9759 |
| `category_lightgbm` | 8-Class | LightGBM | S1 | **0.7871** | **0.6865** | **0.8033** | **0.7443** | **0.7821** | **0.9781** |
| `category_xgboost` | 8-Class | XGBoost | S1 | 0.7603 | 0.6615 | 0.7800 | 0.7176 | 0.7530 | 0.9742 |
| `category_pytorch_dnn` | 8-Class | PyTorch MLP | S1 | 0.7336 | 0.6104 | 0.7572 | 0.6825 | 0.7038 | 0.9653 |
| `fine_grained_logistic_regression` | 34-Class | SGD LogReg | S1 | 0.6421 | 0.5260 | 0.6405 | 0.6300 | 0.5313 | 0.9616 |
| `fine_grained_random_forest` | 34-Class | Random Forest | S1 | 0.7033 | 0.5956 | 0.7105 | 0.6946 | 0.6404 | 0.9779 |
| `fine_grained_lightgbm` | 34-Class | LightGBM | S1 | **0.7246** | **0.6142** | **0.7321** | **0.7149** | **0.6638** | **0.9812** |
| `fine_grained_xgboost` | 34-Class | XGBoost | S1 | 0.7071 | 0.5984 | 0.7153 | 0.6977 | 0.6471 | 0.9793 |
| `fine_grained_pytorch_dnn` | 34-Class | PyTorch MLP | S1 | 0.6246 | 0.5211 | 0.6235 | 0.6140 | 0.5798 | 0.9703 |
| `ablation_s2_binary_lightgbm` | Binary | LightGBM | S2 | 0.8913 | 0.7686 | 0.9088 | 0.6072 | 0.9976 | 0.9741 |
| `ablation_s2_category_lightgbm` | 8-Class | LightGBM | S2 | **0.7987** | **0.6925** | **0.8146** | **0.7579** | **0.7869** | **0.9789** |
| `ablation_s2_fine_grained_lightgbm` | 34-Class | LightGBM | S2 | **0.7376** | **0.6218** | **0.7459** | **0.7282** | **0.6698** | **0.9814** |
| `ablation_s2_fine_grained_xgboost` | 34-Class | XGBoost | S2 | 0.7186 | 0.6034 | 0.7290 | 0.7094 | 0.6524 | 0.9792 |
| `ablation_unweighted_fine_grained_lightgbm` | 34-Class | LightGBM | Unweighted | **0.7433** | **0.6239** | 0.7330 | **0.7333** | 0.6424 | 0.9718 |
| `ablation_unweighted_fine_grained_xgboost` | 34-Class | XGBoost | Unweighted | 0.7403 | 0.6139 | 0.7259 | 0.7306 | 0.6535 | 0.9804 |

---

## 3. Decomposed Validation (`Clean = 72,390` vs. `Ambiguous = 2,782`)

Because Strategy 2 removes only the `8,740` conflicting collision hashes in place, $\mathcal{D}_{\text{val, S2}} \equiv \mathcal{D}_{\text{val, clean}} \subset \mathcal{D}_{\text{val, S1}}$ (`N = 72,390`), with $\mathcal{D}_{\text{val, ambig}} = 2,782$ rows. Evaluating both strategies on the invariant clean slice shows collision purging yields only a **marginal `+0.0022` (+0.22%) LightGBM Macro-F1 difference** (`-0.0008` XGBoost); S2's apparent jump over S1's full validation metric (`0.6218` vs `0.6142`) mostly reflects omitting the `2,782` contradictory rows where models observe ~`41.7%` accuracy.

| Architecture | Strategy | Full Val Acc / F1 (75,172) | Clean Acc / F1 (72,390) | Clean MCC | Ambig Acc / F1 (2,782) |
|---|---|:---:|:---:|:---:|:---:|
| **Binary LightGBM** | S1 vs. S2 | 0.8928 / 0.7665 | 0.8887 / 0.7652 vs. **0.8913 / 0.7686** | 0.6030 vs. **0.6072** | 0.9989 / 0.4997 |
| **Category LightGBM** | S1 vs. S2 | 0.7871 / 0.6865 | 0.7968 / 0.6910 vs. **0.7987 / 0.6925** | 0.7562 vs. **0.7579** | 0.5352 / 0.4236 |
| **Fine-Grained LightGBM** | S1 vs. S2 | 0.7246 / 0.6142 | 0.7364 / 0.6196 vs. **0.7376 / 0.6218** | 0.7270 vs. **0.7282** | 0.4173 / 0.2386 |
| **Fine-Grained XGBoost** | S1 vs. S2 | 0.7071 / 0.5984 | **0.7185 / 0.6042** vs. 0.7186 / 0.6034 | 0.7092 vs. **0.7094** | 0.4105 / 0.2220 |

---

## 4. Validation Ablations, Threat Tiers & Worst-10 Attack Classes

### Balanced vs. Unweighted 34-Class Ablation (`Val = 75,172`)
Unweighted training raises LightGBM Validation Accuracy (`72.46% -> 74.33%`, `+1.88%`) and Benign Recall (`65.84% -> 84.64%`), while causing catastrophic minority-recall drops on Tier 4 Web Exploits (**`36.67% -> 18.53%`** Macro-Recall; XGBoost `36.03% -> 13.22%`):
- `SQLINJECTION`: **`34.13% -> 5.56%`** (`-83.7%` relative); `RECON-PINGSWEEP`: `24.32% -> 5.41%` (`-77.8%`); `XSS`: `27.22% -> 7.10%` (`-73.9%`); `BACKDOOR_MALWARE`: `31.61% -> 11.61%` (`-63.3%`); `BROWSERHIJACKING`: `53.51% -> 31.00%`; `DICTIONARYBRUTEFORCE`: `51.05% -> 32.74%`.

### Functional Threat-Tier Validation Recall (`fine_grained_*`, `Val = 75,172`)
- **Tier 1 Volumetric Floods (16 classes, `40,760`)**: LightGBM **`0.7991`**, RF `0.7930`, XGBoost `0.7918`, DNN `0.7646`, SGD `0.7481`.
- **Tier 2 Mirai Botnets (3 classes, `8,863`)**: LightGBM **`0.9941`**, XGBoost `0.9926`, RF `0.9924`, SGD `0.9427`, DNN `0.6911` (GRE encapsulation signals make Mirai >99% separable by trees).
- **Tier 3 Recon & Spoofing (7 classes, `17,730`)**: LightGBM **`0.4498`**, XGBoost `0.4143`, RF `0.3823`, DNN `0.3029`, SGD `0.3018`.
- **Tier 4 Web & Exploits (7 classes, `1,733`)**: LightGBM **`0.3667`**, XGBoost `0.3603`, RF `0.3594`, DNN `0.3066`, SGD `0.2105` (payload-blind tabular features cap Web precision/F1).
- **Benign (`6,086`)**: LightGBM **`0.6584`**, RF `0.6462`, XGBoost `0.6364`, DNN `0.5871`, SGD `0.4947`.

---

## 5. Computational Resource & 5-Epoch DNN Loss Telemetry

| Experiment ID | Train (s) | Val Pred (s) | Peak RSS (MB) | P50 Latency (us) | Batch (flows/s) |
|---|---:|---:|---:|---:|---:|
| `binary_logistic_regression` | 9.22 | 0.005 | 481.7 | 362.1 | 8,325,008 |
| `binary_random_forest` | 10.76 | 0.268 | 498.9 | 70,883.1 | 62,403 |
| `binary_lightgbm` | 1.46 | 0.038 | 494.7 | 1,983.3 | 938,430 |
| `binary_xgboost` | 2.58 | 0.107 | 497.5 | 1,575.3 | 1,424,258 |
| `binary_pytorch_dnn` | 41.11 | 0.400 | 650.9 | 247.1 | 163,543 |
| `category_logistic_regression` | 9.22 | 0.010 | 484.7 | 143.2 | 8,007,687 |
| `category_random_forest` | 5.03 | 0.106 | 526.9 | 22,467.5 | 330,334 |
| `category_lightgbm` | 15.50 | 0.806 | 484.8 | 1,409.8 | 92,413 |
| `category_xgboost` | 15.39 | 0.281 | 483.6 | 2,244.0 | 264,813 |
| `category_pytorch_dnn` | 57.36 | 0.759 | 638.6 | 613.8 | 123,569 |
| `fine_grained_logistic_regression` | 127.44 | 0.018 | 482.8 | 189.7 | 2,901,494 |
| `fine_grained_random_forest` | 9.72 | 0.525 | 572.6 | 31,296.1 | 157,485 |
| `fine_grained_lightgbm` | 60.23 | 7.828 | 493.8 | 3,504.4 | 10,989 |
| `fine_grained_xgboost` | 76.52 | 1.788 | 501.8 | 3,732.5 | 47,558 |
| `fine_grained_pytorch_dnn` | 80.24 | 1.064 | 638.9 | 942.3 | 74,707 |

- **PyTorch MLP 5-Epoch Loss Convergence (`Train -> Val`)**: Binary (`0.27454 -> 0.21951` / `0.22486 -> 0.21118`), Category (`1.00838 -> 0.78523` / `0.82711 -> 0.75077`), Fine-Grained (`1.67787 -> 1.18950` / `1.24223 -> 1.12791`). Note Random Forest P50 latency (`22.5–70.9 ms`) stems from `n_jobs=-1` single-row joblib thread-pool dispatch overhead despite `62k–330k fps` batch throughput.
