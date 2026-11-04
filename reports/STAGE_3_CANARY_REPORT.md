# STAGE 3 INFRASTRUCTURE CANARY REPORT
**Experiment ID**: `binary_logistic_regression`  
**Task**: Binary Intrusion Detection (Benign vs. Attack)  
**Model Family**: Stochastic Gradient Descent Logistic Regression (`SGDClassifier`)  
**Strategy**: Strategy 1 (Primary Group-Stratified, Leak-Free)  
**Date & Time**: November 3, 2026, 23:34:27 IST  
**Execution Type**: Isolated Infrastructure Canary (Experiment 1 of 19)  
**Status**: Completed Successfully — **Exit Code 0**  

---

## 1. Executive Summary & Purpose

In accordance with strict supervisory authorization, **Experiment 1 (`binary_logistic_regression`) was executed in complete isolation as an infrastructure canary**. 

The purpose of this canary was **not** to benchmark model performance, but to rigorously validate the technical pipeline:
1. End-to-end data ingestion from disk without memory exhaustion.
2. Fit-on-train-only preprocessing and label encoding.
3. Strict enforcement of sealed test partition boundaries (zero test touching).
4. Real-time resource tracking via hardware counters and `psutil` RSS sampling.
5. In-memory streaming single-flow latency and batch throughput profiling.
6. Checkpoint persistence and centralized ledger logging.
7. Verification that execution terminates cleanly without cascading into subsequent experiments.

**Result**: The canary executed to completion in **9.22 seconds** of training time, peak process RSS peaked at **469.7 MB** (a memory delta of only **+154.5 MB**), exactly **one** ledger row was written, and both test sets remained 100% untouched.

---

## 2. Exact Execution Command & Environment

### 2.1 Command Executed
```powershell
.venv\Scripts\python.exe -m src_v2.run_stage3 --exp binary_logistic_regression
```

### 2.2 System & Runtime Environment
* **Operating System**: Windows 11 (AMD64)
* **Processor / Cores**: 16 logical CPU cores
* **Physical System RAM**: ~16.0 GB
* **Python Executable**: `C:\Users\priya\Downloads\capstone\.venv\Scripts\python.exe`
* **Python Version**: `3.13.15` (tags/v3.13.15:4061bc4, Aug 5 2026)
* **Key Dependencies**:
  * `scikit-learn`: `1.9.1`
  * `numpy`: `2.5.3`
  * `pandas`: `3.0.5`
  * `psutil`: `7.2.2`
  * `joblib`: `1.5.3`
  * `torch`: `2.14.0+cpu`
  * `lightgbm`: `4.7.0`
  * `xgboost`: `3.4.1`

---

## 3. Dataset Partitions & Sample Accounting

The canary ingested data strictly through `src_v2.preprocessing.load_partitioned_datasets` with `load_test=False`:

| Partition / Dataset | File Source | Sample Count | Features | Classes | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Partition** | `data/partitions/partition_assignments.csv` | **350,821** | 39 | 2 (`Benign`, `Attack`) | Loaded & Preprocessed (Fitted) |
| **Validation Partition** | `data/partitions/partition_assignments.csv` | **75,172** | 39 | 2 (`Benign`, `Attack`) | Loaded & Preprocessed (Transformed) |
| **Internal Test Partition** | `data/partitions/partition_assignments.csv` | **75,105** | 39 | 2 | **SEALED / UNTOUCHED** (Not loaded, not transformed, not evaluated) |
| **Cross-Capture Holdout** | `data/partitions/cross_capture_test_clean.csv` | **57,280** | 39 | 2 | **SEALED / UNTOUCHED** (Not loaded, not transformed, not evaluated) |

### Class Distribution within Partitions:
* **Training Partition ($N = 350,821$)**:
  * `Benign`: 27,998 flows (7.98%)
  * `Attack`: 322,823 flows (92.02%)
* **Validation Partition ($N = 75,172$)**:
  * `Benign`: 6,086 flows (8.10%)
  * `Attack`: 69,086 flows (91.90%)

---

## 4. Hardware Resource & Timing Profiling

Memory tracking was performed dynamically via `psutil.Process(os.getpid()).memory_info().rss`:

```
Process Memory Profile:
├── Initial Baseline RSS (before data load):   315.2 MB
├── Peak Process RSS (during fit & scaling):   469.7 MB
├── Net Memory Delta:                         +154.5 MB
└── Post-Cleanup RSS (after gc.collect()):     321.4 MB
```

### Execution Timing Breakdown:
* **Data Ingestion & Training-Only Preprocessing**: ~1.85 s
* **Model Training Duration (`model.fit`)**: **9.22 s**
* **Validation Prediction Duration (`model.predict`)**: **0.016 s** (16.2 ms across 75,172 samples)
* **Single-Flow Latency Benchmarking (2,000 samples, 5 runs)**: ~3.71 s
* **Batch Throughput Benchmarking (10,000 samples, 5 runs)**: ~0.08 s
* **Total End-to-End Canary Execution Time**: ~15.2 s

---

## 5. Validation Performance Metrics

Evaluated strictly on the independent Strategy 1 validation partition ($N = 75,172$):

### 5.1 Global Summary Metrics
* **Accuracy**: **82.04%** ($0.82044$)
* **Macro-Averaged F1-Score**: **68.03%** ($0.68034$)
* **Weighted-Averaged F1-Score**: **85.77%** ($0.85770$)
* **Macro-Averaged Precision**: **65.29%** ($0.65291$)
* **Macro-Averaged Recall (Balanced Accuracy)**: **89.24%** ($0.89242$)
* **Macro One-vs-Rest ROC-AUC**: **0.9446** ($0.94458$)
* **ROC-AUC Status**: `Calculated`

### 5.2 Granular Per-Class Performance
Recorded in `results_v2/metrics/per_class/binary_logistic_regression_per_class.json`:

| Class | Precision | Recall | F1-Score | Support (Samples) |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 30.82% ($0.3082$) | **97.83%** ($0.9783$) | 46.87% ($0.4687$) | 6,086 |
| **Attack** | 99.76% ($0.9976$) | 80.65% ($0.8065$) | 89.20% ($0.8920$) | 69,086 |
| **Macro Average** | 65.29% | 89.24% | 68.03% | 75,172 |
| **Weighted Average** | 94.18% | 82.04% | 85.77% | 75,172 |

### 5.3 Diagnostic Insight on Class Weighting:
Because `class_weight='balanced'` was passed to `SGDClassifier`, the optimizer applied inverse-prevalence weighting:
$$w_{\text{Benign}} = \frac{350,821}{2 \times 27,998} \approx 6.265, \qquad w_{\text{Attack}} = \frac{350,821}{2 \times 322,823} \approx 0.543$$
This forced the linear decision hyperplane to heavily penalize false negatives on benign traffic, yielding **97.83% Benign recall** at the cost of lower Benign precision (30.82%). This empirically confirms that balanced class weighting is actively functioning.

---

## 6. Algorithmic Latency & Throughput Benchmarking

Conducted strictly in-memory on pre-standardized feature buffers:

### 6.1 Sequential Single-Flow Streaming Latency (Microseconds / Flow)
*Simulates real-time packet arrival evaluated one sample at a time (`model.predict(x[i:i+1])`):*
* **Evaluated Samples**: 2,000 distinct validation flows
* **Warm-up Priming**: 500 iterations
* **Repetitions**: 5 independent runs
* **Median Latency (P50)**: **370.5 $\mu$s / flow**
* **95th Percentile Latency (P95)**: **611.9 $\mu$s / flow**
* **99th Percentile Latency (P99)**: **1045.8 $\mu$s / flow**
* **Min Latency**: 215.3 $\mu$s
* **Max Latency**: 2841.2 $\mu$s

### 6.2 Batch In-Memory Throughput (Predictions / Second)
*Measures vectorized bulk evaluation on contiguous memory arrays:*
* **Batch Size**: 10,000 flows
* **Warm-up Priming**: 3 iterations
* **Repetitions**: 5 independent runs
* **Throughput**: **7,093,205 predictions / second**
* **Amortized Latency**: **0.141 $\mu$s / sample**

> [!NOTE]
> The single-flow streaming latency (~370 $\mu$s) reflects the Python runtime per-call overhead and scikit-learn input validation on a slice of size $(1, 39)$. The batch throughput (~7.1 million predictions/sec) reflects vectorized BLAS inner-product execution. Neither figure represents network NIC packet capture rates.

---

## 7. Artifact Integrity & Ledger Verification

### 7.1 Model Checkpoint Verification
* **Location**: `C:\Users\priya\Downloads\capstone\models_saved_v2\binary_logistic_regression.joblib`
* **File Size on Disk**: 1,301 bytes
* **Reload Verification**: Successfully loaded via `joblib.load()` and executed inference on test tensor:
  * Returned class: `<class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>`
  * Test inference output: Valid predictions and calibrated probabilities shape `(2, 2)`.

### 7.2 Centralized Ledger Entry Verification
* **Location**: `C:\Users\priya\Downloads\capstone\results_v2\tables\centralized_experiment_ledger.csv`
* **Integrity Check**:
  * **Total Row Count**: Exactly **1** row.
  * **Total Column Count**: Exactly **25** standardized fields.
  * **Field Inspection**:
    ```text
    experiment_id:                    binary_logistic_regression
    target_type:                      binary
    model_name:                       logistic_regression
    strategy:                         Strategy 1 (Primary)
    train_samples:                    350821
    val_samples:                      75172
    num_classes:                      2
    accuracy:                         0.82044
    macro_f1:                         0.68034
    weighted_f1:                      0.8577
    macro_precision:                  0.65291
    macro_recall:                     0.89242
    roc_auc_ovr:                      0.94458
    roc_auc_status:                   Calculated
    train_duration_sec:               9.22
    val_predict_sec:                  0.016
    peak_rss_mb:                      469.7
    rss_delta_mb:                     154.5
    single_latency_p50_us:            370.5
    single_latency_p95_us:            611.9
    single_latency_p99_us:            1045.821
    batch_throughput_samples_per_sec: 7093204.7
    batch_amortized_us:               0.141
    model_params:                     {"max_iter": 100}
    random_seed:                      42
    ```

---

## 8. Test Partition Sealing Confirmation

We formally verified the timestamps and file integrity of the sealed test partitions:
* `data/partitions/cross_capture_test_clean.csv`: Last modified **20:58 IST** (untouched during canary).
* `data/partitions/partition_assignments.csv`: Last modified **20:44 IST** (untouched during canary).
* In `load_partitioned_datasets`, the test fold was explicitly marked `[SEALED / UNTOUCHED]`: `test_df = None`, `X_test = None`, `y_test = None`.
* Zero test records were loaded into memory, standardized, or predicted upon.

---

## 9. Discrepancies, Warnings & Behavioral Observations

1. **Warnings & Errors**: **None**. Process exited with **code 0**. No deprecation warnings, convergence warnings, or unhandled exceptions occurred.
2. **Memory Consumption**: Peak RSS (+154.5 MB delta) was substantially lower than our provisional budget ceiling of 1,200 MB, proving that pandas float32 conversions and garbage collection effectively prevented memory bloat.
3. **Execution Isolation**: The runner executed only Experiment 1 and terminated cleanly; Experiment 2 was not touched.

---

## 10. Conclusion & Stop Point

The infrastructure canary has **successfully verified the entire Stage 3 experimental pipeline**. 

* **Status**: Execution is now **strictly paused**.
* **Next Action**: Awaiting supervisory review of this canary report before proposing any subsequent step.
