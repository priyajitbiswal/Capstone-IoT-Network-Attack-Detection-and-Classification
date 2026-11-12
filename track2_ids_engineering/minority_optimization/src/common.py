"""
Shared data loading, OP29 taxonomy mapping, weighting, evaluation, and latency helpers
for Track 2 minority-class optimization experiments.

Strictly reads Track 1 partitions and preprocessor read-only; never modifies
existing Track 1 or Track 2 artifacts.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401
from track1_dataset_evaluation.src.config import (
    FEATURE_COLUMNS,
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    ORIGINAL_SAMPLE_FILE,
    PARTITION_ASSIGNMENTS_FILE,
    PARTITIONS_DIR as TRACK1_PARTITIONS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)
from track1_dataset_evaluation.src.evaluate import (
    benchmark_batch_throughput,
    benchmark_single_flow_latency,
    evaluate_predictions_v2,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import (
    BaseIDSArchitecture,
    Flat34BaselineArchitecture,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    BENIGN_GLOBAL_IDX,
    FLOOD_CATEGORIES,
    FLOOD_CLASSES,
    HIER_MODELS_DIR,
    HIER_RESULTS_DIR,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    load_track1_splits,
)

# Output directory roots (isolated under track2_ids_engineering/minority_optimization/)
MIN_OPT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = MIN_OPT_ROOT / "configs"
MODELS_DIR = MIN_OPT_ROOT / "models"
RESULTS_DIR = MIN_OPT_ROOT / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in [CONFIGS_DIR, MODELS_DIR, RESULTS_DIR, CM_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Exact E4 hyperparameters from track2_ids_engineering/hierarchical_ids/experiments/train_and_tune_val.py
E4_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Verified OP29 merge mapping (loaded from frozen Track 2 artifact)
OP29_MAP_JSON = HIER_RESULTS_DIR / "merged_flood_taxonomy_map.json"
with open(OP29_MAP_JSON, "r") as _f:
    _OP29_DATA = json.load(_f)

OP29_MERGE_GROUPS: Dict[str, List[str]] = _OP29_DATA["merge_groups"]
OLD_TO_NEW_34_TO_29: Dict[str, str] = _OP29_DATA["old_to_new_34_to_29"]
OP29_CLASSES: List[str] = _OP29_DATA["classes_29"]
OP29_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(OP29_CLASSES)}
assert len(OP29_CLASSES) == 29

# OP29 flood vs non-flood vs minority partitions
_MERGED_ORIGINALS = {m for members in OP29_MERGE_GROUPS.values() for m in members}
OP29_FLOOD_CLASSES: List[str] = sorted(
    list(OP29_MERGE_GROUPS.keys()) + [c for c in FLOOD_CLASSES if c not in _MERGED_ORIGINALS]
)
OP29_NON_FLOOD_CLASSES: List[str] = list(NON_FLOOD_CLASSES)
OP29_MINORITY_CLASSES: List[str] = list(MINORITY_ATTACK_CLASSES)
OP29_BENIGN_IDX: int = OP29_TO_IDX["BENIGN"]

# Projection arrays from 34-class indices to OP29 indices
OLD_IDX_TO_OP29_IDX = np.asarray(
    [OP29_TO_IDX[OLD_TO_NEW_34_TO_29[c]] for c in THIRTY_FOUR_CLASSES], dtype=np.int64
)
PROJ_34_TO_29 = np.zeros((34, 29), dtype=np.float64)
for _i, _j in enumerate(OLD_IDX_TO_OP29_IDX):
    PROJ_34_TO_29[_i, _j] = 1.0


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def compute_power_sample_weights(y: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    """
    Computes class-balanced power weights matching Track 2's mean-class normalization:
        raw_w[c] = (N / (K * N_c)) ** alpha
        norm_w[c] = raw_w[c] / mean_j(raw_w[j])
    When alpha == 0.5, matches hierarchical_ids.src.data_utils.compute_sample_weights(..., 'sqrt_balanced')
    bit-for-bit; when alpha == 1.0, matches 'balanced'.
    """
    if alpha == 0.0:
        return None
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    count_map = dict(zip(classes, counts))
    raw_w = {c: (n_samples / (n_classes * count_map[c])) ** float(alpha) for c in classes}
    mean_w = float(np.mean(list(raw_w.values())))
    norm_w = {c: w / mean_w for c, w in raw_w.items()}
    return np.asarray([norm_w[cls] for cls in y], dtype=np.float32)


class SklearnModelAdapter:
    """Wraps a scikit-learn/LightGBM classifier with exact predict, predict_proba, and size."""

    def __init__(self, name: str, model: Any, checkpoint_path: Path):
        self.name = name
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def model_size_bytes(self) -> int:
        return self.checkpoint_path.stat().st_size if self.checkpoint_path.exists() else 0


class OP29RemappedE4Adapter:
    """Wraps frozen 34-class E4 with deterministic label-mapped 34 -> 29 operational inference."""

    def __init__(self, e4_model: Any, checkpoint_path: Path, name: str = "E4_Baseline_OP29_Remap"):
        self.name = name
        self.e4_model = e4_model
        self.checkpoint_path = Path(checkpoint_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred34 = self.e4_model.predict(X)
        return OLD_IDX_TO_OP29_IDX[pred34]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p34 = self.e4_model.predict_proba(X)
        p29 = (p34 @ PROJ_34_TO_29).astype(np.float64)
        p29 /= np.clip(p29.sum(axis=1, keepdims=True), 1e-12, None)
        return p29

    def model_size_bytes(self) -> int:
        return self.checkpoint_path.stat().st_size if self.checkpoint_path.exists() else 0


def evaluate_predictions_full(
    model_name: str,
    split_name: str,
    taxonomy: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: List[str],
    minority_classes: List[str],
    flood_classes: List[str],
    non_flood_classes: List[str],
    model_obj_for_latency: Any = None,
    X_for_latency: Optional[np.ndarray] = None,
    model_size_bytes: int = 0,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes all required Phase 4 metrics consistently for either 34-class or OP29 taxonomy.
    """
    rss_before = get_rss_mb()
    base = evaluate_predictions_v2(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
    )
    rss_after = get_rss_mb()
    rep = base["per_class_report"]

    flood_f1 = float(np.mean([rep[c]["f1-score"] for c in flood_classes]))
    flood_rec = float(np.mean([rep[c]["recall"] for c in flood_classes]))
    nonflood_f1 = float(np.mean([rep[c]["f1-score"] for c in non_flood_classes]))
    nonflood_rec = float(np.mean([rep[c]["recall"] for c in non_flood_classes]))
    minority_f1 = float(np.mean([rep[c]["f1-score"] for c in minority_classes]))
    minority_rec = float(np.mean([rep[c]["recall"] for c in minority_classes]))

    benign_idx = class_names.index("BENIGN")
    true_benign = y_true == benign_idx
    true_attack = ~true_benign
    pred_attack = y_pred != benign_idx
    benign_recall = float(rep["BENIGN"]["recall"])
    benign_fpr = float(np.mean(pred_attack[true_benign])) if np.any(true_benign) else 0.0
    attack_to_benign_fnr = float(np.mean(~pred_attack[true_attack])) if np.any(true_attack) else 0.0

    # Routing decomposition (Benign / Flood / Non-Flood)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    flood_idx_set = {cls_to_idx[c] for c in flood_classes}
    nonflood_idx_set = {cls_to_idx[c] for c in non_flood_classes}
    true_flood = np.isin(y_true, list(flood_idx_set))
    true_nonflood = np.isin(y_true, list(nonflood_idx_set))
    pred_flood = np.isin(y_pred, list(flood_idx_set))
    pred_nonflood = np.isin(y_pred, list(nonflood_idx_set))

    summary: Dict[str, Any] = {
        "model": model_name,
        "split": split_name,
        "taxonomy": taxonomy,
        "n_classes": len(class_names),
        "n_samples": int(len(y_true)),
        "accuracy": round(base["accuracy"], 5),
        "macro_f1": round(base["macro_f1"], 5),
        "weighted_f1": round(base["weighted_f1"], 5),
        "mcc": round(base["mcc"], 5),
        "macro_precision": round(base["macro_precision"], 5),
        "macro_recall": round(base["macro_recall"], 5),
        "macro_ap_ovr": round(base["macro_ap_ovr"], 5) if base["macro_ap_ovr"] is not None else None,
        "roc_auc_ovr": round(base["roc_auc_ovr"], 5) if base["roc_auc_ovr"] is not None else None,
        "minority_f1": round(minority_f1, 5),
        "minority_recall": round(minority_rec, 5),
        "flood_macro_f1": round(flood_f1, 5),
        "flood_macro_recall": round(flood_rec, 5),
        "non_flood_macro_f1": round(nonflood_f1, 5),
        "non_flood_macro_recall": round(nonflood_rec, 5),
        "benign_f1": round(float(rep["BENIGN"]["f1-score"]), 5),
        "benign_recall": round(benign_recall, 5),
        "benign_fpr": round(benign_fpr, 5),
        "attack_to_benign_fnr": round(attack_to_benign_fnr, 5),
        "err_total_errors": int(np.sum(y_pred != y_true)),
        "err_attack_to_benign_fn": int(np.sum(true_attack & (~pred_attack))),
        "err_benign_to_attack_fp": int(np.sum(true_benign & pred_attack)),
        "err_flood_to_nonflood": int(np.sum(true_flood & pred_nonflood)),
        "err_nonflood_to_flood": int(np.sum(true_nonflood & pred_flood)),
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 3),
        "peak_rss_mb": round(max(rss_before, rss_after), 1),
    }

    if model_obj_for_latency is not None and X_for_latency is not None:
        rng = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(len(X_for_latency))
        X_lat = X_for_latency[perm]
        sb = benchmark_single_flow_latency(
            model=model_obj_for_latency,
            X_sample=X_lat,
            n_samples=min(200, len(X_lat)),
            n_warmup=25,
            n_runs=1,
            scaler=None,
        )
        bb = benchmark_batch_throughput(
            model=model_obj_for_latency,
            X_sample=X_lat,
            batch_size=min(20000, len(X_lat)),
            n_warmup=1,
            n_runs=2,
            scaler=None,
        )
        summary["p50_latency_us"] = sb["median_latency_us"]
        summary["p95_latency_us"] = sb["p95_latency_us"]
        summary["p50_latency_ms"] = round(sb["median_latency_us"] / 1000.0, 3)
        summary["p95_latency_ms"] = round(sb["p95_latency_us"] / 1000.0, 3)
        summary["batch_throughput_fps"] = bb["throughput_predictions_per_sec"]
        summary["batch_amortized_us"] = bb["amortized_latency_us"]

    if extra_meta:
        summary.update(extra_meta)

    per_class_rows = []
    for c in class_names:
        per_class_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "taxonomy": taxonomy,
                "class": c,
                "is_minority": c in minority_classes,
                "is_flood": c in flood_classes,
                "precision": round(float(rep[c]["precision"]), 5),
                "recall": round(float(rep[c]["recall"]), 5),
                "f1": round(float(rep[c]["f1-score"]), 5),
                "support": int(rep[c]["support"]),
            }
        )

    return {
        "summary": summary,
        "per_class_rows": per_class_rows,
        "confusion_matrix": base["confusion_matrix"],
    }


def load_all_partitions_read_only(
    include_test: bool = False,
    include_cross_capture: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Reuses Track 1's Strategy 1 loader read-only and adds OP29 integer target vectors.
    """
    splits = load_track1_splits(
        include_test=include_test,
        include_cross_capture=include_cross_capture,
        verify_disjoint=True,
    )
    for sp_name, sp in splits.items():
        sp["y_op29"] = OLD_IDX_TO_OP29_IDX[sp["y_fine"]]
    return splits
