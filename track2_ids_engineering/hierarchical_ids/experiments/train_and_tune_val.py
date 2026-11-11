"""
Phases 2-4: Modular Training, Validation-Only Tuning & Calibration for E1-E4.

STRICT DATA ISOLATION:
  - Loads ONLY `train` (N = 350,821) and `val` (N = 75,172) partitions.
  - NEVER touches Internal Test (N = 75,105) or Cross-Capture Holdout (N = 57,280).
  - Reuses Track 1's `preprocessor_strategy1.joblib` and Track 1's E1, binary,
    category, and unweighted checkpoints strictly read-only.
  - Caches validation probability outputs during threshold/fusion/calibration sweeps
    for rapid, deterministic validation selection.

Run modularly or end-to-end:
  python -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase components
  python -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e2_e3
  python -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e4
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401
from track1_dataset_evaluation.src.config import (
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    FEATURE_COLUMNS,
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    RANDOM_SEED,
    THIRTY_FOUR_CLASSES,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import (
    Flat34BaselineArchitecture,
    HierarchicalCascade3Stage,
    ParallelSpecialistFusion,
    _FLOOD_LOCAL_TO_GLOBAL_ARR,
    _NON_FLOOD_LOCAL_TO_GLOBAL_ARR,
    _compute_specialist_meta_features,
    apply_temperature_scaling,
    fit_temperature_on_val,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    ATTACK_CATEGORIES_7,
    ATTACK_CAT7_TO_IDX,
    BENIGN_GLOBAL_IDX,
    FLOOD_CATEGORIES,
    FLOOD_CLASSES,
    FLOOD_GLOBAL_INDICES,
    FLOOD_GLOBAL_TO_LOCAL,
    HIER_MODELS_DIR,
    HIER_RESULTS_DIR,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    NON_FLOOD_GLOBAL_INDICES,
    NON_FLOOD_GLOBAL_TO_LOCAL,
    compute_sample_weights,
    load_track1_splits,
)
from track2_ids_engineering.hierarchical_ids.src.metrics_eval import evaluate_end_to_end

HIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
HIER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_FLOOD_CAT7_SET = {i for i, c in enumerate(ATTACK_CATEGORIES_7) if c in FLOOD_CATEGORIES}
_FLOOD_CAT8_SET = {EIGHT_TO_IDX[c] for c in FLOOD_CATEGORIES}
_BENIGN_CAT8_IDX = EIGHT_TO_IDX["Benign"]
_ATTACK_CAT8_ARR = np.asarray([EIGHT_TO_IDX[c] for c in ATTACK_CATEGORIES_7], dtype=np.int64)
_MINORITY_INDICES = [THIRTY_FOUR_CLASSES.index(c) for c in MINORITY_ATTACK_CLASSES]


def _train_or_load_lgbm(
    ckpt_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_scheme: str = "balanced",
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    force: bool = False,
) -> Tuple[lgb.LGBMClassifier, float]:
    if ckpt_path.exists() and not force:
        print(f"    [CACHE] Loading existing {ckpt_path.name}", flush=True)
        return joblib.load(ckpt_path), 0.0
    sample_w = compute_sample_weights(y_train, scheme=weight_scheme)
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, sample_weight=sample_w)
    dt = time.perf_counter() - t0
    joblib.dump(clf, ckpt_path)
    return clf, dt


def _train_or_load_xgb_hist(
    ckpt_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    weight_scheme: str = "balanced",
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    force: bool = False,
) -> Tuple[xgb.XGBClassifier, float]:
    if ckpt_path.exists() and not force:
        print(f"    [CACHE] Loading existing {ckpt_path.name}", flush=True)
        return joblib.load(ckpt_path), 0.0
    sample_w = compute_sample_weights(y_train, scheme=weight_scheme)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, sample_weight=sample_w)
    dt = time.perf_counter() - t0
    joblib.dump(clf, ckpt_path)
    return clf, dt


def _fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=_MINORITY_INDICES, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 5),
        "minority_macro_recall": round(float(np.mean(recalls)), 5),
    }


def run_components_phase(splits: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Trains/loads Stage 2 routers & Stage 3A/3B specialists and benchmarks all components on Val."""
    print("\n=== PHASE A: STAGE 1-3 COMPONENT TRAINING & VALIDATION BENCHMARK ===", flush=True)
    tr, va = splits["train"], splits["val"]
    X_train, y_fine_tr, y_cat_tr = tr["X"], tr["y_fine"], tr["y_category"]
    X_val, y_fine_va, y_cat_va = va["X"], va["y_fine"], va["y_category"]

    p_bin_lgb = TRACK1_MODELS_DIR / "binary_lightgbm.joblib"
    p_bin_rf = TRACK1_MODELS_DIR / "binary_random_forest.joblib"
    p_cat8_lgb = TRACK1_MODELS_DIR / "category_lightgbm.joblib"

    bin_lgb = joblib.load(p_bin_lgb)
    bin_rf = joblib.load(p_bin_rf)
    cat8_lgb = joblib.load(p_cat8_lgb)

    component_rows: List[Dict[str, Any]] = []

    # Stage 1 binary checks
    for bname, bm in [("binary_lightgbm_reused", bin_lgb), ("binary_random_forest_reused", bin_rf)]:
        pred = (bm.predict_proba(X_val)[:, 1] >= 0.50).astype(int)
        component_rows.append({
            "stage": "Stage1_Binary",
            "candidate": bname,
            "weight_scheme": "balanced (Track1)",
            "train_sec": 0.0,
            "val_samples": len(y_fine_va),
            "val_accuracy": round(accuracy_score(va["is_attack"], pred), 5),
            "val_macro_f1": round(f1_score(va["is_attack"], pred, average="macro"), 5),
            "val_attack_recall": round(recall_score(va["is_attack"], pred, pos_label=True), 5),
            "val_benign_recall": round(recall_score(va["is_attack"], pred, pos_label=False), 5),
        })
        print(f"  {bname:<32}: Acc={component_rows[-1]['val_accuracy']:.4f} | AttackRec={component_rows[-1]['val_attack_recall']:.4f} | BenignRec={component_rows[-1]['val_benign_recall']:.4f}", flush=True)

    # Stage 2 routers
    att_tr_mask, att_va_mask = tr["is_attack"], va["is_attack"]
    cat8_to_cat7 = {EIGHT_CLASSES.index(c): ATTACK_CAT7_TO_IDX[c] for c in ATTACK_CATEGORIES_7}
    y_cat7_tr = np.asarray([cat8_to_cat7[int(c)] for c in y_cat_tr[att_tr_mask]], dtype=np.int64)
    y_cat7_va = np.asarray([cat8_to_cat7[int(c)] for c in y_cat_va[att_va_mask]], dtype=np.int64)
    is_flood_att_va = va["is_flood"][att_va_mask]

    for scheme in ["balanced", "sqrt_balanced"]:
        cid = f"stage2_cat7_lgbm_{scheme}"
        ckpt = HIER_MODELS_DIR / f"{cid}.joblib"
        clf, dt = _train_or_load_lgbm(ckpt, X_train[att_tr_mask], y_cat7_tr, weight_scheme=scheme, force=force)
        preds = clf.predict(X_val[att_va_mask])
        flood_pred = np.isin(preds, list(_FLOOD_CAT7_SET))
        row = {
            "stage": "Stage2_Category",
            "candidate": cid,
            "weight_scheme": scheme,
            "train_sec": round(dt, 2),
            "val_samples": int(np.sum(att_va_mask)),
            "val_accuracy": round(accuracy_score(y_cat7_va, preds), 5),
            "val_macro_f1": round(f1_score(y_cat7_va, preds, average="macro", zero_division=0), 5),
            "val_flood_route_f1": round(f1_score(is_flood_att_va, flood_pred, zero_division=0), 5),
            "val_flood_route_recall": round(recall_score(is_flood_att_va, flood_pred, zero_division=0), 5),
        }
        component_rows.append(row)
        print(f"  {cid:<32}: Acc={row['val_accuracy']:.4f} | FloodF1={row['val_flood_route_f1']:.4f}", flush=True)

    raw_cat8_pred = np.argmax(cat8_lgb.predict_proba(X_val[att_va_mask]), axis=1)
    cat8_flood_pred = np.isin(raw_cat8_pred, list(_FLOOD_CAT8_SET))
    inv_ben_count = int(np.sum(raw_cat8_pred == _BENIGN_CAT8_IDX))
    component_rows.append({
        "stage": "Stage2_Category",
        "candidate": "stage2_cat8_lgbm_reused",
        "weight_scheme": "balanced (Track1)",
        "train_sec": 0.0,
        "val_samples": int(np.sum(att_va_mask)),
        "val_accuracy": round(accuracy_score(y_cat_va[att_va_mask], raw_cat8_pred), 5),
        "val_macro_f1": round(f1_score(y_cat_va[att_va_mask], raw_cat8_pred, average="macro", zero_division=0), 5),
        "val_flood_route_f1": round(f1_score(is_flood_att_va, cat8_flood_pred, zero_division=0), 5),
        "val_flood_route_recall": round(recall_score(is_flood_att_va, cat8_flood_pred, zero_division=0), 5),
        "notes": f"Predicted Benign on {inv_ben_count} true attacks",
    })

    # Stage 3A & 3B specialists
    fl_tr_mask, fl_va_mask = tr["is_flood"], va["is_flood"]
    nf_tr_mask, nf_va_mask = tr["is_non_flood"], va["is_non_flood"]
    y_fl_tr = np.asarray([FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_tr[fl_tr_mask]], dtype=np.int64)
    y_fl_va = np.asarray([FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_va[fl_va_mask]], dtype=np.int64)
    y_nf_tr = np.asarray([NON_FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_tr[nf_tr_mask]], dtype=np.int64)
    y_nf_va = np.asarray([NON_FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_va[nf_va_mask]], dtype=np.int64)

    s3_specs = [
        ("lgbm_balanced", "lgbm", "balanced", {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31}),
        ("lgbm_sqrt", "lgbm", "sqrt_balanced", {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31}),
        ("lgbm_d63_sqrt", "lgbm", "sqrt_balanced", {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63}),
        ("xgb_balanced", "xgb", "balanced", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6}),
        ("xgb_sqrt", "xgb", "sqrt_balanced", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6}),
    ]

    best_3a_id, best_3a_f1 = "", -1.0
    best_3b_id, best_3b_f1 = "", -1.0

    for suffix, algo, scheme, params in s3_specs:
        id_3a = f"stage3a_flood_{suffix}"
        ckpt_a = HIER_MODELS_DIR / f"{id_3a}.joblib"
        if algo == "lgbm":
            m3a, dt_a = _train_or_load_lgbm(ckpt_a, X_train[fl_tr_mask], y_fl_tr, weight_scheme=scheme, force=force, **params)
        else:
            m3a, dt_a = _train_or_load_xgb_hist(ckpt_a, X_train[fl_tr_mask], y_fl_tr, num_classes=len(FLOOD_CLASSES), weight_scheme=scheme, force=force, **params)
        p_va_a = m3a.predict_proba(X_val[fl_va_mask])
        pred_a = np.argmax(p_va_a, axis=1)
        f1_a = float(f1_score(y_fl_va, pred_a, average="macro", zero_division=0))
        acc_a = float(accuracy_score(y_fl_va, pred_a))
        nll_a = float(log_loss(y_fl_va, p_va_a, labels=list(range(len(FLOOD_CLASSES)))))
        component_rows.append({
            "stage": "Stage3A_Flood_16", "candidate": id_3a, "weight_scheme": scheme,
            "train_sec": round(dt_a, 2), "val_samples": int(np.sum(fl_va_mask)),
            "val_accuracy": round(acc_a, 5), "val_macro_f1": round(f1_a, 5), "val_nll": round(nll_a, 5),
        })
        if f1_a > best_3a_f1:
            best_3a_f1, best_3a_id = f1_a, id_3a
        print(f"  {id_3a:<32}: Acc={acc_a:.4f} | MacroF1={f1_a:.4f}", flush=True)

        id_3b = f"stage3b_nonflood_{suffix}"
        ckpt_b = HIER_MODELS_DIR / f"{id_3b}.joblib"
        if algo == "lgbm":
            m3b, dt_b = _train_or_load_lgbm(ckpt_b, X_train[nf_tr_mask], y_nf_tr, weight_scheme=scheme, force=force, **params)
        else:
            m3b, dt_b = _train_or_load_xgb_hist(ckpt_b, X_train[nf_tr_mask], y_nf_tr, num_classes=len(NON_FLOOD_CLASSES), weight_scheme=scheme, force=force, **params)
        p_va_b = m3b.predict_proba(X_val[nf_va_mask])
        pred_b = np.argmax(p_va_b, axis=1)
        f1_b = float(f1_score(y_nf_va, pred_b, average="macro", zero_division=0))
        acc_b = float(accuracy_score(y_nf_va, pred_b))
        nll_b = float(log_loss(y_nf_va, p_va_b, labels=list(range(len(NON_FLOOD_CLASSES)))))
        component_rows.append({
            "stage": "Stage3B_NonFlood_17", "candidate": id_3b, "weight_scheme": scheme,
            "train_sec": round(dt_b, 2), "val_samples": int(np.sum(nf_va_mask)),
            "val_accuracy": round(acc_b, 5), "val_macro_f1": round(f1_b, 5), "val_nll": round(nll_b, 5),
        })
        if f1_b > best_3b_f1:
            best_3b_f1, best_3b_id = f1_b, id_3b
        print(f"  {id_3b:<32}: Acc={acc_b:.4f} | MacroF1={f1_b:.4f}", flush=True)

    pd.DataFrame(component_rows).to_csv(HIER_RESULTS_DIR / "val_component_benchmarks.csv", index=False)
    sel = {"best_3a_id": best_3a_id, "best_3a_f1": best_3a_f1, "best_3b_id": best_3b_id, "best_3b_f1": best_3b_f1}
    print(f"  [SAVED] val_component_benchmarks.csv | Selected: 3A={best_3a_id}, 3B={best_3b_id}", flush=True)
    return sel


def run_e2_e3_val_phase(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Fast cached-probability validation sweeps for E2 (Cascade) and E3 (Parallel Fusion)."""
    print("\n=== PHASE B: FAST VALIDATION TUNING FOR E2 & E3 ===", flush=True)
    va = splits["val"]
    X_val, y_fine_va = va["X"], va["y_fine"]
    fl_va_mask, nf_va_mask, att_va_mask = va["is_flood"], va["is_non_flood"], va["is_attack"]

    comp_df = pd.read_csv(HIER_RESULTS_DIR / "val_component_benchmarks.csv")
    best_3a_id = comp_df[comp_df["stage"] == "Stage3A_Flood_16"].sort_values("val_macro_f1", ascending=False).iloc[0]["candidate"]
    best_3b_id = comp_df[comp_df["stage"] == "Stage3B_NonFlood_17"].sort_values("val_macro_f1", ascending=False).iloc[0]["candidate"]
    best_3a_ckpt = HIER_MODELS_DIR / f"{best_3a_id}.joblib"
    best_3b_ckpt = HIER_MODELS_DIR / f"{best_3b_id}.joblib"
    print(f"  Using specialists: 3A={best_3a_id}, 3B={best_3b_id}", flush=True)

    p_bin_lgb = TRACK1_MODELS_DIR / "binary_lightgbm.joblib"
    p_bin_rf = TRACK1_MODELS_DIR / "binary_random_forest.joblib"
    p_cat8 = TRACK1_MODELS_DIR / "category_lightgbm.joblib"
    p_s2_bal = HIER_MODELS_DIR / "stage2_cat7_lgbm_balanced.joblib"
    p_s2_sqrt = HIER_MODELS_DIR / "stage2_cat7_lgbm_sqrt_balanced.joblib"

    m_bin_lgb = joblib.load(p_bin_lgb)
    m_bin_rf = joblib.load(p_bin_rf)
    m_cat8 = joblib.load(p_cat8)
    m_s2_bal = joblib.load(p_s2_bal)
    m_s2_sqrt = joblib.load(p_s2_sqrt)
    m_3a = joblib.load(best_3a_ckpt)
    m_3b = joblib.load(best_3b_ckpt)

    # Precompute validation probabilities ONCE!
    print("  Precomputing validation probabilities across stages...", flush=True)
    s1_probs = {
        "bin_lgbm": (m_bin_lgb.predict_proba(X_val)[:, 1], p_bin_lgb),
        "bin_rf": (m_bin_rf.predict_proba(X_val)[:, 1], p_bin_rf),
    }
    s2_probs = {
        "cat7_balanced": (m_s2_bal.predict_proba(X_val), "cat7_attack", p_s2_bal),
        "cat7_sqrt": (m_s2_sqrt.predict_proba(X_val), "cat7_attack", p_s2_sqrt),
        "cat8_reused": (m_cat8.predict_proba(X_val), "cat8", p_cat8),
    }
    p3a_all = m_3a.predict_proba(X_val)
    p3b_all = m_3b.predict_proba(X_val)
    pred3a_global = _FLOOD_LOCAL_TO_GLOBAL_ARR[np.argmax(p3a_all, axis=1)]
    pred3b_global = _NON_FLOOD_LOCAL_TO_GLOBAL_ARR[np.argmax(p3b_all, axis=1)]

    sweep_rows: List[Dict[str, Any]] = []

    # --- E2 Cached Sweep ---
    best_e2_f1 = -1.0
    best_e2_cfg: Dict[str, Any] = {}

    for s1_id, (p_att, s1_p) in s1_probs.items():
        for s2_id, (p_s2, s2_mode, s2_p) in s2_probs.items():
            # Precompute p_flood and argmax route mask for Stage 2
            if s2_mode == "cat8":
                raw_cat = np.argmax(p_s2, axis=1)
                inv = raw_cat == _BENIGN_CAT8_IDX
                p_att_c = p_s2[:, _ATTACK_CAT8_ARR].copy()
                p_att_c /= np.maximum(p_att_c.sum(axis=1, keepdims=True), 1e-12)
                cat_pred = np.where(inv, _ATTACK_CAT8_ARR[np.argmax(p_att_c, axis=1)], raw_cat)
                argmax_flood = np.isin(cat_pred, list(_FLOOD_CAT8_SET))
                p_flood_s2 = np.clip((p_s2[:, EIGHT_TO_IDX["DDoS"]] + p_s2[:, EIGHT_TO_IDX["DoS"]]) / np.maximum(1.0 - p_s2[:, _BENIGN_CAT8_IDX], 1e-12), 0.0, 1.0)
            else:
                cat_local = np.argmax(p_s2, axis=1)
                argmax_flood = np.isin(cat_local, list(_FLOOD_CAT7_SET))
                p_flood_s2 = np.clip(sum(p_s2[:, idx] for idx in _FLOOD_CAT7_SET), 0.0, 1.0)

            for tau_att in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                is_att = p_att >= tau_att
                for f_tau in [None, 0.40, 0.50, 0.60]:
                    route_fl = argmax_flood if f_tau is None else (p_flood_s2 >= f_tau)
                    y_pred = np.where(~is_att, BENIGN_GLOBAL_IDX, np.where(route_fl, pred3a_global, pred3b_global))
                    m = _fast_metrics(y_fine_va, y_pred)
                    ft_str = "argmax" if f_tau is None else f"{f_tau:.2f}"
                    cid = f"E2_{s1_id}_{s2_id}_t{int(tau_att*100):02d}_f{ft_str}"
                    sweep_rows.append({"candidate_id": cid, "family": "E2", "s1": s1_id, "s2": s2_id, "tau_att": tau_att, "f_tau": ft_str, **m})
                    if m["macro_f1"] > best_e2_f1:
                        best_e2_f1 = m["macro_f1"]
                        best_e2_cfg = {
                            "name": "E2_Cascade_ValTuned",
                            "s1_id": s1_id, "s1_ckpt": str(s1_p),
                            "s2_id": s2_id, "s2_mode": s2_mode, "s2_ckpt": str(s2_p),
                            "attack_threshold": tau_att,
                            "flood_route_threshold": f_tau,
                            "stage3a_id": best_3a_id, "stage3a_ckpt": str(best_3a_ckpt),
                            "stage3b_id": best_3b_id, "stage3b_ckpt": str(best_3b_ckpt),
                        }

    print(f"  [VAL BEST E2] {best_e2_cfg['s1_id']} + {best_e2_cfg['s2_id']} (tau_att={best_e2_cfg['attack_threshold']}, f_tau={best_e2_cfg['flood_route_threshold']}) -> Val Macro-F1={best_e2_f1:.4f}", flush=True)

    # --- E3 Calibration & Fusion Sweep ---
    y_fl_va = np.asarray([FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_va[fl_va_mask]], dtype=np.int64)
    y_nf_va = np.asarray([NON_FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_va[nf_va_mask]], dtype=np.int64)
    temp_3a = fit_temperature_on_val(p3a_all[fl_va_mask], y_fl_va)
    temp_3b = fit_temperature_on_val(p3b_all[nf_va_mask], y_nf_va)
    p3a_cal = apply_temperature_scaling(p3a_all, temp_3a)
    p3b_cal = apply_temperature_scaling(p3b_all, temp_3b)
    print(f"  Fitted Val Temperature Scaling: T_3A={temp_3a:.4f} | T_3B={temp_3b:.4f}", flush=True)

    # Fit E3 LogisticRegression Meta-Gate strictly on validation attacks
    meta_X_att = _compute_specialist_meta_features(p3a_cal[att_va_mask], p3b_cal[att_va_mask])
    y_fl_bin_att = va["is_flood"][att_va_mask].astype(int)
    meta_gate_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    meta_gate_lr.fit(meta_X_att, y_fl_bin_att)
    meta_gate_ckpt = HIER_MODELS_DIR / "e3_meta_gate_lr.joblib"
    joblib.dump(meta_gate_lr, meta_gate_ckpt)

    meta_X_all = _compute_specialist_meta_features(p3a_cal, p3b_cal)
    g_meta_all = meta_gate_lr.predict_proba(meta_X_all)[:, 1]
    print(f"  Meta-Gate Flood/Non-Flood Val Attack Acc: {accuracy_score(y_fl_bin_att, meta_gate_lr.predict(meta_X_att)):.4f}", flush=True)

    # Evaluate E3 NaiveMax, TempAlpha, MetaGate across Stage 1 choices & thresholds
    # 1) NaiveMax
    p_att_e2 = s1_probs[best_e2_cfg["s1_id"]][0]
    is_att_e2 = p_att_e2 >= best_e2_cfg["attack_threshold"]
    naive_fl_wins = p3a_all.max(axis=1) >= p3b_all.max(axis=1)
    y_pred_naive = np.where(~is_att_e2, BENIGN_GLOBAL_IDX, np.where(naive_fl_wins, pred3a_global, pred3b_global))
    m_naive = _fast_metrics(y_fine_va, y_pred_naive)
    sweep_rows.append({"candidate_id": "E3_Parallel_NaiveMax", "family": "E3_naive", "s1": best_e2_cfg["s1_id"], "tau_att": best_e2_cfg["attack_threshold"], **m_naive})
    print(f"  E3_Parallel_NaiveMax : Val Acc={m_naive['accuracy']:.4f} | Macro-F1={m_naive['macro_f1']:.4f}", flush=True)

    # 2) TempAlpha sweep
    best_alpha, best_alpha_f1 = 0.50, -1.0
    max_3a_cal = p3a_cal.max(axis=1)
    max_3b_cal = p3b_cal.max(axis=1)
    for alpha in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        fl_wins = (alpha * max_3a_cal) >= ((1.0 - alpha) * max_3b_cal)
        y_pred_a = np.where(~is_att_e2, BENIGN_GLOBAL_IDX, np.where(fl_wins, pred3a_global, pred3b_global))
        ma = _fast_metrics(y_fine_va, y_pred_a)
        sweep_rows.append({"candidate_id": f"E3_TempAlpha_a{int(alpha*100)}", "family": "E3_alpha", "s1": best_e2_cfg["s1_id"], "tau_att": best_e2_cfg["attack_threshold"], "alpha": alpha, **ma})
        if ma["macro_f1"] > best_alpha_f1:
            best_alpha_f1, best_alpha = ma["macro_f1"], alpha
    print(f"  E3_TempAlpha (alpha={best_alpha:.2f}): Val Macro-F1={best_alpha_f1:.4f}", flush=True)

    # 3) MetaGate Stage 1 threshold sweep
    meta_fl_wins = (g_meta_all * max_3a_cal) >= ((1.0 - g_meta_all) * max_3b_cal)
    best_meta_f1, best_meta_tau, best_meta_s1 = -1.0, best_e2_cfg["attack_threshold"], (best_e2_cfg["s1_id"], best_e2_cfg["s1_ckpt"])
    for s1_id, (p_att, s1_p) in s1_probs.items():
        for tau_att in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            is_att = p_att >= tau_att
            y_pred_m = np.where(~is_att, BENIGN_GLOBAL_IDX, np.where(meta_fl_wins, pred3a_global, pred3b_global))
            mm = _fast_metrics(y_fine_va, y_pred_m)
            sweep_rows.append({"candidate_id": f"E3_MetaGate_{s1_id}_t{int(tau_att*100):02d}", "family": "E3_meta", "s1": s1_id, "tau_att": tau_att, **mm})
            if mm["macro_f1"] > best_meta_f1:
                best_meta_f1, best_meta_tau, best_meta_s1 = mm["macro_f1"], tau_att, (s1_id, str(s1_p))
    print(f"  [VAL BEST E3 MetaGate] {best_meta_s1[0]} (tau_att={best_meta_tau}) -> Val Macro-F1={best_meta_f1:.4f}", flush=True)

    pd.DataFrame(sweep_rows).to_csv(HIER_RESULTS_DIR / "val_e2_e3_sweeps.csv", index=False)

    return {
        "E2_default": {
            "name": "E2_Cascade_Default_T050",
            "s1_id": "bin_lgbm", "s1_ckpt": str(p_bin_lgb),
            "s2_id": "stage2_cat7_lgbm_balanced", "s2_mode": "cat7_attack", "s2_ckpt": str(p_s2_bal),
            "attack_threshold": 0.50, "flood_route_threshold": None,
            "stage3a_id": best_3a_id, "stage3a_ckpt": str(best_3a_ckpt),
            "stage3b_id": best_3b_id, "stage3b_ckpt": str(best_3b_ckpt),
        },
        "E2_tuned": best_e2_cfg,
        "E3_naive": {
            "name": "E3_Parallel_NaiveMax",
            "s1_id": best_e2_cfg["s1_id"], "s1_ckpt": best_e2_cfg["s1_ckpt"],
            "attack_threshold": best_e2_cfg["attack_threshold"], "fusion_mode": "naive_max",
            "stage3a_id": best_3a_id, "stage3a_ckpt": str(best_3a_ckpt),
            "stage3b_id": best_3b_id, "stage3b_ckpt": str(best_3b_ckpt),
        },
        "E3_temp_alpha": {
            "name": "E3_Parallel_TempAlpha",
            "s1_id": best_e2_cfg["s1_id"], "s1_ckpt": best_e2_cfg["s1_ckpt"],
            "attack_threshold": best_e2_cfg["attack_threshold"], "fusion_mode": "temp_alpha",
            "temp_flood": temp_3a, "temp_non_flood": temp_3b, "alpha_flood": best_alpha,
            "stage3a_id": best_3a_id, "stage3a_ckpt": str(best_3a_ckpt),
            "stage3b_id": best_3b_id, "stage3b_ckpt": str(best_3b_ckpt),
        },
        "E3_meta_gate": {
            "name": "E3_Parallel_MetaGate",
            "s1_id": best_meta_s1[0], "s1_ckpt": best_meta_s1[1],
            "attack_threshold": best_meta_tau, "fusion_mode": "meta_gate_lr",
            "temp_flood": temp_3a, "temp_non_flood": temp_3b, "meta_gate_ckpt": str(meta_gate_ckpt),
            "stage3a_id": best_3a_id, "stage3a_ckpt": str(best_3a_ckpt),
            "stage3b_id": best_3b_id, "stage3b_ckpt": str(best_3b_ckpt),
        },
    }


def run_e4_val_phase(splits: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Trains/loads E4 flat 34-class improvements and tunes post-hoc calibration strictly on Val."""
    print("\n=== PHASE C: E4 IMPROVED FLAT 34-CLASS TRAINING & VALIDATION TUNING ===", flush=True)
    tr, va = splits["train"], splits["val"]
    X_train, y_fine_tr = tr["X"], tr["y_fine"]
    X_val, y_fine_va = va["X"], va["y_fine"]

    p_e1_lgb = TRACK1_MODELS_DIR / "fine_grained_lightgbm.joblib"
    p_e1_unw = TRACK1_MODELS_DIR / "ablation_unweighted_fine_grained_lightgbm.joblib"

    e4_models: Dict[str, Tuple[Any, Path, float]] = {
        "E4_LGBM_Balanced_Track1": (joblib.load(p_e1_lgb), p_e1_lgb, 0.0),
        "E4_LGBM_Unweighted_Track1": (joblib.load(p_e1_unw), p_e1_unw, 0.0),
    }

    e4_specs = [
        ("E4_LGBM_SqrtBalanced", "lgbm", "sqrt_balanced", {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31}),
        ("E4_LGBM_LogBalanced", "lgbm", "log_balanced", {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31}),
        ("E4_LGBM_Clipped10", "lgbm", "clipped_balanced", {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31}),
        ("E4_LGBM_Leaves63_Sqrt", "lgbm", "sqrt_balanced", {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63}),
        ("E4_XGB_Depth8_Sqrt", "xgb", "sqrt_balanced", {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 8}),
    ]

    for eid, algo, scheme, params in e4_specs:
        ckpt = HIER_MODELS_DIR / f"{eid.lower()}.joblib"
        print(f"  Training/loading {eid}...", flush=True)
        if algo == "lgbm":
            m, dt = _train_or_load_lgbm(ckpt, X_train, y_fine_tr, weight_scheme=scheme, force=force, **params)
        else:
            m, dt = _train_or_load_xgb_hist(ckpt, X_train, y_fine_tr, num_classes=34, weight_scheme=scheme, force=force, **params)
        e4_models[eid] = (m, ckpt, dt)

    _, tr_counts = np.unique(y_fine_tr, return_counts=True)
    tr_priors = tr_counts / len(y_fine_tr)

    e4_rows: List[Dict[str, Any]] = []
    best_e4_id, best_e4_f1, best_e4_temp, best_e4_bias = "", -1.0, 1.0, None

    for eid, (m, ckpt, dt) in e4_models.items():
        p_raw = m.predict_proba(X_val)
        pred_raw = np.argmax(p_raw, axis=1)
        m_raw = _fast_metrics(y_fine_va, pred_raw)
        e4_rows.append({"candidate_id": eid, "train_sec": round(dt, 2), "temperature": 1.0, "tau_prior": 0.0, **m_raw})
        print(f"  {eid:<28}: Acc={m_raw['accuracy']:.4f} | MacroF1={m_raw['macro_f1']:.4f} | MinRec={m_raw['minority_macro_recall']:.4f} ({dt:.1f}s)", flush=True)
        if m_raw["macro_f1"] > best_e4_f1:
            best_e4_f1, best_e4_id, best_e4_temp, best_e4_bias = m_raw["macro_f1"], eid, 1.0, None

        t_val = fit_temperature_on_val(p_raw, y_fine_va)
        log_p = np.log(np.clip(p_raw.astype(np.float64), 1e-12, 1.0)) / t_val
        for tau_prior in [0.0, 0.25, 0.50, 0.75]:
            bias = -tau_prior * np.log(tr_priors) if tau_prior > 0 else None
            scores = log_p + bias[None, :] if bias is not None else log_p
            pred_cal = np.argmax(scores, axis=1)
            mc = _fast_metrics(y_fine_va, pred_cal)
            cid = f"{eid}_Cal_T{t_val:.2f}_p{int(tau_prior*100)}"
            e4_rows.append({"candidate_id": cid, "train_sec": round(dt, 2), "temperature": round(t_val, 4), "tau_prior": tau_prior, **mc})
            if mc["macro_f1"] > best_e4_f1:
                best_e4_f1, best_e4_id, best_e4_temp, best_e4_bias = mc["macro_f1"], eid, t_val, bias
                print(f"    -> New Best E4 ({cid}): Acc={mc['accuracy']:.4f} | MacroF1={mc['macro_f1']:.4f} | MinRec={mc['minority_macro_recall']:.4f}", flush=True)

    pd.DataFrame(e4_rows).to_csv(HIER_RESULTS_DIR / "val_e4_candidates.csv", index=False)
    print(f"  [VAL BEST E4] {best_e4_id} (T={best_e4_temp:.4f}, bias={'yes' if best_e4_bias is not None else 'no'}) -> Val Macro-F1={best_e4_f1:.4f}", flush=True)

    return {
        "name": "E4_Improved_Flat34",
        "base_candidate_id": best_e4_id,
        "checkpoint": str(e4_models[best_e4_id][1]),
        "temperature": best_e4_temp,
        "class_logit_bias": best_e4_bias.tolist() if best_e4_bias is not None else None,
    }


def save_frozen_config(e2_e3_cfg: Optional[Dict[str, Any]], e4_cfg: Optional[Dict[str, Any]]) -> None:
    cfg_path = HIER_RESULTS_DIR / "frozen_architectures_config.json"
    existing: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            existing = json.load(f)

    existing.update({
        "random_seed": RANDOM_SEED,
        "feature_count": len(FEATURE_COLUMNS),
        "preprocessor_path": str(TRACK1_MODELS_DIR / "preprocessor_strategy1.joblib"),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "E1": {
            "name": "E1_Flat34_LGBM_Balanced",
            "checkpoint": str(TRACK1_MODELS_DIR / "fine_grained_lightgbm.joblib"),
            "source": "Track 1 reused checkpoint (Strategy 1 balanced)",
        },
    })
    if e2_e3_cfg is not None:
        existing.update(e2_e3_cfg)
    if e4_cfg is not None:
        existing["E4"] = e4_cfg

    with open(cfg_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[SAVED] Updated {cfg_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["components", "e2_e3", "e4", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    splits = load_track1_splits(include_test=False, include_cross_capture=False, verify_disjoint=True)

    e2_e3_cfg = None
    e4_cfg = None
    if args.phase in ("components", "all"):
        run_components_phase(splits, force=args.force)
    if args.phase in ("e2_e3", "all"):
        e2_e3_cfg = run_e2_e3_val_phase(splits)
    if args.phase in ("e4", "all"):
        e4_cfg = run_e4_val_phase(splits, force=args.force)

    if e2_e3_cfg is not None or e4_cfg is not None:
        save_frozen_config(e2_e3_cfg, e4_cfg)

    print(f"\nCompleted phase='{args.phase}' in {time.perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
