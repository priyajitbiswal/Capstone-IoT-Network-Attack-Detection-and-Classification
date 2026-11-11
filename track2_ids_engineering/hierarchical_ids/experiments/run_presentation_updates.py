"""
Phase 1 Presentation Cleanup & Final Additive Evaluation for Track 2:
  1. Fast E2 Re-Evaluation: Swaps Stage 1 binary_random_forest.joblib for
     track1_dataset_evaluation/models/binary_lightgbm.joblib using the already
     validation-selected E2 parameters (tau_att=0.15, cat7_sqrt, f_tau=0.40) from
     val_e2_e3_sweeps.csv. Evaluates read-only on Validation (75,172), Internal
     Test (75,105), and Cross-Capture (57,280) with matched latency profiling.
  2. Merged-Flood Operational Taxonomy Audit & E4 Evaluation: Audits exact
     feature-vector collisions and mitigation equivalence across the 4 Flood twin
     groups (SYN/SYNONYMOUSIP, TCP, UDP, HTTP), recalculates E4's Accuracy,
     Macro-F1, Weighted-F1, MCC, per-class metrics, and merged confusion matrices
     on the exact same Internal Test (75,105) samples (plus Val & Cross checks),
     strictly distinguishing the original 34-class task from operational grouping.
  3. Generates two presentation-ready comparison charts in results/figures/.

All outputs use NEW filenames and leave historical Track 1 & Track 2 files untouched.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    RANDOM_SEED,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)
from track1_dataset_evaluation.src.evaluate import (
    benchmark_batch_throughput,
    benchmark_single_flow_latency,
    evaluate_predictions_v2,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import (
    Flat34BaselineArchitecture,
    HierarchicalCascade3Stage,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    HIER_MODELS_DIR,
    HIER_RESULTS_DIR,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    load_track1_splits,
)
from track2_ids_engineering.hierarchical_ids.src.metrics_eval import evaluate_end_to_end

# Operational mitigation-equivalent flood twin groups (verified protocol equivalents)
OP29_MERGE_GROUPS: Dict[str, List[str]] = {
    "FLOOD_SYN": ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD", "DDOS-SYNONYMOUSIP_FLOOD"],
    "FLOOD_TCP": ["DDOS-TCP_FLOOD", "DOS-TCP_FLOOD"],
    "FLOOD_UDP": ["DDOS-UDP_FLOOD", "DOS-UDP_FLOOD"],
    "FLOOD_HTTP": ["DDOS-HTTP_FLOOD", "DOS-HTTP_FLOOD"],
}

# Strict 4-pair sensitivity check (keeps DDOS-SYNONYMOUSIP_FLOOD separate as L3 src==dst variant)
OP30_MERGE_GROUPS: Dict[str, List[str]] = {
    "FLOOD_SYN_PAIR": ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD"],
    "FLOOD_TCP": ["DDOS-TCP_FLOOD", "DOS-TCP_FLOOD"],
    "FLOOD_UDP": ["DDOS-UDP_FLOOD", "DOS-UDP_FLOOD"],
    "FLOOD_HTTP": ["DDOS-HTTP_FLOOD", "DOS-HTTP_FLOOD"],
}


def build_operational_mapping(
    merge_groups: Dict[str, List[str]],
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Maps the 34 canonical classes onto K merged classes cleanly.
    Returns (new_class_names, old_idx_to_new_idx, prob_projection_matrix_34xK, old_name_to_new_name).
    """
    old_to_new_name: Dict[str, str] = {}
    grouped_members: set = set()
    for group_name, members in merge_groups.items():
        for m in members:
            assert m in THIRTY_FOUR_TO_IDX, f"Unknown class in merge group {group_name}: {m}"
            assert m not in grouped_members, f"Duplicate class in merge groups: {m}"
            grouped_members.add(m)
            old_to_new_name[m] = group_name

    for cls_name in THIRTY_FOUR_CLASSES:
        if cls_name not in old_to_new_name:
            old_to_new_name[cls_name] = cls_name

    # Deterministic order: BENIGN first if present, merged FLOOD_* next, remaining sorted
    merged_group_names = list(merge_groups.keys())
    unmerged = [c for c in THIRTY_FOUR_CLASSES if c not in grouped_members]
    new_class_names = sorted(unmerged + merged_group_names)
    new_to_idx = {name: i for i, name in enumerate(new_class_names)}

    old_idx_to_new_idx = np.zeros(len(THIRTY_FOUR_CLASSES), dtype=np.int64)
    proj_mat = np.zeros((len(THIRTY_FOUR_CLASSES), len(new_class_names)), dtype=np.float64)

    for old_i, old_name in enumerate(THIRTY_FOUR_CLASSES):
        new_i = new_to_idx[old_to_new_name[old_name]]
        old_idx_to_new_idx[old_i] = new_i
        proj_mat[old_i, new_i] = 1.0

    assert np.all(proj_mat.sum(axis=1) == 1.0), "Every old class must map to exactly 1 new class"
    return new_class_names, old_idx_to_new_idx, proj_mat, old_to_new_name


def audit_twin_groups_feature_overlap(
    cm_e4_test: np.ndarray,
) -> pd.DataFrame:
    """
    Audits exact 39-feature vector overlaps in Train (deduped hashes across classes)
    and exact Test swaps in E4's 34x34 confusion matrix for all 4 flood twin groups.
    """
    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)
    df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
    df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_clean["partition"] = df_assign["partition"].values
    df_train = df_clean[df_clean["partition"] == "train"].copy()
    df_train["U"] = df_train["Label"].astype(str).str.strip().str.upper()

    audit_specs = [
        ("SYN_TRIO (29-cls)", ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD", "DDOS-SYNONYMOUSIP_FLOOD"]),
        ("SYN_PAIR (30-cls)", ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD"]),
        ("TCP_PAIR", ["DDOS-TCP_FLOOD", "DOS-TCP_FLOOD"]),
        ("UDP_PAIR", ["DDOS-UDP_FLOOD", "DOS-UDP_FLOOD"]),
        ("HTTP_PAIR", ["DDOS-HTTP_FLOOD", "DOS-HTTP_FLOOD"]),
    ]

    relevant_labels = sorted({m for _, members in audit_specs for m in members})
    sub = df_train[df_train["U"].isin(relevant_labels)].copy()
    sub["vec_hash"] = pd.util.hash_pandas_object(sub[FEATURE_COLUMNS], index=False)
    hash_sets = {lbl: set(sub.loc[sub["U"] == lbl, "vec_hash"]) for lbl in relevant_labels}

    rows: List[Dict[str, Any]] = []
    for gname, members in audit_specs:
        idxs = [THIRTY_FOUR_TO_IDX[m] for m in members]
        sub_cm = cm_e4_test[np.ix_(idxs, idxs)]
        total_support = int(sum(cm_e4_test[i].sum() for i in idxs))
        diag_correct = int(np.trace(sub_cm))
        within_group_swaps = int(sub_cm.sum() - diag_correct)
        leaked_outside = int(total_support - sub_cm.sum())

        # Count unique feature hashes shared by >=2 member classes in Train
        all_member_hashes: Dict[int, int] = {}
        for m in members:
            for h in hash_sets[m]:
                all_member_hashes[h] = all_member_hashes.get(h, 0) + 1
        shared_train_hashes = sum(1 for c in all_member_hashes.values() if c >= 2)

        rows.append({
            "group": gname,
            "members": " | ".join(members),
            "test_support": total_support,
            "raw_correct": diag_correct,
            "internal_twin_swaps": within_group_swaps,
            "leaked_outside_group": leaked_outside,
            "in_group_retention_rate": round(sub_cm.sum() / max(1, total_support), 5),
            "train_cross_class_shared_hashes": shared_train_hashes,
        })

    return pd.DataFrame(rows)


def evaluate_fast_e2(
    splits: Dict[str, Dict[str, Any]],
    cm_dir: Path,
) -> pd.DataFrame:
    """Evaluates E2_Cascade_ValTuned_LGBM (Stage 1 LightGBM) read-only across Val/Test/Cross."""
    print("\n" + "=" * 80)
    print("TASK 1: E2 FAST LGBM STAGE 1 RE-EVALUATION (Val-Frozen tau_att=0.15, f_tau=0.40)")
    print("=" * 80, flush=True)

    s1_path = TRACK1_MODELS_DIR / "binary_lightgbm.joblib"
    s2_path = HIER_MODELS_DIR / "stage2_cat7_lgbm_sqrt_balanced.joblib"
    s3a_path = HIER_MODELS_DIR / "stage3a_flood_lgbm_d63_sqrt.joblib"
    s3b_path = HIER_MODELS_DIR / "stage3b_nonflood_lgbm_d63_sqrt.joblib"

    fast_e2 = HierarchicalCascade3Stage(
        name="E2_Cascade_ValTuned_LGBM",
        stage1_model=joblib.load(s1_path),
        stage2_model=joblib.load(s2_path),
        flood_model=joblib.load(s3a_path),
        non_flood_model=joblib.load(s3b_path),
        stage2_mode="cat7_attack",
        attack_threshold=0.15,
        flood_route_threshold=0.40,
        checkpoint_paths=[s1_path, s2_path, s3a_path, s3b_path],
    )

    cfg_out = {
        "name": fast_e2.name,
        "selection_origin": "val_e2_e3_sweeps.csv row E2_bin_lgbm_cat7_sqrt_t15_f0.40 (Val Macro-F1=0.63811)",
        "tuning_on_test": False,
        "s1_id": "bin_lgbm",
        "s1_ckpt": str(s1_path),
        "s2_id": "cat7_sqrt",
        "s2_mode": "cat7_attack",
        "s2_ckpt": str(s2_path),
        "attack_threshold": 0.15,
        "flood_route_threshold": 0.40,
        "stage3a_ckpt": str(s3a_path),
        "stage3b_ckpt": str(s3b_path),
    }
    with open(HIER_RESULTS_DIR / "e2_fast_lgbm_config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows: List[Dict[str, Any]] = []
    minority_rows: List[Dict[str, Any]] = []
    stage_diag_all: Dict[str, Any] = {}

    for split_name in ["val", "test", "cross"]:
        sp = splits[split_name]
        X_raw = sp["X"]
        perm = rng.permutation(len(X_raw))
        X_lat = X_raw[perm]

        res = evaluate_end_to_end(
            arch=fast_e2,
            X=X_raw,
            y_fine_true=sp["y_fine"],
            y_cat_true=sp["y_category"],
            is_attack_true=sp["is_attack"],
            is_flood_true=sp["is_flood"],
            is_non_flood_true=sp["is_non_flood"],
            run_latency_bench=False,
        )

        # Identical protocol to run_final_evaluation.py (non-RF: 1000 samples x 3 runs, 20k batch x 5)
        sb = benchmark_single_flow_latency(
            model=fast_e2,
            X_sample=X_lat,
            n_samples=1000,
            n_warmup=100,
            n_runs=3,
            scaler=None,
        )
        bb = benchmark_batch_throughput(
            model=fast_e2,
            X_sample=X_lat,
            batch_size=min(20000, len(X_lat)),
            n_warmup=2,
            n_runs=5,
            scaler=None,
        )

        s = res["summary"]
        ed = s.pop("error_decomposition")
        mp = s.pop("minority_per_class")
        row = {
            "split": split_name,
            **s,
            "single_latency_p50_us": sb["median_latency_us"],
            "single_latency_p95_us": sb["p95_latency_us"],
            "single_latency_p99_us": sb["p99_latency_us"],
            "batch_throughput_fps": bb["throughput_predictions_per_sec"],
            "batch_amortized_us": bb["amortized_latency_us"],
            **{f"err_{k}": v for k, v in ed.items()},
        }
        summary_rows.append(row)

        min_row = {
            "split": split_name,
            "architecture": fast_e2.name,
            "minority_macro_recall": s["minority_macro_recall"],
            "minority_macro_f1": s["minority_macro_f1"],
        }
        for mc in MINORITY_ATTACK_CLASSES:
            min_row[f"{mc}_recall"] = mp[mc]["recall"]
            min_row[f"{mc}_f1"] = mp[mc]["f1"]
        minority_rows.append(min_row)

        stage_diag_all[split_name] = res["stage_diagnostics"]
        np.save(cm_dir / f"cm_E2_Cascade_ValTuned_LGBM_{split_name}.npy", res["confusion_matrix"])

        print(
            f"  [{split_name.upper():<5}] Acc={s['accuracy']:.5f} | MacroF1={s['macro_f1']:.5f} | "
            f"MCC={s['mcc']:.5f} | MinRec={s['minority_macro_recall']:.5f} | MinF1={s['minority_macro_f1']:.5f} | "
            f"P50={sb['median_latency_us']:.1f}us | P95={sb['p95_latency_us']:.1f}us | "
            f"Batch={bb['throughput_predictions_per_sec']:.1f}fps",
            flush=True,
        )

    df_e2_fast = pd.DataFrame(summary_rows)
    df_e2_fast.to_csv(HIER_RESULTS_DIR / "e2_fast_lgbm_evaluation.csv", index=False)
    pd.DataFrame(minority_rows).to_csv(HIER_RESULTS_DIR / "e2_fast_lgbm_minority_per_class.csv", index=False)
    with open(HIER_RESULTS_DIR / "e2_fast_lgbm_stage_diagnostics.json", "w") as f:
        json.dump(stage_diag_all, f, indent=2)

    return df_e2_fast


def evaluate_merged_flood_e4(
    splits: Dict[str, Dict[str, Any]],
    cm_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluates frozen E4 under original 34-class vs justified 29/30-class Merged-Flood taxonomies."""
    print("\n" + "=" * 80)
    print("TASK 2: MERGED-FLOOD OPERATIONAL TAXONOMY AUDIT & E4 RE-EVALUATION")
    print("=" * 80, flush=True)

    e4_path = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
    e4_arch = Flat34BaselineArchitecture(
        name="E4_LGBM_Leaves63_Sqrt",
        model=joblib.load(e4_path),
        checkpoint_path=e4_path,
    )

    # Confirm E4 test latency in same session for sanity telemetry (read-only)
    rng = np.random.default_rng(RANDOM_SEED)
    _ = rng.permutation(len(splits["val"]["X"]))
    perm_test = rng.permutation(len(splits["test"]["X"]))
    e4_test_lat = benchmark_single_flow_latency(
        model=e4_arch,
        X_sample=splits["test"]["X"][perm_test],
        n_samples=1000,
        n_warmup=100,
        n_runs=3,
        scaler=None,
    )
    print(
        f"  [Sanity] E4 Test same-session single-flow P50={e4_test_lat['median_latency_us']:.1f}us "
        f"(official Test P50 in final_comparison_test.csv = 4154.25us)",
        flush=True,
    )

    map29_names, map29_idx, map29_proj, map29_dict = build_operational_mapping(OP29_MERGE_GROUPS)
    map30_names, map30_idx, map30_proj, map30_dict = build_operational_mapping(OP30_MERGE_GROUPS)

    summary_rows: List[Dict[str, Any]] = []
    per_class_test_rows: List[Dict[str, Any]] = []
    audit_df = pd.DataFrame()

    for split_name in ["val", "test", "cross"]:
        sp = splits[split_name]
        trace = e4_arch.predict_with_trace(sp["X"])
        y_true_34 = sp["y_fine"]
        y_pred_34 = trace["y_pred"]
        y_prob_34 = trace["y_prob"]

        # Original 34-class baseline verification
        base34 = evaluate_predictions_v2(
            y_true=y_true_34,
            y_pred=y_pred_34,
            y_prob=y_prob_34,
            class_names=THIRTY_FOUR_CLASSES,
        )
        if split_name == "test":
            # Verify bit-exact parity with saved historical CM and final_comparison_test.csv
            saved_cm = np.load(cm_dir / "cm_E4_Improved_Flat34_E4_LGBM_Leaves63_Sqrt_test.npy")
            assert np.array_equal(base34["confusion_matrix"], saved_cm), "E4 test CM mismatch!"
            assert abs(base34["accuracy"] - 0.75366) < 1e-4, f"Acc mismatch {base34['accuracy']}"
            assert abs(base34["macro_f1"] - 0.63523) < 1e-4, f"MacroF1 mismatch {base34['macro_f1']}"
            assert abs(base34["mcc"] - 0.74402) < 1e-4, f"MCC mismatch {base34['mcc']}"
            audit_df = audit_twin_groups_feature_overlap(base34["confusion_matrix"])
            audit_df.to_csv(HIER_RESULTS_DIR / "merged_flood_twin_group_audit.csv", index=False)

        # Helper to record taxonomy evaluation cleanly
        def _eval_taxonomy(
            taxonomy_id: str,
            mode: str,
            n_classes: int,
            class_names: List[str],
            idx_map: np.ndarray,
            proj_mat: np.ndarray,
        ) -> Dict[str, Any]:
            y_true_new = idx_map[y_true_34]
            y_prob_new = (y_prob_34 @ proj_mat).astype(np.float64)
            y_prob_new /= np.clip(y_prob_new.sum(axis=1, keepdims=True), 1e-12, None)
            if mode == "label_mapped":
                y_pred_new = idx_map[y_pred_34]
                if n_classes < 34:
                    collapsed = (proj_mat.T @ base34["confusion_matrix"] @ proj_mat).astype(np.int64)
                    direct_cm = confusion_matrix(y_true_new, y_pred_new, labels=list(range(n_classes)))
                    assert np.array_equal(collapsed, direct_cm), "Collapsed CM parity check failed!"
            else:
                y_pred_new = np.argmax(y_prob_new, axis=1)

            m = evaluate_predictions_v2(
                y_true=y_true_new,
                y_pred=y_pred_new,
                y_prob=y_prob_new,
                class_names=class_names,
            )
            rep = m["per_class_report"]
            minority_rec = float(np.mean([rep[c]["recall"] for c in MINORITY_ATTACK_CLASSES]))
            minority_f1 = float(np.mean([rep[c]["f1-score"] for c in MINORITY_ATTACK_CLASSES]))
            nonflood_f1 = float(np.mean([rep[c]["f1-score"] for c in NON_FLOOD_CLASSES]))

            if n_classes == 29:
                flood_keys = ["FLOOD_SYN", "FLOOD_TCP", "FLOOD_UDP", "FLOOD_HTTP"] + [
                    c for c in FLOOD_CLASSES if c not in {m for grp in OP29_MERGE_GROUPS.values() for m in grp}
                ]
            elif n_classes == 30:
                flood_keys = ["FLOOD_SYN_PAIR", "FLOOD_TCP", "FLOOD_UDP", "FLOOD_HTTP"] + [
                    c for c in FLOOD_CLASSES if c not in {m for grp in OP30_MERGE_GROUPS.values() for m in grp}
                ]
            else:
                flood_keys = list(FLOOD_CLASSES)

            flood_f1 = float(np.mean([rep[c]["f1-score"] for c in flood_keys]))
            flood_rec = float(np.mean([rep[c]["recall"] for c in flood_keys]))

            row = {
                "split": split_name,
                "taxonomy": taxonomy_id,
                "inference_mode": mode,
                "n_classes": n_classes,
                "n_samples": len(y_true_34),
                "accuracy": round(m["accuracy"], 5),
                "macro_f1": round(m["macro_f1"], 5),
                "weighted_f1": round(m["weighted_f1"], 5),
                "mcc": round(m["mcc"], 5),
                "macro_precision": round(m["macro_precision"], 5),
                "macro_recall": round(m["macro_recall"], 5),
                "macro_ap_ovr": round(m["macro_ap_ovr"], 5) if m["macro_ap_ovr"] is not None else None,
                "roc_auc_ovr": round(m["roc_auc_ovr"], 5) if m["roc_auc_ovr"] is not None else None,
                "flood_macro_f1": round(flood_f1, 5),
                "flood_macro_recall": round(flood_rec, 5),
                "non_flood_macro_f1": round(nonflood_f1, 5),
                "minority_macro_f1": round(minority_f1, 5),
                "minority_macro_recall": round(minority_rec, 5),
                "benign_f1": round(float(rep["BENIGN"]["f1-score"]), 5),
                "benign_recall": round(float(rep["BENIGN"]["recall"]), 5),
            }
            summary_rows.append(row)

            if split_name in ("val", "test") and taxonomy_id == "OP29_MERGED_FLOOD" and mode == "label_mapped":
                np.save(cm_dir / f"cm_E4_merged_flood_29_{split_name}.npy", m["confusion_matrix"])

            if split_name == "test" and (
                taxonomy_id == "ORIG34" or (taxonomy_id == "OP29_MERGED_FLOOD" and mode == "label_mapped")
            ):
                for cname in class_names:
                    per_class_test_rows.append({
                        "taxonomy": taxonomy_id,
                        "class": cname,
                        "precision": round(float(rep[cname]["precision"]), 5),
                        "recall": round(float(rep[cname]["recall"]), 5),
                        "f1": round(float(rep[cname]["f1-score"]), 5),
                        "support": int(rep[cname]["support"]),
                    })

            return row

        r34 = _eval_taxonomy("ORIG34", "native_34", 34, THIRTY_FOUR_CLASSES, np.arange(34), np.eye(34))
        r29_map = _eval_taxonomy("OP29_MERGED_FLOOD", "label_mapped", 29, map29_names, map29_idx, map29_proj)
        r29_prob = _eval_taxonomy("OP29_MERGED_FLOOD", "prob_sum_argmax", 29, map29_names, map29_idx, map29_proj)
        if split_name == "test":
            r30_map = _eval_taxonomy("OP30_STRICT_4_PAIR", "label_mapped", 30, map30_names, map30_idx, map30_proj)

        print(
            f"  [{split_name.upper():<5}] ORIG34: Acc={r34['accuracy']:.5f} MacroF1={r34['macro_f1']:.5f} MCC={r34['mcc']:.5f} | "
            f"OP29(map): Acc={r29_map['accuracy']:.5f} MacroF1={r29_map['macro_f1']:.5f} MCC={r29_map['mcc']:.5f} FloodF1={r29_map['flood_macro_f1']:.5f} | "
            f"OP29(prob): Acc={r29_prob['accuracy']:.5f} MacroF1={r29_prob['macro_f1']:.5f}",
            flush=True,
        )
        if split_name == "test":
            print(
                f"  [TEST OP30 Sensitivity (SYNONYMOUSIP unmerged)] Acc={r30_map['accuracy']:.5f} "
                f"MacroF1={r30_map['macro_f1']:.5f} MCC={r30_map['mcc']:.5f}",
                flush=True,
            )

    df_summary = pd.DataFrame(summary_rows)
    df_per_cls = pd.DataFrame(per_class_test_rows)
    df_summary.to_csv(HIER_RESULTS_DIR / "merged_flood_e4_summary.csv", index=False)
    df_per_cls.to_csv(HIER_RESULTS_DIR / "merged_flood_e4_per_class_test.csv", index=False)

    # Save exact taxonomy dictionary for demo integration and reproducibility
    with open(HIER_RESULTS_DIR / "merged_flood_taxonomy_map.json", "w") as f:
        json.dump(
            {
                "primary_operational_taxonomy": "OP29_MERGED_FLOOD",
                "merge_groups": OP29_MERGE_GROUPS,
                "old_to_new_34_to_29": map29_dict,
                "classes_29": map29_names,
            },
            f,
            indent=2,
        )

    return df_summary, df_per_cls, audit_df


def generate_presentation_figures(
    df_e2_fast: pd.DataFrame,
    df_merged: pd.DataFrame,
    df_per_cls_test: pd.DataFrame,
) -> None:
    """Generates clean presentation PNG charts in hierarchical_ids/results/figures/."""
    fig_dir = HIER_RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(HIER_RESULTS_DIR / "final_comparison_test.csv")
    fast_test = df_e2_fast[df_e2_fast["split"] == "test"].iloc[0]

    def _get_row(name_prefix: str) -> pd.Series:
        m = df_orig[df_orig["architecture"].str.startswith(name_prefix)]
        assert len(m) == 1, f"Expected unique match for {name_prefix}, found {len(m)}"
        return m.iloc[0]

    r_e1 = _get_row("E1_Flat34_LGBM_Balanced")
    r_e2_def = _get_row("E2_Cascade_Default_T050")
    r_e2_rf = _get_row("E2_Cascade_ValTuned")
    r_e3_meta = _get_row("E3_Parallel_MetaGate")
    r_e4 = _get_row("E4_Improved_Flat34")

    # Figure 1: Architecture Trade-offs + E2 Latency Fix (Test N=75,105)
    labels = [
        "E1 Flat Balanced\n(Track 1)",
        "E2 Default T=0.50\n(S1 LGBM)",
        "E2 Tuned (Legacy RF)\n(S1 RF n_jobs=-1)",
        "E2 Tuned (Fast LGBM)\n(S1 LGBM T=0.15)",
        "E3 MetaGate\n(S1 RF, No S2)",
        "E4 Flat L63 Sqrt\n(Primary Rec.)",
    ]
    rows = [r_e1, r_e2_def, r_e2_rf, fast_test, r_e3_meta, r_e4]
    acc_vals = [float(r["accuracy"]) * 100.0 for r in rows]
    f1_vals = [float(r["macro_f1"]) * 100.0 for r in rows]
    mcc_vals = [float(r["mcc"]) * 100.0 for r in rows]
    p50_ms = [float(r["single_latency_p50_us"]) / 1000.0 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    w = 0.25
    b1 = ax1.bar(x - w, acc_vals, width=w, label="Accuracy (%)", color="#2563EB")
    b2 = ax1.bar(x, f1_vals, width=w, label="Macro-F1 (x100)", color="#059669")
    b3 = ax1.bar(x + w, mcc_vals, width=w, label="MCC (x100)", color="#7C3AED")

    ax1.set_title("34-Class Performance on Internal Test (N = 75,105)", fontsize=11.5, fontweight="bold", pad=26)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.2, rotation=22, ha="right")
    ax1.set_ylim(50, 81)
    ax1.set_ylabel("Score (% / x100)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, fontsize=8.5, frameon=True)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)

    colors_lat = ["#64748B", "#0284C7", "#DC2626", "#16A34A", "#9333EA", "#1D4ED8"]
    bars_lat = ax2.barh(x, p50_ms, color=colors_lat, height=0.56)
    ax2.invert_yaxis()
    ax2.set_title("Single-Flow P50 Latency (ms, Log Scale)", fontsize=11.5, fontweight="bold", pad=26)
    ax2.set_xscale("log")
    ax2.set_xlim(1.0, 175.0)
    ax2.set_yticks(x)
    ax2.set_yticklabels([lbl.replace("\n", " ") for lbl in labels], fontsize=8.5)
    ax2.set_xlabel("Single-Flow P50 Latency (ms, log scale) - Lower is Better", fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars_lat, p50_ms):
        ax2.text(val * 1.12, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f} ms", va="center", ha="left", fontsize=8.5, fontweight="bold")

    fig.suptitle("Track 2 IDS Architecture Evaluation & E2 Latency Fix (Read-Only Internal Test)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(fig_dir / "e1_e4_architecture_tradeoffs.png", dpi=200)
    plt.close(fig)

    # Figure 2: Original 34-Class Benchmark vs Alternative 29-Class Operational Mitigation Taxonomy
    t_orig = df_merged[(df_merged["split"] == "test") & (df_merged["taxonomy"] == "ORIG34")].iloc[0]
    t_op29 = df_merged[
        (df_merged["split"] == "test")
        & (df_merged["taxonomy"] == "OP29_MERGED_FLOOD")
        & (df_merged["inference_mode"] == "label_mapped")
    ].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=200)
    fig.patch.set_facecolor("white")

    m_labels = ["Accuracy", "Macro-F1", "MCC", "Flood Macro-F1", "Non-Flood Macro-F1", "Minority Macro-F1"]
    m_keys = ["accuracy", "macro_f1", "mcc", "flood_macro_f1", "non_flood_macro_f1", "minority_macro_f1"]
    v34 = [float(t_orig[k]) for k in m_keys]
    v29 = [float(t_op29[k]) for k in m_keys]

    xm = np.arange(len(m_labels))
    wm = 0.34
    b34 = ax1.bar(xm - wm / 2, v34, width=wm, label="Original 34-Class Benchmark (Academic Standard)", color="#3B82F6")
    b29 = ax1.bar(xm + wm / 2, v29, width=wm, label="29-Class Operational Taxonomy (Mitigation-Merged)", color="#10B981")
    ax1.set_title("Distinct Tasks: Standard 34-Class vs. 29-Class Mitigation Taxonomy", fontsize=10.8, fontweight="bold", pad=28)
    ax1.set_xticks(xm)
    ax1.set_xticklabels(m_labels, fontsize=8.5, rotation=18, ha="right")
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score (0 - 1)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=1, fontsize=8.0, frameon=True)
    for bars in (b34, b29):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.014, f"{h:.3f}", ha="center", va="bottom", fontsize=7.6)

    # Right panel: Pre-merge 34-class F1 of the 9 twin labels vs Merged 4 Operational Flood Groups
    pc34 = df_per_cls_test[df_per_cls_test["taxonomy"] == "ORIG34"].set_index("class")
    pc29 = df_per_cls_test[df_per_cls_test["taxonomy"] == "OP29_MERGED_FLOOD"].set_index("class")

    grp_display = [
        ("SYN Trio\n(SYN+DoS-SYN+SynIP)", ["DDOS-SYN_FLOOD", "DOS-SYN_FLOOD", "DDOS-SYNONYMOUSIP_FLOOD"], "FLOOD_SYN"),
        ("TCP Twins\n(DDoS+DoS TCP)", ["DDOS-TCP_FLOOD", "DOS-TCP_FLOOD"], "FLOOD_TCP"),
        ("UDP Twins\n(DDoS+DoS UDP)", ["DDOS-UDP_FLOOD", "DOS-UDP_FLOOD"], "FLOOD_UDP"),
        ("HTTP Twins\n(DDoS+DoS HTTP)", ["DDOS-HTTP_FLOOD", "DOS-HTTP_FLOOD"], "FLOOD_HTTP"),
    ]
    pre_mean_f1 = [float(np.mean([pc34.loc[c, "f1"] for c in members])) for _, members, _ in grp_display]
    post_f1 = [float(pc29.loc[gcol, "f1"]) for _, _, gcol in grp_display]
    xg = np.arange(len(grp_display))
    bpre = ax2.bar(xg - wm / 2, pre_mean_f1, width=wm, label="Mean Unmerged Twin F1 (34-Class)", color="#F59E0B")
    bpost = ax2.bar(xg + wm / 2, post_f1, width=wm, label="Merged Protocol Group F1 (29-Class)", color="#059669")
    ax2.set_title("Flood Protocol Twin Resolution (7,840 Swaps Inside 4 Groups)", fontsize=10.8, fontweight="bold", pad=28)
    ax2.set_xticks(xg)
    ax2.set_xticklabels([g[0] for g in grp_display], fontsize=8.8)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("F1-Score on Internal Test", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=1, fontsize=8.0, frameon=True)
    for bars in (bpre, bpost):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    fig.suptitle(
        "E4 Internal Test (N=75,105): 34-Class Benchmark vs. Alternative 29-Class Operational Mitigation View\n"
        "(Operational grouping reflects IP-stripped dataset limits; NOT directly comparable as a 34-class gain)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(fig_dir / "e4_34class_vs_merged_flood.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved presentation figures to {fig_dir}", flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true", help="Regenerate figures from saved CSVs only")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.plots_only:
        df_e2_fast = pd.read_csv(HIER_RESULTS_DIR / "e2_fast_lgbm_evaluation.csv")
        df_merged = pd.read_csv(HIER_RESULTS_DIR / "merged_flood_e4_summary.csv")
        df_per_cls_test = pd.read_csv(HIER_RESULTS_DIR / "merged_flood_e4_per_class_test.csv")
        generate_presentation_figures(df_e2_fast, df_merged, df_per_cls_test)
        print(f"Regenerated figures in {time.perf_counter() - t0:.1f}s")
        return

    splits = load_track1_splits(include_test=True, include_cross_capture=True, verify_disjoint=True)
    cm_dir = HIER_RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    df_e2_fast = evaluate_fast_e2(splits, cm_dir)
    df_merged, df_per_cls_test, audit_df = evaluate_merged_flood_e4(splits, cm_dir)
    generate_presentation_figures(df_e2_fast, df_merged, df_per_cls_test)

    print("\n--- TWIN GROUP AUDIT SUMMARY (INTERNAL TEST E4) ---")
    print(audit_df.to_string(index=False))
    print(f"\nCompleted presentation updates in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
