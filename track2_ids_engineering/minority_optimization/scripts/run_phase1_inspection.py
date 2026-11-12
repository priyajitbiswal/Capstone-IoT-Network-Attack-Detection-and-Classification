"""
Phase 1 Inspection & Baseline Verification Script (Read-Only on existing artifacts).

Inspects and exports:
  - Complete 34-class and OP29 distributions across raw sample, clean deduped, Train, Val, Internal Test, and Cross-Capture
  - Exact 39 input feature metadata and duplicate/collision metadata
  - Existing E4 baseline metrics across Val, Test, and Cross for both 34-class and OP29
  - Experiment plan JSON with fixed random seed and parameter grids
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track1_dataset_evaluation.src.config import (
    FEATURE_COLUMNS,
    LABEL_MAPPING_8CLASSES,
    ORIGINAL_SAMPLE_FILE,
    PARTITION_ASSIGNMENTS_FILE,
    PARTITIONS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    HIER_MODELS_DIR,
    HIER_RESULTS_DIR,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
)
from track2_ids_engineering.minority_optimization.src.common import (
    CONFIGS_DIR,
    E4_LGBM_PARAMS,
    OLD_TO_NEW_34_TO_29,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    OP29_MINORITY_CLASSES,
    OP29_NON_FLOOD_CLASSES,
    OP29RemappedE4Adapter,
    RESULTS_DIR,
    SklearnModelAdapter,
    evaluate_predictions_full,
    load_all_partitions_read_only,
)


def main() -> None:
    print("=== PHASE 1: DATASET, PARTITION & BASELINE E4 INSPECTION ===")
    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)
    df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
    df_cross = pd.read_csv(PARTITIONS_DIR / "cross_capture_test_clean.csv")
    with open(PARTITIONS_DIR / "partition_metadata.json", "r") as f:
        pmeta = json.load(f)
    with open(PARTITIONS_DIR / "cross_capture_metadata.json", "r") as f:
        cmeta = json.load(f)

    df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_clean["partition"] = df_assign["partition"].values

    for d in (df_raw, df_clean, df_cross):
        d["U"] = d[TARGET_COLUMN].astype(str).str.strip().str.upper()

    # Exact assertions against known partition totals
    n_raw = len(df_raw)
    n_clean = len(df_clean)
    n_unique_hashes = int(df_clean[FEATURE_COLUMNS].drop_duplicates().shape[0])
    n_train = int((df_clean["partition"] == "train").sum())
    n_val = int((df_clean["partition"] == "val").sum())
    n_test = int((df_clean["partition"] == "test").sum())
    n_cross = len(df_cross)

    assert n_raw == 529562, f"Expected 529562 raw, got {n_raw}"
    assert n_clean == 501098, f"Expected 501098 clean, got {n_clean}"
    assert n_unique_hashes == 491422, f"Expected 491422 unique hashes, got {n_unique_hashes}"
    assert n_train == 350821, f"Expected 350821 train, got {n_train}"
    assert n_val == 75172, f"Expected 75172 val, got {n_val}"
    assert n_test == 75105, f"Expected 75105 test, got {n_test}"
    assert n_cross == 57280, f"Expected 57280 cross, got {n_cross}"

    print(
        f"Verified counts: Raw={n_raw:,}, Deduped={n_clean:,} (removed {n_raw - n_clean:,} exact dupes), "
        f"Unique feature vectors={n_unique_hashes:,}, Conflicting collision hashes={pmeta['duplicate_investigation']['conflicting_label_collisions']['unique_conflicting_hashes']:,}"
    )
    print(f"Verified splits: Train={n_train:,}, Val={n_val:,}, Internal Test={n_test:,}, Cross-Capture={n_cross:,}")
    print(f"Exact 8 Minority Classes: {MINORITY_ATTACK_CLASSES}")

    # Build 34-class distribution table
    rows_34 = []
    for c in THIRTY_FOUR_CLASSES:
        rows_34.append(
            {
                "class_34": c,
                "category_8": LABEL_MAPPING_8CLASSES[c],
                "op29_class": OLD_TO_NEW_34_TO_29[c],
                "family": "Benign" if c == "BENIGN" else ("Flood" if c in FLOOD_CLASSES else "Non-Flood"),
                "is_minority_8": c in MINORITY_ATTACK_CLASSES,
                "raw_sample": int((df_raw["U"] == c).sum()),
                "clean_total": int((df_clean["U"] == c).sum()),
                "train": int(((df_clean["U"] == c) & (df_clean["partition"] == "train")).sum()),
                "val": int(((df_clean["U"] == c) & (df_clean["partition"] == "val")).sum()),
                "test": int(((df_clean["U"] == c) & (df_clean["partition"] == "test")).sum()),
                "cross_capture": int((df_cross["U"] == c).sum()),
            }
        )
    df_dist_34 = pd.DataFrame(rows_34)
    df_dist_34.to_csv(RESULTS_DIR / "phase1_class_distribution_34.csv", index=False)

    # Build OP29 distribution table
    rows_29 = []
    for c29 in OP29_CLASSES:
        members = [k for k, v in OLD_TO_NEW_34_TO_29.items() if v == c29]
        rows_29.append(
            {
                "op29_class": c29,
                "merged_from_n": len(members),
                "source_classes": " | ".join(sorted(members)),
                "family": "Benign" if c29 == "BENIGN" else ("Flood" if c29 in OP29_FLOOD_CLASSES else "Non-Flood"),
                "is_minority_8": c29 in OP29_MINORITY_CLASSES,
                "raw_sample": int(df_raw["U"].isin(members).sum()),
                "clean_total": int(df_clean["U"].isin(members).sum()),
                "train": int((df_clean["U"].isin(members) & (df_clean["partition"] == "train")).sum()),
                "val": int((df_clean["U"].isin(members) & (df_clean["partition"] == "val")).sum()),
                "test": int((df_clean["U"].isin(members) & (df_clean["partition"] == "test")).sum()),
                "cross_capture": int(df_cross["U"].isin(members).sum()),
            }
        )
    df_dist_29 = pd.DataFrame(rows_29)
    df_dist_29.to_csv(RESULTS_DIR / "phase1_class_distribution_op29.csv", index=False)

    # Read existing historical baseline CSVs (without modifying any existing file)
    hist_val = pd.read_csv(HIER_RESULTS_DIR / "final_comparison_val.csv")
    hist_test = pd.read_csv(HIER_RESULTS_DIR / "final_comparison_test.csv")
    hist_cross = pd.read_csv(HIER_RESULTS_DIR / "final_comparison_cross.csv")
    hist_merged = pd.read_csv(HIER_RESULTS_DIR / "merged_flood_e4_summary.csv")

    e4_hist_34 = {}
    for sp, hdf in [("val", hist_val), ("test", hist_test), ("cross", hist_cross)]:
        row = hdf[hdf["architecture"].str.startswith("E4_Improved_Flat34")].iloc[0].to_dict()
        e4_hist_34[sp] = row

    # Verify frozen E4 in memory (strictly read-only inference) to capture unified same-session telemetry for both ORIG34 & OP29
    splits = load_all_partitions_read_only(include_test=True, include_cross_capture=True)
    e4_ckpt = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
    e4_model = joblib.load(e4_ckpt)
    e4_34_adapter = SklearnModelAdapter("E4_Baseline_34", e4_model, e4_ckpt)
    e4_29_adapter = OP29RemappedE4Adapter(e4_model, e4_ckpt, "E4_Baseline_OP29_Remap")

    baseline_rows = []
    per_class_test_rows = []

    for sp_name in ["val", "test", "cross"]:
        sp = splits[sp_name]
        X = sp["X"]
        p34 = e4_34_adapter.predict_proba(X)
        yhat34 = np.argmax(p34, axis=1)
        res34 = evaluate_predictions_full(
            model_name="E4_Baseline_34",
            split_name=sp_name,
            taxonomy="ORIG34",
            y_true=sp["y_fine"],
            y_pred=yhat34,
            y_prob=p34,
            class_names=THIRTY_FOUR_CLASSES,
            minority_classes=MINORITY_ATTACK_CLASSES,
            flood_classes=FLOOD_CLASSES,
            non_flood_classes=NON_FLOOD_CLASSES,
            model_obj_for_latency=e4_34_adapter if sp_name in ("test", "cross") else None,
            X_for_latency=X if sp_name in ("test", "cross") else None,
            model_size_bytes=e4_34_adapter.model_size_bytes(),
        )
        # Verify exact match with historical saved CSV
        assert abs(res34["summary"]["accuracy"] - float(e4_hist_34[sp_name]["accuracy"])) < 1e-4
        assert abs(res34["summary"]["macro_f1"] - float(e4_hist_34[sp_name]["macro_f1"])) < 1e-4
        assert abs(res34["summary"]["mcc"] - float(e4_hist_34[sp_name]["mcc"])) < 1e-4
        baseline_rows.append(res34["summary"])
        if sp_name == "test":
            per_class_test_rows.extend(res34["per_class_rows"])

        p29 = e4_29_adapter.predict_proba(X)
        yhat29 = e4_29_adapter.predict(X)
        res29 = evaluate_predictions_full(
            model_name="E4_Baseline_OP29_Remap",
            split_name=sp_name,
            taxonomy="OP29",
            y_true=sp["y_op29"],
            y_pred=yhat29,
            y_prob=p29,
            class_names=OP29_CLASSES,
            minority_classes=OP29_MINORITY_CLASSES,
            flood_classes=OP29_FLOOD_CLASSES,
            non_flood_classes=OP29_NON_FLOOD_CLASSES,
            model_obj_for_latency=e4_29_adapter if sp_name in ("test", "cross") else None,
            X_for_latency=X if sp_name in ("test", "cross") else None,
            model_size_bytes=e4_29_adapter.model_size_bytes(),
        )
        hist_op29 = hist_merged[
            (hist_merged["split"] == sp_name)
            & (hist_merged["taxonomy"] == "OP29_MERGED_FLOOD")
            & (hist_merged["inference_mode"] == "label_mapped")
        ].iloc[0]
        assert abs(res29["summary"]["accuracy"] - float(hist_op29["accuracy"])) < 1e-4
        assert abs(res29["summary"]["macro_f1"] - float(hist_op29["macro_f1"])) < 1e-4
        assert abs(res29["summary"]["mcc"] - float(hist_op29["mcc"])) < 1e-4
        baseline_rows.append(res29["summary"])
        if sp_name == "test":
            per_class_test_rows.extend(res29["per_class_rows"])

    df_base = pd.DataFrame(baseline_rows)
    df_base.to_csv(RESULTS_DIR / "phase1_baseline_e4.csv", index=False)
    pd.DataFrame(per_class_test_rows).to_csv(RESULTS_DIR / "phase1_baseline_e4_per_class_test.csv", index=False)
    print("\nSaved Phase 1 tables & verified bit-exact parity with historical E4 (ORIG34 & OP29).")
    print(df_base[["model", "split", "taxonomy", "accuracy", "macro_f1", "mcc", "minority_f1", "minority_recall", "benign_fpr", "attack_to_benign_fnr"]].to_string(index=False))

    # Save Experiment Plan JSON (fixed seeds and explicit experiment boundaries)
    plan = {
        "random_seed": RANDOM_SEED,
        "base_lgbm_params": E4_LGBM_PARAMS,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "minority_classes_8": list(MINORITY_ATTACK_CLASSES),
        "partitions": {
            "raw_sample": n_raw,
            "deduped_clean": n_clean,
            "unique_feature_vectors": n_unique_hashes,
            "conflicting_collision_hashes": int(pmeta["duplicate_investigation"]["conflicting_label_collisions"]["unique_conflicting_hashes"]),
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "cross_capture_clean": n_cross,
            "cross_capture_raw_extracted": int(cmeta["total_records"]),
        },
        "experiment_1_smote": {
            "train_only": True,
            "weighting": "unweighted (alpha=0.0, isolated from Experiment 2)",
            "candidates": [
                "SMOTE_Std_Cap4000 (standard linear 5-NN SMOTE on all 8 minority classes up to min(4000, 8*n_c))",
                "SMOTE_ProtoSafe_Cap4000 (within-Protocol-Type 5-NN + invariant repair, skip sub-strata < 10 samples, cap min(4000, 8*n_c))",
                "SMOTE_ProtoSafe_Cap8000 (within-Protocol-Type 5-NN + invariant repair, skip sub-strata < 10 samples, cap min(8000, 10*n_c))",
                "SMOTE_ProtoSafe_Selective (skip classes with <15% any-same-class 5-NN purity [UPLOADING_ATTACK, RECON-PINGSWEEP] & filter isolated outlier seeds, cap 6000)",
            ],
            "taxonomies": ["ORIG34", "OP29"],
        },
        "experiment_2_class_weighting": {
            "smote": False,
            "power_alphas": [0.25, 0.50, 0.75, 1.00],
            "weight_labels": {
                "0.25": "Power_a025",
                "0.50": "SqrtBalanced_a050",
                "0.75": "Power_a075",
                "1.00": "Balanced_a100",
            },
            "taxonomies": ["ORIG34", "OP29"],
        },
        "experiment_3_rare_specialist": {
            "zero_flood_in_training": True,
            "specialist_train_pool": 161533,
            "specialist_positives_rare8": 9108,
            "specialist_negatives_benign_common_nonflood": 152425,
            "stage_a_binary_weights_grid": [0.0, 0.50, 1.00],
            "stage_b_8cls_weights_grid": [0.0, 0.50, 1.00],
            "validation_policy_grid": {
                "tau_rare": [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                "tau_cls": [0.0, 0.25, 0.40, 0.55],
                "tau_e4_keep": [0.65, 0.75, 0.85, 1.01],
            },
            "taxonomies": ["ORIG34", "OP29"],
        },
    }
    with open(CONFIGS_DIR / "experiment_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nSaved {CONFIGS_DIR / 'experiment_plan.json'}")


if __name__ == "__main__":
    main()
