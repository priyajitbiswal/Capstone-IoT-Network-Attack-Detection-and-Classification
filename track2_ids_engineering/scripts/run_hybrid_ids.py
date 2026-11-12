"""
CLI Entrypoint for the Promoted Main Track 2 IDS Pipeline (E4 + Exp3 Hybrid Specialist).

Examples:
  # Verify read-only on Internal Test or Cross-Capture (OP29 default)
  python -m track2_ids_engineering.scripts.run_hybrid_ids --split test --taxonomy OP29
  python -m track2_ids_engineering.scripts.run_hybrid_ids --split cross --taxonomy ORIG34 --sample-n 5000

  # Run batch inference on a raw CICIoT2023 CSV (with 39 canonical feature columns)
  python -m track2_ids_engineering.scripts.run_hybrid_ids --input path/to/flows.csv --output preds.csv --taxonomy OP29
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track1_dataset_evaluation.src.config import THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    load_track1_splits,
)
from track2_ids_engineering.minority_optimization.src.common import (
    OLD_IDX_TO_OP29_IDX,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    OP29_MINORITY_CLASSES,
    OP29_NON_FLOOD_CLASSES,
    evaluate_predictions_full,
)
from track2_ids_engineering.minority_optimization.src.specialist import (
    RARE_8_CLASSES,
    analyze_routing_and_override_outcomes,
)
from track2_ids_engineering.src.hybrid_pipeline import (
    load_promoted_hybrid_ids,
    preprocess_raw_flows,
)


def run_split_verification(split: str, taxonomy: str, sample_n: Optional[int]) -> None:
    """Evaluates the promoted pipeline read-only on an existing Track 1 Strategy 1 split."""
    model, _ = load_promoted_hybrid_ids(taxonomy=taxonomy, with_preprocessor=False)
    splits = load_track1_splits(
        include_test=(split == "test"),
        include_cross_capture=(split == "cross"),
        verify_disjoint=False,
    )
    sp = splits[split]
    X = sp["X"]
    y_34 = sp["y_fine"]
    y_true = y_34 if taxonomy == "ORIG34" else OLD_IDX_TO_OP29_IDX[y_34]

    if sample_n is not None and sample_n < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_n, replace=False)
        X, y_true = X[idx], y_true[idx]

    trace = model.predict_with_trace(X)
    is_op29 = taxonomy == "OP29"
    eval_res = evaluate_predictions_full(
        model_name=model.name,
        split_name=split,
        taxonomy=taxonomy,
        y_true=y_true,
        y_pred=trace["y_pred"],
        y_prob=trace["y_prob"],
        class_names=OP29_CLASSES if is_op29 else THIRTY_FOUR_CLASSES,
        minority_classes=OP29_MINORITY_CLASSES if is_op29 else MINORITY_ATTACK_CLASSES,
        flood_classes=OP29_FLOOD_CLASSES if is_op29 else FLOOD_CLASSES,
        non_flood_classes=OP29_NON_FLOOD_CLASSES if is_op29 else NON_FLOOD_CLASSES,
    )
    metrics = eval_res["summary"]
    routing = analyze_routing_and_override_outcomes(
        y_true=y_true,
        e4_pred=trace["e4_pred"],
        hybrid_pred=trace["y_pred"],
        taxonomy=taxonomy,
    )

    print(
        f"[{model.name}] split={split} (n={len(X):,}) | "
        f"tau=({model.tau_rare}, {model.tau_cls}, {model.tau_e4_keep}) | "
        f"Size={model.model_size_bytes() / (1024 * 1024):.2f} MB"
    )
    print(
        f"  Accuracy={metrics['accuracy']:.5f} | MacroF1={metrics['macro_f1']:.5f} | "
        f"MCC={metrics['mcc']:.5f} | MinorityF1={metrics['minority_f1']:.5f} | "
        f"MinRec={metrics['minority_recall']:.5f} | BenignFPR={metrics['benign_fpr']:.5f}"
    )
    diag = trace["diagnostics"]
    print(
        f"  Routing: FloodBypass={diag['n_e4_flood_bypass']:,}, Routed={diag['n_routed_to_specialist']:,}, "
        f"Overrides={diag['n_specialist_overrides']:,}, TrueRareLostInFlood={routing['true_rare_bypassed_by_e4_flood']}"
    )


def run_csv_inference(input_csv: Path, output_csv: Optional[Path], taxonomy: str) -> None:
    """Runs Strategy 1 preprocessing + Hybrid Specialist inference on raw flow CSVs."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    model, preprocessor = load_promoted_hybrid_ids(taxonomy=taxonomy, with_preprocessor=True)
    X_scaled = preprocess_raw_flows(df, preprocessor)
    trace = model.predict_with_trace(X_scaled)

    class_names = model.class_names
    preds = [class_names[int(i)] for i in trace["y_pred"]]
    e4_preds = [class_names[int(i)] for i in trace["e4_pred"]]
    probs = trace["y_prob"]
    confs = probs[np.arange(len(preds)), trace["y_pred"]]
    sb_top = [RARE_8_CLASSES[int(i)] for i in trace["stage_b_top_idx"]]

    out_df = pd.DataFrame({
        "pred_label": preds,
        "confidence": np.round(confs, 5),
        "e4_base_label": e4_preds,
        "route": np.where(trace["flood_bypass"], "E4_FLOOD_BYPASS", "SPECIALIST_NONFLOOD"),
        "stage_a_rare_prob": np.round(trace["stage_a_rare_prob"], 5),
        "stage_b_top_rare": sb_top,
        "override_applied": trace["override_applied"],
        "executions_per_flow": trace["executions_per_flow"],
    })
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Saved {len(out_df):,} predictions to {output_csv}")
    else:
        print(out_df.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promoted Main Track 2 Hybrid IDS Runner")
    parser.add_argument("--taxonomy", choices=["OP29", "ORIG34"], default="OP29")
    parser.add_argument("--split", choices=["val", "test", "cross"], default="test")
    parser.add_argument("--sample-n", type=int, default=None, help="Optional subsample for quick verification.")
    parser.add_argument("--input", type=Path, default=None, help="Raw CSV path with 39 canonical features.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path for --input mode.")
    args = parser.parse_args()

    if args.input is not None:
        run_csv_inference(args.input, args.output, taxonomy=args.taxonomy)
    else:
        run_split_verification(split=args.split, taxonomy=args.taxonomy, sample_n=args.sample_n)


if __name__ == "__main__":
    main()
