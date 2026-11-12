"""
Experiment 3 Helper: Zero-Flood Rare Non-Flood Specialist & Gated Hybrid IDS.

Architecture:
  - Stage A (Binary Detector): Rare-8 Non-Flood Attack (1) vs Benign + Common Non-Flood (0),
    trained strictly on Zero-Flood Training rows (161,533 rows: 9,108 pos vs 152,425 neg).
  - Stage B (8-Class Classifier): Classifies among the 8 rare Non-Flood minority classes,
    trained strictly on the 9,108 rare Non-Flood Training rows.
  - Hybrid Integration with frozen E4 Base IDS:
      * E4-predicted Flood flows remain untouched (bypass specialist).
      * E4-predicted Non-Flood (incl. BENIGN) flows are evaluated by Stage A + Stage B.
      * Gated override policy (tau_rare, tau_cls, tau_e4_keep) selects when specialist
        overrides E4, tuned strictly on Validation to maximize Minority F1 / Macro-F1
        without exploding Benign FPR.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from track1_dataset_evaluation.src.config import FEATURE_COLUMNS, THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import FLOOD_CLASSES, MINORITY_ATTACK_CLASSES
from track2_ids_engineering.minority_optimization.src.common import (
    OP29_BENIGN_IDX,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    PROJ_34_TO_29,
)

# Canonical ordered names matching Stage B output indices 0..7
RARE_8_CLASSES: List[str] = list(MINORITY_ATTACK_CLASSES)
RARE_8_SET = set(RARE_8_CLASSES)

# Target indices of the 8 rare classes inside ORIG34 and OP29 taxonomies
RARE_8_IN_34_IDX = np.array([THIRTY_FOUR_CLASSES.index(c) for c in RARE_8_CLASSES], dtype=np.int32)
RARE_8_IN_29_IDX = np.array([OP29_CLASSES.index(c) for c in RARE_8_CLASSES], dtype=np.int32)
FLOOD_34_IDX = np.array([THIRTY_FOUR_CLASSES.index(c) for c in FLOOD_CLASSES], dtype=np.int32)
FLOOD_29_IDX = np.array([OP29_CLASSES.index(c) for c in OP29_FLOOD_CLASSES], dtype=np.int32)
BENIGN_34_IDX = THIRTY_FOUR_CLASSES.index("BENIGN")


def build_zero_flood_specialist_train_pools(splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Constructs strict Zero-Flood Training & Validation pools for Stage A and Stage B.
    Asserts ZERO Flood attack rows exist in the specialist training pools.
    """
    tr = splits["train"]
    val = splits["val"]

    # Zero-Flood filter: keep Benign + all Non-Flood classes (~is_flood)
    tr_nf_mask = ~tr["is_flood"]
    val_nf_mask = ~val["is_flood"]

    # Verify strict 0 Flood records in Train specialist pool
    y_tr_nf_34 = tr["y_fine"][tr_nf_mask]
    assert not np.any(np.isin(y_tr_nf_34, FLOOD_34_IDX)), "LEAKAGE ERROR: Flood sample found in specialist train pool!"
    assert int(np.sum(tr_nf_mask)) == 161533, f"Expected 161,533 Zero-Flood Train rows, got {np.sum(tr_nf_mask)}"

    # Binary target: 1 iff one of the 8 rare minority classes
    tr_is_rare = np.isin(tr["y_fine"], RARE_8_IN_34_IDX)
    val_is_rare = np.isin(val["y_fine"], RARE_8_IN_34_IDX)

    tr_rare_mask = tr_nf_mask & tr_is_rare
    val_rare_mask = val_nf_mask & val_is_rare

    assert int(np.sum(tr_rare_mask)) == 9108, f"Expected 9,108 Rare Train rows, got {np.sum(tr_rare_mask)}"

    # Map global 34-class id -> local 0..7 index inside Stage B
    g34_to_rare8 = np.full(34, -1, dtype=np.int32)
    for local_i, g34_i in enumerate(RARE_8_IN_34_IDX):
        g34_to_rare8[g34_i] = local_i

    y_tr_stage_a = tr_is_rare[tr_nf_mask].astype(np.int32)
    y_val_nf_stage_a = val_is_rare[val_nf_mask].astype(np.int32)
    y_val_all_stage_a = val_is_rare.astype(np.int32)

    y_tr_stage_b = g34_to_rare8[tr["y_fine"][tr_rare_mask]]
    y_val_stage_b = g34_to_rare8[val["y_fine"][val_rare_mask]]
    assert np.all(y_tr_stage_b >= 0) and np.all(y_val_stage_b >= 0)

    return {
        "X_tr_stage_a": tr["X"][tr_nf_mask],
        "y_tr_stage_a": y_tr_stage_a,
        "X_val_nf_stage_a": val["X"][val_nf_mask],
        "y_val_nf_stage_a": y_val_nf_stage_a,
        "X_val_all": val["X"],
        "y_val_all_stage_a": y_val_all_stage_a,
        "X_tr_stage_b": tr["X"][tr_rare_mask],
        "y_tr_stage_b": y_tr_stage_b,
        "X_val_stage_b": val["X"][val_rare_mask],
        "y_val_stage_b": y_val_stage_b,
    }


def apply_gated_hybrid_policy(
    p_e4_tax: np.ndarray,
    p_stage_a_rare: np.ndarray,
    p_stage_b_8: np.ndarray,
    taxonomy: str,
    tau_rare: float,
    tau_cls: float,
    tau_e4_keep: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Combines frozen E4 probabilities (`ORIG34` or `OP29` remapped) with Stage A & Stage B
    specialist probabilities under threshold policy `(tau_rare, tau_cls, tau_e4_keep)`.

    Rules:
      1. Flows predicted by E4 as a Flood class bypass the specialist untouched.
      2. Flows predicted by E4 as BENIGN or any Non-Flood class are routed to Specialist:
         Override to Stage B's argmax rare class iff:
           - p_rare >= tau_rare, AND
           - max(p_stage_b) >= tau_cls, AND
           - (E4 already predicted a Rare-8 class OR max(P_e4) < tau_e4_keep)
    """
    if taxonomy == "ORIG34":
        flood_idx = FLOOD_34_IDX
        rare_in_tax_idx = RARE_8_IN_34_IDX
        n_cls = 34
    elif taxonomy == "OP29":
        flood_idx = FLOOD_29_IDX
        rare_in_tax_idx = RARE_8_IN_29_IDX
        n_cls = 29
    else:
        raise ValueError(f"Unsupported taxonomy: {taxonomy}")

    e4_pred = np.argmax(p_e4_tax, axis=1)
    e4_conf = p_e4_tax[np.arange(len(e4_pred)), e4_pred]

    is_e4_flood = np.isin(e4_pred, flood_idx)
    routed_mask = ~is_e4_flood
    e4_already_rare = np.isin(e4_pred, rare_in_tax_idx)

    b_local_best = np.argmax(p_stage_b_8, axis=1)
    b_conf = p_stage_b_8[np.arange(len(b_local_best)), b_local_best]
    b_tax_best = rare_in_tax_idx[b_local_best]

    override_mask = (
        routed_mask
        & (p_stage_a_rare >= tau_rare)
        & (b_conf >= tau_cls)
        & (e4_already_rare | (e4_conf < tau_e4_keep))
    )

    y_pred = e4_pred.copy()
    y_pred[override_mask] = b_tax_best[override_mask]

    # Calibrated simplex probability fusion for routed samples (for ROC-AUC / AP consistency)
    p_hybrid = p_e4_tax.astype(np.float64).copy()
    if np.any(override_mask):
        ov_idx = np.where(override_mask)[0]
        p_r = np.clip(p_stage_a_rare[ov_idx], 1e-6, 1.0 - 1e-6)[:, None]
        # Redistribute (1 - p_r) over non-rare classes proportional to E4 mass, and p_r over 8 rare classes via Stage B
        non_rare_mask = np.ones(n_cls, dtype=bool)
        non_rare_mask[rare_in_tax_idx] = False
        e4_nr = p_hybrid[ov_idx][:, non_rare_mask]
        nr_sum = np.sum(e4_nr, axis=1, keepdims=True)
        e4_nr_norm = np.where(nr_sum > 1e-9, e4_nr / np.maximum(nr_sum, 1e-9), 1.0 / np.sum(non_rare_mask))

        # Place Stage B weighted mass into rare slots and remaining mass into non-rare slots
        p_new = np.zeros_like(p_hybrid[ov_idx])
        p_new[:, non_rare_mask] = (1.0 - p_r) * e4_nr_norm
        p_new[:, rare_in_tax_idx] = p_r * p_stage_b_8[ov_idx]
        p_hybrid[ov_idx] = p_new

    diagnostics = {
        "n_total": int(len(e4_pred)),
        "n_e4_flood_bypass": int(np.sum(is_e4_flood)),
        "n_routed_to_specialist": int(np.sum(routed_mask)),
        "n_specialist_overrides": int(np.sum(override_mask)),
        "override_rate_of_routed": round(float(np.sum(override_mask) / max(1, np.sum(routed_mask))), 5),
    }
    return y_pred, p_hybrid.astype(np.float32), diagnostics


def analyze_routing_and_override_outcomes(
    y_true: np.ndarray,
    e4_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    taxonomy: str,
) -> Dict[str, int]:
    """
    Quantifies routing loss (Rare/Non-Flood erroneously bypassed as Flood by E4)
    and exact breakdown of Specialist overrides (beneficial vs Benign FP vs Common-to-Rare).
    """
    if taxonomy == "ORIG34":
        flood_idx = FLOOD_34_IDX
        rare_idx = RARE_8_IN_34_IDX
        benign_idx = BENIGN_34_IDX
    else:
        flood_idx = FLOOD_29_IDX
        rare_idx = RARE_8_IN_29_IDX
        benign_idx = OP29_BENIGN_IDX

    true_rare = np.isin(y_true, rare_idx)
    true_benign = y_true == benign_idx
    true_common_nf = (~true_rare) & (~true_benign) & (~np.isin(y_true, flood_idx))
    e4_flood = np.isin(e4_pred, flood_idx)

    changed = hybrid_pred != e4_pred
    e4_right = e4_pred == y_true
    hyb_right = hybrid_pred == y_true

    return {
        "true_rare_total": int(np.sum(true_rare)),
        "true_rare_bypassed_by_e4_flood": int(np.sum(true_rare & e4_flood)),
        "true_rare_reached_specialist": int(np.sum(true_rare & (~e4_flood))),
        "overrides_total": int(np.sum(changed)),
        "overrides_beneficial_wrong_to_right": int(np.sum(changed & (~e4_right) & hyb_right)),
        "overrides_harmful_right_to_wrong": int(np.sum(changed & e4_right & (~hyb_right))),
        "overrides_benign_turned_into_rare_fp": int(np.sum(changed & true_benign & (e4_pred == benign_idx))),
        "overrides_common_nf_turned_into_rare": int(np.sum(changed & true_common_nf)),
        "overrides_true_rare_reclassified_to_right": int(np.sum(changed & true_rare & hyb_right)),
        "overrides_true_rare_reclassified_to_wrong": int(np.sum(changed & true_rare & e4_right & (~hyb_right))),
    }


class HybridE4SpecialistAdapter:
    """
    Deployable end-to-end E4 + Zero-Flood Specialist wrapper implementing `.predict`,
    `.predict_proba`, and `.model_size_bytes` for unified latency & accuracy benchmarking.
    """

    def __init__(
        self,
        name: str,
        e4_model: Any,
        stage_a_model: Any,
        stage_b_model: Any,
        taxonomy: str,
        tau_rare: float,
        tau_cls: float,
        tau_e4_keep: float,
        checkpoint_paths: List[Path],
    ) -> None:
        self.name = name
        self.e4_model = e4_model
        self.stage_a_model = stage_a_model
        self.stage_b_model = stage_b_model
        self.taxonomy = taxonomy
        self.tau_rare = float(tau_rare)
        self.tau_cls = float(tau_cls)
        self.tau_e4_keep = float(tau_e4_keep)
        self.checkpoint_paths = checkpoint_paths
        self.flood_idx = FLOOD_34_IDX if taxonomy == "ORIG34" else FLOOD_29_IDX

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p_e4_34 = self.e4_model.predict_proba(X)
        p_e4_tax = p_e4_34 if self.taxonomy == "ORIG34" else (p_e4_34 @ PROJ_34_TO_29)
        e4_pred = np.argmax(p_e4_tax, axis=1)
        routed_mask = ~np.isin(e4_pred, self.flood_idx)

        # Only run Stage A / Stage B on Non-Flood routed flows (true sequential cascade savings!)
        p_a = np.zeros(len(X), dtype=np.float32)
        p_b = np.zeros((len(X), 8), dtype=np.float32)
        if np.any(routed_mask):
            X_r = X[routed_mask]
            p_a_r = self.stage_a_model.predict_proba(X_r)[:, 1]
            p_a[routed_mask] = p_a_r

            # Only invoke Stage B when Stage A crosses tau_rare
            cand_sub = p_a_r >= self.tau_rare
            if np.any(cand_sub):
                routed_indices = np.where(routed_mask)[0]
                cand_indices = routed_indices[cand_sub]
                p_b[cand_indices] = self.stage_b_model.predict_proba(X[cand_indices])

        return apply_gated_hybrid_policy(
            p_e4_tax=p_e4_tax,
            p_stage_a_rare=p_a,
            p_stage_b_8=p_b,
            taxonomy=self.taxonomy,
            tau_rare=self.tau_rare,
            tau_cls=self.tau_cls,
            tau_e4_keep=self.tau_e4_keep,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred, _, _ = self._forward(X)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p_hyb, _ = self._forward(X)
        return p_hyb

    def model_size_bytes(self) -> int:
        # Sum unique checkpoint sizes on disk (E4 + Stage A + Stage B)
        unique_paths = {Path(p).resolve() for p in self.checkpoint_paths}
        return sum(int(p.stat().st_size) for p in unique_paths if p.exists())
