"""
Experiment 1 Helper: Feature-Space Realism Audit & Controlled Train-Only SMOTE.

All nearest-neighbor queries and synthetic sample generation operate strictly on
the Training partition (350,821 rows). Validation, Internal Test, and Cross-Capture
data are never touched here.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from track1_dataset_evaluation.src.config import FEATURE_COLUMNS, RANDOM_SEED, THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import MINORITY_ATTACK_CLASSES


def _batched_topk_full_train(
    X_queries: np.ndarray,
    X_ref: np.ndarray,
    k: int = 5,
    exclude_exact_self_idx: np.ndarray | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Finds exact top-k Euclidean nearest neighbors in `X_ref` for `X_queries`
    via float32 GEMM distance computation. Fast (~1-2s) and bounded memory on ~350k rows.
    """
    ref_t = np.ascontiguousarray(X_ref.astype(np.float32).T)
    ref_sq = np.sum(X_ref.astype(np.float32) ** 2, axis=1)  # (N_ref,)
    n_q = len(X_queries)
    out_idx = np.empty((n_q, k), dtype=np.int32)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        Q = X_queries[start:end].astype(np.float32)
        q_sq = np.sum(Q ** 2, axis=1, keepdims=True)
        # Squared Euclidean distance matrix (B, N_ref)
        dists = q_sq + ref_sq[None, :] - 2.0 * (Q @ ref_t)
        if exclude_exact_self_idx is not None:
            rows = np.arange(end - start)
            dists[rows, exclude_exact_self_idx[start:end]] = np.inf
        part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row_ids = np.arange(end - start)[:, None]
        part_d = dists[row_ids, part]
        order = np.argsort(part_d, axis=1)
        out_idx[start:end] = part[row_ids, order]

    return out_idx


def _detect_integer_feature_mask(X_train_raw: np.ndarray) -> np.ndarray:
    """
    Identifies columns in Train raw space that are strictly integer-valued
    (accounting for float32 StandardScaler round-trip precision ~5e-6 * |x|).
    """
    tol = 1e-3 + 5e-6 * np.abs(X_train_raw)
    return np.all(np.abs(X_train_raw - np.round(X_train_raw)) <= tol, axis=0)


def _chawla_smote_from_seeds(
    X_seeds: np.ndarray,
    X_pool: np.ndarray,
    n_to_generate: int,
    k: int,
    rng: np.random.Generator,
    allow_self_match: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates `n_to_generate` synthetic points by interpolating each chosen seed in
    `X_seeds` toward one of its k-NN in `X_pool`.
    Returns `(X_syn, picked_seed_idx_in_seeds, picked_nbr_idx_in_pool)`.
    """
    n_seeds = len(X_seeds)
    n_pool = len(X_pool)
    if n_to_generate <= 0 or n_seeds < 2 or n_pool < 2:
        return np.empty((0, X_seeds.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    k_eff = min(k, n_pool - (0 if allow_self_match else 1))
    if k_eff < 1:
        return np.empty((0, X_seeds.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=k_eff + (0 if allow_self_match else 1), metric="euclidean")
    nn.fit(X_pool)
    nbrs = nn.kneighbors(X_seeds, return_distance=False)
    if not allow_self_match:
        # Remove exact 0-index self match assuming X_seeds rows appear inside X_pool
        nbrs = nbrs[:, 1 : 1 + k_eff]
    else:
        nbrs = nbrs[:, :k_eff]

    seed_choices = rng.integers(0, n_seeds, size=n_to_generate)
    nbr_slot = rng.integers(0, k_eff, size=n_to_generate)
    nbr_pool_idx = nbrs[seed_choices, nbr_slot]
    alpha = rng.uniform(0.0, 1.0, size=(n_to_generate, 1)).astype(np.float32)

    X_base = X_seeds[seed_choices].astype(np.float32)
    X_tgt = X_pool[nbr_pool_idx].astype(np.float32)
    X_syn = X_base + alpha * (X_tgt - X_base)
    return X_syn, seed_choices.astype(np.int32), nbr_pool_idx.astype(np.int32)


def _repair_raw_invariants(
    X_syn_raw: np.ndarray,
    int_col_mask: np.ndarray,
    col_idx: Dict[str, int],
) -> np.ndarray:
    """
    Projects synthetic feature rows back onto observable CICIoT2023 feature invariants:
      - Clamps non-negative counters/metrics at >= 0
      - Rounds discrete integer features (Protocol Type, packet counts, header lengths, flags)
      - Re-enforces exact Variance = Std^2, Tot size = AVG, and Min <= AVG <= Max
    """
    X_rep = X_syn_raw.copy().astype(np.float64)
    X_rep = np.maximum(X_rep, 0.0)

    # Round strictly-integer columns observed on Train
    X_rep[:, int_col_mask] = np.round(X_rep[:, int_col_mask])

    # Keep Tot size consistent with AVG (exact identity on raw CICIoT2023)
    i_avg = col_idx["AVG"]
    i_tot_size = col_idx["Tot size"]
    i_std = col_idx["Std"]
    i_var = col_idx["Variance"]
    i_min = col_idx["Min"]
    i_max = col_idx["Max"]

    X_rep[:, i_min] = np.minimum(X_rep[:, i_min], X_rep[:, i_avg])
    X_rep[:, i_max] = np.maximum(X_rep[:, i_max], X_rep[:, i_avg])
    X_rep[:, i_tot_size] = X_rep[:, i_avg]
    X_rep[:, i_var] = X_rep[:, i_std] ** 2

    return X_rep


def audit_smote_realism(
    X_train_scaled: np.ndarray,
    y_train_34: np.ndarray,
    prep: Any,
    target_cap: int = 4000,
    mult_cap: int = 8,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Audits how standard Chawla SMOTE behaves on CICIoT2023 training data across all 8
    minority classes vs protocol-stratified & invariant-repaired SMOTE.
    Strictly uses Train split only.
    """
    rng = np.random.default_rng(random_state)
    col_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    X_train_raw = prep.scaler.inverse_transform(X_train_scaled.astype(np.float64))
    int_col_mask = _detect_integer_feature_mask(X_train_raw)

    i_proto = col_idx["Protocol Type"]
    i_std = col_idx["Std"]
    i_var = col_idx["Variance"]
    i_tcp = col_idx["TCP"]
    i_udp = col_idx["UDP"]
    benign_idx = THIRTY_FOUR_CLASSES.index("BENIGN")

    # Precompute Train integer protocol labels per sample
    train_proto = np.round(X_train_raw[:, i_proto]).astype(np.int32)
    valid_protocols = np.unique(train_proto)

    rows = []
    for cls_name in MINORITY_ATTACK_CLASSES:
        cls_id = THIRTY_FOUR_CLASSES.index(cls_name)
        cls_train_idx = np.where(y_train_34 == cls_id)[0]
        n_c = len(cls_train_idx)

        # 1. Real neighborhood overlap in full 350,821 Train rows
        real_5nn = _batched_topk_full_train(
            X_train_scaled[cls_train_idx], X_train_scaled, k=5, exclude_exact_self_idx=cls_train_idx
        )
        real_5nn_labels = y_train_34[real_5nn]
        same_frac = np.mean(real_5nn_labels == cls_id, axis=1)
        benign_frac = np.mean(real_5nn_labels == benign_idx, axis=1)

        # 2. Generate Standard SMOTE synthetic batch (for auditing violations)
        n_target = min(target_cap, mult_cap * n_c)
        n_syn = max(0, n_target - n_c)
        X_cls_scaled = X_train_scaled[cls_train_idx]
        X_syn_std, pick_seed, pick_nbr = _chawla_smote_from_seeds(
            X_cls_scaled, X_cls_scaled, n_to_generate=n_syn, k=5, rng=rng, allow_self_match=False
        )
        X_syn_std_raw = prep.scaler.inverse_transform(X_syn_std.astype(np.float64))

        # Protocol mixing checks
        seed_proto = train_proto[cls_train_idx[pick_seed]]
        nbr_proto = train_proto[cls_train_idx[pick_nbr]]
        proto_pair_mismatch_pct = float(np.mean(seed_proto != nbr_proto) * 100.0) if len(X_syn_std) else 0.0

        syn_proto_vals = X_syn_std_raw[:, i_proto]
        non_int_proto_pct = (
            float(np.mean(np.min(np.abs(syn_proto_vals[:, None] - valid_protocols[None, :]), axis=1) > 0.05) * 100.0)
            if len(X_syn_std)
            else 0.0
        )

        # Variance vs Std^2 mismatch (relative error on Std^2 > 1e-3)
        var_pred = X_syn_std_raw[:, i_std] ** 2
        var_interp = X_syn_std_raw[:, i_var]
        rel_var_err = np.abs(var_interp - var_pred) / np.maximum(var_pred, 1e-3)
        var_violation_pct = float(np.mean(rel_var_err > 0.01) * 100.0)

        # Strictly-integer columns turning fractional (beyond float32 scaler tolerance)
        if np.any(int_col_mask):
            int_vals = X_syn_std_raw[:, int_col_mask]
            int_tol = 0.02 + 5e-6 * np.abs(int_vals)
            int_frac_violations = np.abs(int_vals - np.round(int_vals)) > int_tol
            syn_with_any_frac_int_pct = float(np.mean(np.any(int_frac_violations, axis=1)) * 100.0)
        else:
            syn_with_any_frac_int_pct = 0.0

        # Conflicting TCP + UDP transport indicators or TCP flag vs UDP proto
        tcp_udp_conflict_pct = float(
            np.mean((X_syn_std_raw[:, i_tcp] > 0.05) & (X_syn_std_raw[:, i_udp] > 0.05)) * 100.0
        )

        # 3. Full-Train 1-NN of Standard SMOTE vs ProtoSafe SMOTE
        syn_std_1nn = _batched_topk_full_train(X_syn_std, X_train_scaled, k=1)[:, 0]
        syn_std_1nn_labels = y_train_34[syn_std_1nn]
        std_1nn_wrong_cls_pct = float(np.mean(syn_std_1nn_labels != cls_id) * 100.0)
        std_1nn_benign_pct = float(np.mean(syn_std_1nn_labels == benign_idx) * 100.0)

        # 4. Generate ProtoSafe sample (within-protocol >= 10 strata + invariant repair)
        X_syn_ps = _generate_class_smote_protosafe(
            X_cls_scaled=X_cls_scaled,
            proto_cls=train_proto[cls_train_idx],
            n_syn=n_syn,
            int_col_mask=int_col_mask,
            col_idx=col_idx,
            prep=prep,
            safe_seed_mask=None,
            rng=rng,
        )
        if len(X_syn_ps) > 0:
            syn_ps_1nn = _batched_topk_full_train(X_syn_ps, X_train_scaled, k=1)[:, 0]
            syn_ps_1nn_labels = y_train_34[syn_ps_1nn]
            ps_1nn_wrong_cls_pct = float(np.mean(syn_ps_1nn_labels != cls_id) * 100.0)
            ps_1nn_benign_pct = float(np.mean(syn_ps_1nn_labels == benign_idx) * 100.0)
        else:
            ps_1nn_wrong_cls_pct = 0.0
            ps_1nn_benign_pct = 0.0

        # Protocol breakdown string (e.g. "6:280|17:28|1:4")
        u_p, u_c = np.unique(train_proto[cls_train_idx], return_counts=True)
        proto_dist = "|".join(f"{int(p)}:{int(c)}" for p, c in sorted(zip(u_p, u_c), key=lambda x: -x[1]))

        rows.append(
            {
                "class_34": cls_name,
                "train_count": int(n_c),
                "n_synthetic_std": int(n_syn),
                "n_synthetic_protosafe": int(len(X_syn_ps)),
                "protocol_breakdown": proto_dist,
                "real_5nn_same_class_mean": round(float(np.mean(same_frac)), 4),
                "real_pct_seeds_ge1_same_5nn": round(float(np.mean(same_frac > 0.0) * 100.0), 2),
                "real_pct_seeds_majority_benign_5nn": round(float(np.mean(benign_frac > 0.5) * 100.0), 2),
                "std_proto_pair_mismatch_pct": round(proto_pair_mismatch_pct, 2),
                "std_invalid_nonint_proto_pct": round(non_int_proto_pct, 2),
                "std_variance_identity_err_pct": round(var_violation_pct, 2),
                "std_any_int_col_fractional_pct": round(syn_with_any_frac_int_pct, 2),
                "std_tcp_udp_dual_active_pct": round(tcp_udp_conflict_pct, 2),
                "std_syn_1nn_wrong_class_pct": round(std_1nn_wrong_cls_pct, 2),
                "std_syn_1nn_benign_pct": round(std_1nn_benign_pct, 2),
                "protosafe_syn_1nn_wrong_class_pct": round(ps_1nn_wrong_cls_pct, 2),
                "protosafe_syn_1nn_benign_pct": round(ps_1nn_benign_pct, 2),
            }
        )

    return pd.DataFrame(rows)


def _generate_class_smote_protosafe(
    X_cls_scaled: np.ndarray,
    proto_cls: np.ndarray,
    n_syn: int,
    int_col_mask: np.ndarray,
    col_idx: Dict[str, int],
    prep: Any,
    safe_seed_mask: np.ndarray | None,
    rng: np.random.Generator,
    min_stratum_size: int = 10,
) -> np.ndarray:
    """
    Generates protocol-stratified synthetic points inside valid protocol strata (`>= 10` samples),
    repairs physical CICIoT2023 feature invariants in raw space, and re-scales using `prep.scaler`.
    """
    if n_syn <= 0:
        return np.empty((0, X_cls_scaled.shape[1]), dtype=np.float32)

    u_protos, u_counts = np.unique(proto_cls, return_counts=True)
    eligible = [(int(p), int(c)) for p, c in zip(u_protos, u_counts) if c >= min_stratum_size]
    if not eligible:
        return np.empty((0, X_cls_scaled.shape[1]), dtype=np.float32)

    if safe_seed_mask is not None:
        # Allocate proportionally across eligible strata weighted by safe seed count
        elig_weights = []
        for p, c in eligible:
            n_safe = int(np.sum((proto_cls == p) & safe_seed_mask))
            elig_weights.append((p, n_safe))
        tot_safe = sum(w for _, w in elig_weights)
        if tot_safe == 0:
            return np.empty((0, X_cls_scaled.shape[1]), dtype=np.float32)
        counts_per_stratum = [(p, int(round(n_syn * w / tot_safe))) for p, w in elig_weights if w >= 2]
    else:
        tot_c = sum(c for _, c in eligible)
        counts_per_stratum = [(p, int(round(n_syn * c / tot_c))) for p, c in eligible]

    syn_chunks = []
    for p, n_gen in counts_per_stratum:
        if n_gen <= 0:
            continue
        p_idx = np.where(proto_cls == p)[0]
        X_pool = X_cls_scaled[p_idx]
        if safe_seed_mask is not None:
            seed_sub = np.where((proto_cls == p) & safe_seed_mask)[0]
            if len(seed_sub) < 2:
                continue
            X_seeds = X_cls_scaled[seed_sub]
            # X_seeds is a subset of X_pool -> use allow_self_match=False cleanly via NN(k+1) vs X_pool, skipping 0-dist matches
            k_eff = min(5, len(X_pool) - 1)
            nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X_pool)
            dists, nbrs = nn.kneighbors(X_seeds, return_distance=True)
            # Drop self (col 0 where distance ~ 0)
            nbrs_clean = np.empty((len(X_seeds), k_eff), dtype=np.int32)
            for row_i in range(len(X_seeds)):
                # Keep indices where dist > 1e-7 or skip col 0
                cand = [j for j, d in zip(nbrs[row_i], dists[row_i]) if d > 1e-7]
                if len(cand) < k_eff:
                    cand = list(nbrs[row_i, 1 : 1 + k_eff])
                nbrs_clean[row_i] = cand[:k_eff]
            sc = rng.integers(0, len(X_seeds), size=n_gen)
            ns = rng.integers(0, k_eff, size=n_gen)
            alpha = rng.uniform(0.0, 1.0, size=(n_gen, 1)).astype(np.float32)
            X_b = X_seeds[sc].astype(np.float32)
            X_t = X_pool[nbrs_clean[sc, ns]].astype(np.float32)
            X_s = X_b + alpha * (X_t - X_b)
        else:
            X_s, _, _ = _chawla_smote_from_seeds(
                X_pool, X_pool, n_to_generate=n_gen, k=5, rng=rng, allow_self_match=False
            )
        if len(X_s) > 0:
            syn_chunks.append(X_s)

    if not syn_chunks:
        return np.empty((0, X_cls_scaled.shape[1]), dtype=np.float32)

    X_syn_scaled = np.vstack(syn_chunks)
    X_syn_raw = prep.scaler.inverse_transform(X_syn_scaled.astype(np.float64))
    X_syn_raw_rep = _repair_raw_invariants(X_syn_raw, int_col_mask, col_idx)
    return prep.scaler.transform(X_syn_raw_rep).astype(np.float32)


def generate_smote_train_dataset(
    X_train_scaled: np.ndarray,
    y_train_34: np.ndarray,
    y_train_target: np.ndarray,
    mode: str,
    prep: Any,
    target_names: List[str],
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies strictly Train-only SMOTE under the specified mode:
      - 'SMOTE_Std_Cap4000'
      - 'SMOTE_ProtoSafe_Cap4000'
      - 'SMOTE_ProtoSafe_Cap8000'
      - 'SMOTE_ProtoSafe_Selective'

    Note: `y_train_34` is used ONLY to pick each of the 8 minority classes uniformly
    so both ORIG34 and OP29 receive identical synthetic rows; returned `y_aug` uses `y_train_target`.
    """
    rng = np.random.default_rng(random_state)
    col_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    X_train_raw = prep.scaler.inverse_transform(X_train_scaled.astype(np.float64))
    int_col_mask = _detect_integer_feature_mask(X_train_raw)
    i_proto = col_idx["Protocol Type"]
    train_proto = np.round(X_train_raw[:, i_proto]).astype(np.int32)

    X_extra_list: List[np.ndarray] = []
    y_extra_list: List[np.ndarray] = []
    per_class_generated: Dict[str, int] = {}
    skipped_classes: List[str] = []

    for cls_name in MINORITY_ATTACK_CLASSES:
        cls_34_id = THIRTY_FOUR_CLASSES.index(cls_name)
        cls_idx = np.where(y_train_34 == cls_34_id)[0]
        n_c = len(cls_idx)
        target_cls_id = target_names.index(cls_name)

        if mode == "SMOTE_Std_Cap4000":
            n_target = min(4000, 8 * n_c)
            n_syn = max(0, n_target - n_c)
            X_syn, _, _ = _chawla_smote_from_seeds(
                X_train_scaled[cls_idx], X_train_scaled[cls_idx], n_to_generate=n_syn, k=5, rng=rng, allow_self_match=False
            )
        elif mode in ("SMOTE_ProtoSafe_Cap4000", "SMOTE_ProtoSafe_Cap8000"):
            cap, mult = (4000, 8) if mode.endswith("4000") else (8000, 10)
            n_target = min(cap, mult * n_c)
            n_syn = max(0, n_target - n_c)
            X_syn = _generate_class_smote_protosafe(
                X_cls_scaled=X_train_scaled[cls_idx],
                proto_cls=train_proto[cls_idx],
                n_syn=n_syn,
                int_col_mask=int_col_mask,
                col_idx=col_idx,
                prep=prep,
                safe_seed_mask=None,
                rng=rng,
            )
        elif mode == "SMOTE_ProtoSafe_Selective":
            # Full-Train 5-NN purity check (Train only, cached per class for ORIG34/OP29 reuse)
            from track2_ids_engineering.minority_optimization.src.common import MODELS_DIR
            cache_dir = MODELS_DIR / "exp1_smote"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mask_cache = cache_dir / f"safe_seed_5nn_{cls_name}.npy"
            if mask_cache.exists():
                has_same_5nn = np.load(mask_cache)
            else:
                real_5nn = _batched_topk_full_train(
                    X_train_scaled[cls_idx], X_train_scaled, k=5, exclude_exact_self_idx=cls_idx
                )
                has_same_5nn = np.any(y_train_34[real_5nn] == cls_34_id, axis=1)
                np.save(mask_cache, has_same_5nn)
            pct_any_same = float(np.mean(has_same_5nn) * 100.0)
            if pct_any_same < 15.0:
                skipped_classes.append(f"{cls_name} (same_5nn={pct_any_same:.1f}% < 15%)")
                per_class_generated[cls_name] = 0
                continue
            n_target = min(6000, 8 * n_c)
            n_syn = max(0, n_target - n_c)
            X_syn = _generate_class_smote_protosafe(
                X_cls_scaled=X_train_scaled[cls_idx],
                proto_cls=train_proto[cls_idx],
                n_syn=n_syn,
                int_col_mask=int_col_mask,
                col_idx=col_idx,
                prep=prep,
                safe_seed_mask=has_same_5nn,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown SMOTE mode: {mode}")

        per_class_generated[cls_name] = int(len(X_syn))
        if len(X_syn) > 0:
            X_extra_list.append(X_syn.astype(np.float32))
            y_extra_list.append(np.full(len(X_syn), target_cls_id, dtype=y_train_target.dtype))

    if X_extra_list:
        X_aug = np.vstack([X_train_scaled.astype(np.float32)] + X_extra_list)
        y_aug = np.concatenate([y_train_target] + y_extra_list)
    else:
        X_aug = X_train_scaled.astype(np.float32)
        y_aug = y_train_target.copy()

    # Shuffle augmented training rows deterministically
    perm = rng.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    meta = {
        "mode": mode,
        "train_rows_before": int(len(X_train_scaled)),
        "synthetic_rows_added": int(len(X_aug) - len(X_train_scaled)),
        "train_rows_after": int(len(X_aug)),
        "per_class_generated": per_class_generated,
        "skipped_classes": skipped_classes,
    }
    return X_aug, y_aug, meta
