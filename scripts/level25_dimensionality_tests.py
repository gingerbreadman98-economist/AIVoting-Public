#!/usr/bin/env python3
"""
Dimensionality tests for the help/hurt/neutral structure of Absolute
Allocation votes (hidden-reference run).

Tests whether the decodable evaluative signal is one bipolar axis or more:

1. score_dimensionality: can the help probe's 1-D cross-validated score carry
   all decodable hurt information (and vice versa)? Robust to coefficient
   noise, unlike single-direction projection removal.
2. neutral_decodability: is zero-allocation (abstention) linearly decodable at
   all, from the full space and from the help/hurt scores?
3. neutral_geometry: split-half-corrected cosine between help-vs-neutral and
   hurt-vs-neutral directions. Note: full-data (non-split) cosines are inflated
   by shared training rows; only cross-half values are interpretable.

Usage:
  python level25_dimensionality_tests.py --mech-output-dir <dir> [--layer 16]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--probe-c", type=float, default=0.1)
    parser.add_argument("--direction-c", type=float, default=0.01)
    parser.add_argument("--split-half-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_inputs(base: Path, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    npz = np.load(base / "self_answer_activations.npz")
    layers = npz["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} not in saved layers: {layers}")
    X = npz["candidate_activations"][:, layers.index(layer), :].astype(np.float64)
    labels = pd.read_csv(base / "self_answer_vote_labels.csv")
    row_index = pd.read_csv(base / "activation_row_index.csv")
    labels = row_index.merge(
        labels,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    return X, labels


def bool_column(labels: pd.DataFrame, column: str) -> np.ndarray:
    return labels[column].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()


def logreg(c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c, solver="lbfgs", class_weight="balanced", max_iter=3000, random_state=seed
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def score_dimensionality(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    help_y = bool_column(labels, "voter_help_label")
    hurt_y = bool_column(labels, "voter_hurt_label")
    neutral_y = (labels["voter_allocation"].astype(float).abs() < 1e-12).astype(int).to_numpy()
    groups = labels["prompt_id"].astype(str).to_numpy()
    tests: dict[str, list[float]] = {
        "hurt_from_full": [],
        "hurt_from_help_score_only": [],
        "help_from_full": [],
        "help_from_hurt_score_only": [],
        "neutral_from_full": [],
        "neutral_from_help_score_only": [],
        "neutral_from_helphurt_scores": [],
    }
    for train_idx, test_idx in GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        m_help = logreg(args.probe_c, args.seed).fit(X_tr, help_y[train_idx])
        m_hurt = logreg(args.probe_c, args.seed).fit(X_tr, hurt_y[train_idx])
        s_help_tr = m_help.decision_function(X_tr).reshape(-1, 1)
        s_help_te = m_help.decision_function(X_te).reshape(-1, 1)
        s_hurt_tr = m_hurt.decision_function(X_tr).reshape(-1, 1)
        s_hurt_te = m_hurt.decision_function(X_te).reshape(-1, 1)
        tests["hurt_from_full"].append(
            roc_auc_score(hurt_y[test_idx], m_hurt.predict_proba(X_te)[:, 1])
        )
        tests["help_from_full"].append(
            roc_auc_score(help_y[test_idx], m_help.predict_proba(X_te)[:, 1])
        )
        m = logreg(1.0, args.seed).fit(s_help_tr, hurt_y[train_idx])
        tests["hurt_from_help_score_only"].append(
            roc_auc_score(hurt_y[test_idx], m.predict_proba(s_help_te)[:, 1])
        )
        m = logreg(1.0, args.seed).fit(s_hurt_tr, help_y[train_idx])
        tests["help_from_hurt_score_only"].append(
            roc_auc_score(help_y[test_idx], m.predict_proba(s_hurt_te)[:, 1])
        )
        m_neutral = logreg(args.probe_c, args.seed).fit(X_tr, neutral_y[train_idx])
        tests["neutral_from_full"].append(
            roc_auc_score(neutral_y[test_idx], m_neutral.predict_proba(X_te)[:, 1])
        )
        m = logreg(1.0, args.seed).fit(s_help_tr, neutral_y[train_idx])
        tests["neutral_from_help_score_only"].append(
            roc_auc_score(neutral_y[test_idx], m.predict_proba(s_help_te)[:, 1])
        )
        both_tr = np.hstack([s_help_tr, s_hurt_tr])
        both_te = np.hstack([s_help_te, s_hurt_te])
        m = logreg(1.0, args.seed).fit(both_tr, neutral_y[train_idx])
        tests["neutral_from_helphurt_scores"].append(
            roc_auc_score(neutral_y[test_idx], m.predict_proba(both_te)[:, 1])
        )
    return pd.DataFrame(
        [
            {"test": name, "mean_auc": float(np.mean(v)), "std_auc": float(np.std(v)), "n_folds": len(v)}
            for name, v in tests.items()
        ]
    )


def fit_direction(X: np.ndarray, y: np.ndarray, c: float, seed: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), logreg(c, seed))
    model.fit(X, y)
    return model[-1].coef_.reshape(-1)


def neutral_geometry(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    help_y = bool_column(labels, "voter_help_label")
    hurt_y = bool_column(labels, "voter_hurt_label")
    neutral = labels["voter_allocation"].astype(float).abs() < 1e-12
    mask_hn = (help_y == 1) | neutral.to_numpy()
    mask_tn = (hurt_y == 1) | neutral.to_numpy()
    prompts = labels["prompt_id"].astype(str).to_numpy()
    unique_prompts = np.unique(prompts)
    d_hn = fit_direction(X[mask_hn], help_y[mask_hn], args.direction_c, args.seed)
    d_tn = fit_direction(X[mask_tn], hurt_y[mask_tn], args.direction_c, args.seed)
    raw = cosine(d_hn, d_tn)
    rng = np.random.default_rng(args.seed)
    rel_hn, rel_tn, cross = [], [], []
    for repeat in range(args.split_half_repeats):
        perm = rng.permutation(unique_prompts)
        half_a = set(perm[: len(perm) // 2])
        in_a = np.isin(prompts, list(half_a))
        a1, b1 = mask_hn & in_a, mask_hn & ~in_a
        a2, b2 = mask_tn & in_a, mask_tn & ~in_a
        h_a = fit_direction(X[a1], help_y[a1], args.direction_c, args.seed + repeat)
        h_b = fit_direction(X[b1], help_y[b1], args.direction_c, args.seed + repeat)
        t_a = fit_direction(X[a2], hurt_y[a2], args.direction_c, args.seed + repeat)
        t_b = fit_direction(X[b2], hurt_y[b2], args.direction_c, args.seed + repeat)
        rel_hn.append(cosine(h_a, h_b))
        rel_tn.append(cosine(t_a, t_b))
        cross.append(0.5 * (cosine(h_a, t_b) + cosine(h_b, t_a)))
    rel_hn_arr, rel_tn_arr, cross_arr = map(np.array, (rel_hn, rel_tn, cross))
    valid = (rel_hn_arr > 0) & (rel_tn_arr > 0)
    corrected = cross_arr[valid] / np.sqrt(rel_hn_arr[valid] * rel_tn_arr[valid])
    return pd.DataFrame(
        [
            {
                "raw_full_data_cosine_hn_tn": raw,
                "note_raw": "inflated by shared neutral rows; use cross-half values",
                "mean_rel_help_vs_neutral": float(rel_hn_arr.mean()),
                "std_rel_help_vs_neutral": float(rel_hn_arr.std()),
                "mean_rel_hurt_vs_neutral": float(rel_tn_arr.mean()),
                "std_rel_hurt_vs_neutral": float(rel_tn_arr.std()),
                "mean_cross_half_cosine": float(cross_arr.mean()),
                "std_cross_half_cosine": float(cross_arr.std()),
                "mean_corrected_cosine": float(corrected.mean()) if len(corrected) else np.nan,
                "std_corrected_cosine": float(corrected.std()) if len(corrected) else np.nan,
                "n_valid_corrections": int(valid.sum()),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"dimensionality_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    X, labels = load_inputs(base, args.layer)

    score_summary = score_dimensionality(X, labels, args)
    score_summary.to_csv(out_dir / "score_dimensionality.csv", index=False)
    geometry_summary = neutral_geometry(X, labels, args)
    geometry_summary.to_csv(out_dir / "neutral_geometry.csv", index=False)
    pd.DataFrame(
        [
            {
                "mech_output_dir": str(base),
                "layer": args.layer,
                "probe_c": args.probe_c,
                "direction_c": args.direction_c,
                "split_half_repeats": args.split_half_repeats,
                "seed": args.seed,
            }
        ]
    ).to_csv(out_dir / "run_config.csv", index=False)

    print("Score dimensionality (1-D sufficiency) tests")
    print(score_summary.round(3).to_string(index=False))
    print("\nNeutral-contrast direction geometry")
    print(geometry_summary.round(3).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
