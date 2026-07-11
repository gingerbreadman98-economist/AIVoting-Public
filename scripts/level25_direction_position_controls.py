#!/usr/bin/env python3
"""
Controls for the Absolute Allocation mechanistic analysis (paperMech.tex,
Section "Controls: Direction Reliability and Position-Matched Decodability").

Control 1: split-half reliability of help/hurt probe directions, with
attenuation-corrected help/hurt cosine. Tests whether the raw help/hurt
direction cosine reflects genuine asymmetry or estimation noise.

Control 2: position-matched decodability. Refits binary probes within each
shown-position stratum so position cannot contribute to AUC.

Usage:
  python level25_direction_position_controls.py --mech-output-dir <dir> [--layer 16]
Requires a mech output dir produced with --save-activation-matrix and a vote
run that recorded candidate_display_order.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BINARY_TARGETS = [
    "voter_help_label",
    "voter_hurt_label",
    "voter_best_pick_vote",
    "voter_signed_top_choice",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--split-half-repeats", type=int, default=10)
    parser.add_argument("--logistic-c", type=float, default=0.1)
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

    def shown_position(row: pd.Series) -> float:
        order = str(row.get("candidate_display_order", "")).split(",")
        try:
            return order.index(str(row["candidate_id"])) + 1
        except ValueError:
            return np.nan

    labels["shown_position"] = labels.apply(shown_position, axis=1)
    return X, labels


def bool_column(labels: pd.DataFrame, column: str) -> np.ndarray:
    return labels[column].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()


def fit_direction(X: np.ndarray, y: np.ndarray, c: float, seed: int) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=seed
        ),
    )
    model.fit(X, y)
    return model[-1].coef_.reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def split_half_direction_control(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    help_y = bool_column(labels, "voter_help_label")
    hurt_y = bool_column(labels, "voter_hurt_label")
    raw = cosine(
        fit_direction(X, help_y, args.logistic_c, args.seed),
        fit_direction(X, hurt_y, args.logistic_c, args.seed),
    )
    prompts = labels["prompt_id"].astype(str).to_numpy()
    unique_prompts = np.unique(prompts)
    rng = np.random.default_rng(args.seed)
    help_rel, hurt_rel, cross, corrected = [], [], [], []
    for repeat in range(args.split_half_repeats):
        perm = rng.permutation(unique_prompts)
        half_a = set(perm[: len(unique_prompts) // 2])
        mask_a = np.isin(prompts, list(half_a))
        mask_b = ~mask_a
        h_a = fit_direction(X[mask_a], help_y[mask_a], args.logistic_c, args.seed + repeat)
        h_b = fit_direction(X[mask_b], help_y[mask_b], args.logistic_c, args.seed + repeat)
        t_a = fit_direction(X[mask_a], hurt_y[mask_a], args.logistic_c, args.seed + repeat)
        t_b = fit_direction(X[mask_b], hurt_y[mask_b], args.logistic_c, args.seed + repeat)
        rel_h = cosine(h_a, h_b)
        rel_t = cosine(t_a, t_b)
        c = (cosine(h_a, t_b) + cosine(h_b, t_a)) / 2
        help_rel.append(rel_h)
        hurt_rel.append(rel_t)
        cross.append(c)
        if rel_h > 0 and rel_t > 0:
            corrected.append(c / np.sqrt(rel_h * rel_t))
    return pd.DataFrame(
        [
            {
                "layer": args.layer,
                "n_repeats": args.split_half_repeats,
                "raw_help_hurt_cosine": raw,
                "mean_help_reliability": float(np.mean(help_rel)),
                "std_help_reliability": float(np.std(help_rel)),
                "mean_hurt_reliability": float(np.mean(hurt_rel)),
                "std_hurt_reliability": float(np.std(hurt_rel)),
                "mean_cross_half_cosine": float(np.mean(cross)),
                "std_cross_half_cosine": float(np.std(cross)),
                "mean_corrected_cosine": float(np.mean(corrected)),
                "std_corrected_cosine": float(np.std(corrected)),
            }
        ]
    )


def grouped_auc(
    X: np.ndarray, y: np.ndarray, prompt_ids: pd.Series, args: argparse.Namespace
) -> tuple[float, float, int]:
    groups = prompt_ids.astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    aucs = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        steps = [StandardScaler()]
        if args.pca_components > 0:
            steps.append(
                PCA(
                    n_components=min(args.pca_components, len(train_idx) - 1, X.shape[1]),
                    random_state=args.seed,
                )
            )
        steps.append(
            LogisticRegression(
                C=args.logistic_c,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                random_state=args.seed,
            )
        )
        model = make_pipeline(*steps)
        model.fit(X[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def position_matched_control(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    if labels["shown_position"].isna().any():
        raise ValueError(
            "Could not recover shown_position for every row. Re-run the vote "
            "script so candidate_display_order is recorded."
        )
    rows = []
    for target in BINARY_TARGETS:
        y = bool_column(labels, target)
        per_position = []
        for position in [1, 2, 3, 4]:
            mask = (labels["shown_position"] == position).to_numpy()
            auc, std, n_folds = grouped_auc(X[mask], y[mask], labels.loc[mask, "prompt_id"], args)
            per_position.append(auc)
            rows.append(
                {
                    "target": target,
                    "shown_position": position,
                    "n_rows": int(mask.sum()),
                    "n_positive": int(y[mask].sum()),
                    "n_folds": n_folds,
                    "mean_auc": auc,
                    "std_auc": std,
                }
            )
        rows.append(
            {
                "target": target,
                "shown_position": "pooled",
                "n_rows": int(len(y)),
                "n_positive": int(y.sum()),
                "n_folds": np.nan,
                "mean_auc": float(np.mean(per_position)),
                "std_auc": float(np.std(per_position)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"direction_position_controls_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    X, labels = load_inputs(base, args.layer)
    direction_summary = split_half_direction_control(X, labels, args)
    direction_summary.to_csv(out_dir / "direction_reliability_control.csv", index=False)
    position_summary = position_matched_control(X, labels, args)
    position_summary.to_csv(out_dir / "position_matched_control.csv", index=False)

    run_config = pd.DataFrame(
        [
            {
                "mech_output_dir": str(base),
                "layer": args.layer,
                "pca_components": args.pca_components,
                "split_half_repeats": args.split_half_repeats,
                "logistic_c": args.logistic_c,
                "seed": args.seed,
            }
        ]
    )
    run_config.to_csv(out_dir / "run_config.csv", index=False)

    print("Direction reliability control")
    print(direction_summary.round(3).to_string(index=False))
    print("\nPosition-matched control")
    print(position_summary.round(3).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
