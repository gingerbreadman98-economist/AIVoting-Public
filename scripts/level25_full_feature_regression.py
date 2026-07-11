#!/usr/bin/env python3
"""
Post-hoc full-feature regressions for Level 2.5 self-answer activations.

This consumes a Level 2.5 mechanistic output directory produced with
--save-activation-matrix. It fits prompt-grouped models using hidden-state
features, scalar geometry features, and concatenations of both.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEYS = ["prompt_id", "evaluator_id", "candidate_id"]

TARGETS: dict[str, str] = {
    "voter_help_label": "classification",
    "voter_hurt_label": "classification",
    "voter_signed_top_choice": "classification",
    "voter_best_pick_vote": "classification",
    "judge_winner": "classification",
    "voter_allocation": "regression",
    "voter_allocation_centered": "regression",
    "voter_allocation_z": "regression",
    "voter_positive_mass": "regression",
    "voter_negative_mass": "regression",
    "voter_signed_abs_share": "regression",
    "voter_allocation_rank": "regression",
    "voter_borda_points": "regression",
    "voter_borda_points_centered": "regression",
    "voter_borda_points_z": "regression",
}

SCALAR_COLS = [
    "distance_to_self_answer",
    "cosine_to_self_answer",
    "candidate_minus_self_norm",
    "distance_to_self_rank_closest",
    "distance_to_prompt_candidate_centroid",
    "distance_to_centroid_outlier_rank",
    "mean_distance_to_other_candidates",
    "min_distance_to_other_candidate",
    "max_distance_to_other_candidate",
    "mean_cosine_to_other_candidates",
    "min_cosine_to_other_candidate",
    "max_cosine_to_other_candidate",
    "mean_self_distance_advantage_vs_others",
    "min_self_distance_advantage_vs_others",
    "max_self_distance_advantage_vs_others",
    "mean_self_cosine_advantage_vs_others",
    "min_self_cosine_advantage_vs_others",
    "max_self_cosine_advantage_vs_others",
    "self_distance_change_from_previous_layer",
]

INTERACTION_PAIRS = {
    "self_dist_x_centroid_dist": (
        "distance_to_self_answer",
        "distance_to_prompt_candidate_centroid",
    ),
    "self_dist_x_outlier_rank": (
        "distance_to_self_answer",
        "distance_to_centroid_outlier_rank",
    ),
    "self_rank_x_centroid_dist": (
        "distance_to_self_rank_closest",
        "distance_to_prompt_candidate_centroid",
    ),
    "cosine_self_x_centroid_dist": (
        "cosine_to_self_answer",
        "distance_to_prompt_candidate_centroid",
    ),
    "self_change_x_self_dist": (
        "self_distance_change_from_previous_layer",
        "distance_to_self_answer",
    ),
    "self_distance_advantage_x_self_cosine_advantage": (
        "mean_self_distance_advantage_vs_others",
        "mean_self_cosine_advantage_vs_others",
    ),
    "self_distance_advantage_x_mean_pairwise_distance": (
        "mean_self_distance_advantage_vs_others",
        "mean_distance_to_other_candidates",
    ),
    "self_cosine_advantage_x_mean_pairwise_cosine": (
        "mean_self_cosine_advantage_vs_others",
        "mean_cosine_to_other_candidates",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--targets", default="")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices, e.g. 16,20,24")
    parser.add_argument("--feature-sets", default="all")
    parser.add_argument(
        "--ridge-alphas",
        "--ridge-alpha",
        dest="ridge_alphas",
        default="0.1,1,10,100,1000,10000,100000",
        help="Comma-separated alpha grid swept with group-aware inner CV.",
    )
    parser.add_argument(
        "--logistic-cs",
        "--logistic-c",
        dest="logistic_cs",
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated C grid swept with group-aware inner CV.",
    )
    parser.add_argument("--pca-components", type=int, default=0, help="0 disables PCA for activation models.")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_float_list(text: str, flag: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError(f"{flag} must contain at least one value")
    return values


def inner_group_splits(
    prompt_ids: pd.Series,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Group-aware inner CV splits (by prompt) for hyperparameter selection."""
    groups = prompt_ids.astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return None
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def parse_layer_filter(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    wanted = [int(x.strip()) for x in text.split(",") if x.strip()]
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"Requested layers not present in activation file: {missing}")
    return wanted


def load_inputs(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    activation_path = base / "self_answer_activations.npz"
    if not activation_path.exists():
        raise FileNotFoundError(
            f"{activation_path} not found. Re-run level25_self_answer_activation_analysis.py "
            "with --save-activation-matrix."
        )
    arrays = np.load(activation_path)
    candidate = arrays["candidate_activations"]
    self_answer = arrays["self_activations"]
    layers = [int(x) for x in arrays["layers"].tolist()]

    row_index_path = base / "activation_row_index.csv"
    if row_index_path.exists():
        row_index = pd.read_csv(row_index_path)
    else:
        labels = pd.read_csv(base / "self_answer_vote_labels.csv")
        row_index = labels[KEYS].copy()
    labels = pd.read_csv(base / "self_answer_vote_labels.csv")
    geometry = pd.read_csv(base / "candidate_to_self_geometry.csv")
    labels = row_index.merge(labels, on=KEYS, how="left", validate="one_to_one")
    return labels, geometry, candidate, self_answer, layers


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SCALAR_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for name, (left, right) in INTERACTION_PAIRS.items():
        out[name] = out[left].astype(float) * out[right].astype(float)
    return out


def build_scalar_wide(geometry: pd.DataFrame, row_index: pd.DataFrame, selected_layers: list[int]) -> np.ndarray:
    cols = SCALAR_COLS + list(INTERACTION_PAIRS)
    wide: pd.DataFrame | None = None
    for layer in selected_layers:
        sub = geometry[geometry["layer_index"].astype(int) == int(layer)].copy()
        sub = add_interactions(sub)
        part = sub[KEYS + cols].rename(columns={c: f"L{layer}_{c}" for c in cols})
        wide = part if wide is None else wide.merge(part, on=KEYS, how="inner", validate="one_to_one")
    if wide is None:
        raise ValueError("No scalar geometry rows matched requested layers.")
    aligned = row_index[KEYS].merge(wide, on=KEYS, how="left", validate="one_to_one")
    feature_cols = [c for c in aligned.columns if c not in KEYS]
    return aligned[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def flatten_layers(arr: np.ndarray, layer_positions: list[int]) -> np.ndarray:
    return arr[:, layer_positions, :].reshape(arr.shape[0], -1)


def make_feature_sets(
    candidate: np.ndarray,
    self_answer: np.ndarray,
    scalar: np.ndarray,
    layer_positions: list[int],
) -> dict[str, np.ndarray]:
    cand = flatten_layers(candidate, layer_positions)
    self_vec = flatten_layers(self_answer, layer_positions)
    delta = cand - self_vec
    return {
        "candidate_activation": cand,
        "candidate_minus_self_activation": delta,
        "candidate_plus_self_delta_activation": np.concatenate([cand, delta], axis=1),
        "self_activation": self_vec,
        "scalar_geometry": scalar,
        "candidate_activation_plus_scalar": np.concatenate([cand, scalar], axis=1),
        "candidate_delta_plus_scalar": np.concatenate([delta, scalar], axis=1),
        "candidate_plus_delta_plus_scalar": np.concatenate([cand, delta, scalar], axis=1),
        "candidate_self_delta_plus_scalar": np.concatenate([cand, self_vec, delta, scalar], axis=1),
    }


def make_model(
    kind: str,
    args: argparse.Namespace,
    feature_dim: int,
    n_train: int,
    logistic_cs: list[float],
    ridge_alphas: list[float],
    inner_cv: list[tuple[np.ndarray, np.ndarray]] | None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if args.pca_components > 0:
        pca_components = min(args.pca_components, feature_dim, max(1, n_train - 1))
        if pca_components < feature_dim:
            steps.append(("pca", PCA(n_components=pca_components, random_state=args.seed)))
    if kind == "classification":
        if len(logistic_cs) > 1 and inner_cv is not None:
            model: Any = LogisticRegressionCV(
                Cs=logistic_cs,
                cv=inner_cv,
                scoring="roc_auc",
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
            )
        else:
            model = LogisticRegression(
                C=float(np.median(logistic_cs)),
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
            )
        steps.append(("model", model))
    else:
        # Group-aware inner CV when possible; otherwise RidgeCV falls back to
        # efficient leave-one-out (GCV) alpha selection.
        steps.append(("model", RidgeCV(alphas=ridge_alphas, cv=inner_cv)))
    return Pipeline(steps)


def chosen_hyperparameter(model: Pipeline, kind: str) -> float:
    fitted = model.named_steps["model"]
    if kind == "classification":
        if hasattr(fitted, "C_"):
            return float(np.asarray(fitted.C_).ravel()[0])
        return float(fitted.C)
    return float(fitted.alpha_)


def prompt_group_folds(prompt_ids: pd.Series, seed: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = prompt_ids.astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_folds, n_groups)
    if splits < 2:
        raise ValueError("Need at least two prompt groups for cross-validation.")
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate(
    X: np.ndarray,
    labels: pd.DataFrame,
    target: str,
    kind: str,
    args: argparse.Namespace,
    logistic_cs: list[float],
    ridge_alphas: list[float],
) -> dict[str, Any]:
    folds = prompt_group_folds(labels["prompt_id"], args.seed)
    y_raw = labels[target]
    fold_rows = []
    for train_idx, test_idx in folds:
        inner_cv = inner_group_splits(labels["prompt_id"].iloc[train_idx])
        model = make_model(
            kind, args, X.shape[1], len(train_idx), logistic_cs, ridge_alphas, inner_cv
        )
        if kind == "classification":
            y_train = y_raw.iloc[train_idx].astype(bool).astype(int).to_numpy()
            y_test = y_raw.iloc[test_idx].astype(bool).astype(int).to_numpy()
            if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                continue
            try:
                model.fit(X[train_idx], y_train)
            except Exception:
                # Inner CV can fail on degenerate splits (e.g. a single-class
                # inner fold); fall back to a fixed mid-grid C.
                model = make_model(
                    kind, args, X.shape[1], len(train_idx), logistic_cs, ridge_alphas, None
                )
                model.fit(X[train_idx], y_train)
            probs = model.predict_proba(X[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            fold_rows.append(
                {
                    "accuracy": accuracy_score(y_test, preds),
                    "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                    "auc": safe_auc(y_test, probs),
                    "chosen_c": chosen_hyperparameter(model, kind),
                }
            )
        else:
            y_train = y_raw.iloc[train_idx].astype(float).to_numpy()
            y_test = y_raw.iloc[test_idx].astype(float).to_numpy()
            model.fit(X[train_idx], y_train)
            preds = model.predict(X[test_idx])
            fold_rows.append(
                {
                    "rmse": math.sqrt(mean_squared_error(y_test, preds)),
                    "r2": r2_score(y_test, preds),
                    "chosen_alpha": chosen_hyperparameter(model, kind),
                }
            )
    if not fold_rows:
        return {"n_folds": 0}
    out: dict[str, Any] = {"n_folds": len(fold_rows)}
    for key in fold_rows[0]:
        values = np.array([row[key] for row in fold_rows], dtype=float)
        out[f"mean_{key}"] = float(np.nanmean(values))
        out[f"std_{key}"] = float(np.nanstd(values))
    return out


def main() -> None:
    args = parse_args()
    logistic_cs = parse_float_list(args.logistic_cs, "--logistic-cs")
    ridge_alphas = parse_float_list(args.ridge_alphas, "--ridge-alphas")
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"full_feature_regressions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, geometry, candidate, self_answer, available_layers = load_inputs(base)
    selected_layers = parse_layer_filter(args.layers, available_layers)
    layer_positions = [available_layers.index(layer) for layer in selected_layers]
    scalar = build_scalar_wide(geometry, labels, selected_layers)
    feature_sets = make_feature_sets(candidate, self_answer, scalar, layer_positions)

    requested_feature_sets = list(feature_sets)
    if args.feature_sets.strip().lower() != "all":
        requested_feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
        missing = sorted(set(requested_feature_sets) - set(feature_sets))
        if missing:
            raise ValueError(f"Unknown feature sets: {missing}")

    requested_targets = list(TARGETS)
    if args.targets.strip():
        requested_targets = [x.strip() for x in args.targets.split(",") if x.strip()]
        missing_targets = sorted(set(requested_targets) - set(TARGETS))
        if missing_targets:
            raise ValueError(f"Unknown targets: {missing_targets}")
    unavailable_targets = sorted(set(requested_targets) - set(labels.columns))
    if unavailable_targets:
        raise ValueError(
            f"Requested target column(s) not found in self_answer_vote_labels.csv: {unavailable_targets}"
        )

    rows = []
    for feature_name in requested_feature_sets:
        X = feature_sets[feature_name]
        for target in requested_targets:
            kind = TARGETS[target]
            metrics = evaluate(X, labels, target, kind, args, logistic_cs, ridge_alphas)
            rows.append(
                {
                    "feature_set": feature_name,
                    "feature_dim": int(X.shape[1]),
                    "target": target,
                    "kind": kind,
                    "layers_json": json.dumps(selected_layers),
                    "pca_components": args.pca_components,
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "full_feature_regression_results.csv", index=False)

    best_rows = []
    for target, group in results.groupby("target"):
        kind = group["kind"].iloc[0]
        metric = "mean_auc" if kind == "classification" else "mean_r2"
        valid = group[np.isfinite(group[metric])].copy()
        if valid.empty:
            continue
        best = valid.loc[valid[metric].idxmax()].to_dict()
        best["selection_metric"] = metric
        best["selection_value"] = best[metric]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_df = best_df.sort_values("target")
    best_df.to_csv(out_dir / "full_feature_regression_best.csv", index=False)

    run_config = pd.DataFrame(
        [
            {
                "mech_output_dir": str(base),
                "n_rows": int(len(labels)),
                "n_prompts": int(labels["prompt_id"].nunique()),
                "available_layers_json": json.dumps(available_layers),
                "selected_layers_json": json.dumps(selected_layers),
                "ridge_alphas_json": json.dumps(ridge_alphas),
                "logistic_cs_json": json.dumps(logistic_cs),
                "pca_components": args.pca_components,
                "feature_sets_json": json.dumps(requested_feature_sets),
                "targets_json": json.dumps(requested_targets),
            }
        ]
    )
    run_config.to_csv(out_dir / "run_config.csv", index=False)
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nFull-feature best results")
    print(
        best_df[
            ["target", "feature_set", "feature_dim", "selection_metric", "selection_value"]
        ].to_string(index=False)
        if not best_df.empty
        else "No models fit."
    )
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive}")


if __name__ == "__main__":
    main()
