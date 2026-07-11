#!/usr/bin/env python3
"""
Answer-length controls for the Absolute Allocation mechanistic analysis
(paperMech.tex, Limitations: "Answer length is an uncontrolled covariate").

The four candidate styles (concise, quick first pass, deep, normal) differ
systematically in length, and verbosity is a well-documented LLM-judge bias.
This script tests whether the decodable help/hurt/allocation signal is
separable from answer length, mirroring level25_direction_position_controls.py
(position-matched control) and the residual-beyond-Borda analysis in
level25_extra_offline_analysis.py.

Outputs (timestamped subdir of the mech output dir):
  length_distribution_by_polarity.csv  length vs help/hurt/neutral/top, and
                                        correlations length vs votes.
  length_only_baselines.csv            length-only probe metrics (the analogue
                                        of the position-only baseline).
  length_covariate_probes.csv          activations vs activations+length.
  length_matched_control.csv           decodability refit within length strata
                                        so length cannot contribute to AUC.
  residual_beyond_rank_with_length.csv residual allocation R^2 after adding
                                        length to the ordinal baseline.

Length metric: whitespace word count by default (--length-metric word) or
character count (--length-metric char). Both are always recorded; the chosen
metric drives strata and the covariate.

Usage:
  python level25_length_controls.py --mech-output-dir <dir> [--layer 16]
Requires a mech output dir with self_answer_activations.npz,
activation_row_index.csv, self_answer_vote_labels.csv, and
self_answer_vote_rows_with_text.csv (for candidate_answer text), produced by a
vote run that recorded candidate_display_order.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BINARY_TARGETS = [
    "voter_help_label",
    "voter_hurt_label",
    "voter_best_pick_vote",
    "voter_signed_top_choice",
]

CONTINUOUS_TARGETS = [
    "voter_allocation",
    "voter_allocation_z",
    "voter_positive_mass",
    "voter_negative_mass",
]

KEY = ["prompt_id", "evaluator_id", "candidate_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--length-metric", choices=["word", "char"], default="word")
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(str(text).split())


def char_count(text: str) -> int:
    return len(str(text))


def load_inputs(base: Path, layer: int, length_metric: str) -> tuple[np.ndarray, pd.DataFrame]:
    npz = np.load(base / "self_answer_activations.npz")
    layers = npz["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} not in saved layers: {layers}")
    X = npz["candidate_activations"][:, layers.index(layer), :].astype(np.float64)

    labels = pd.read_csv(base / "self_answer_vote_labels.csv")
    row_index = pd.read_csv(base / "activation_row_index.csv")
    labels = row_index.merge(labels, on=KEY, how="left", validate="one_to_one")

    text_path = base / "self_answer_vote_rows_with_text.csv"
    if not text_path.exists():
        raise FileNotFoundError(
            f"{text_path} not found; candidate_answer text is required to compute length."
        )
    text = pd.read_csv(text_path)
    if "candidate_answer" not in text.columns:
        raise ValueError("self_answer_vote_rows_with_text.csv lacks a candidate_answer column.")
    text = text[KEY + ["candidate_answer"]].drop_duplicates(subset=KEY)
    text["answer_word_len"] = text["candidate_answer"].map(word_count)
    text["answer_char_len"] = text["candidate_answer"].map(char_count)
    labels = labels.merge(
        text[KEY + ["answer_word_len", "answer_char_len"]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if labels[["answer_word_len", "answer_char_len"]].isna().any().any():
        raise ValueError("Some rows have no candidate_answer text; cannot compute length.")

    labels["length_metric"] = (
        labels["answer_word_len"] if length_metric == "word" else labels["answer_char_len"]
    ).astype(float)

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


def one_hot_position(positions: pd.Series) -> np.ndarray:
    enc = OneHotEncoder(categories=[[1, 2, 3, 4]], sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(positions.astype(int).to_numpy().reshape(-1, 1))


def make_pca_steps(n_train: int, n_features: int, pca_components: int, seed: int) -> list:
    steps = [StandardScaler()]
    if pca_components > 0:
        n_pca = max(1, min(pca_components, n_train - 1, n_features))
        steps.append(PCA(n_components=n_pca, random_state=seed))
    return steps


def grouped_metric(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: pd.Series,
    kind: str,
    pca_components: int,
    logistic_c: float,
    seed: int,
) -> tuple[float, float, int]:
    groups = prompt_ids.astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        if kind == "classification" and (
            len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2
        ):
            continue
        steps = make_pca_steps(len(train_idx), X.shape[1], pca_components, seed)
        if kind == "classification":
            steps.append(
                LogisticRegression(
                    C=logistic_c,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                )
            )
            model = make_pipeline(*steps)
            model.fit(X[train_idx], y[train_idx])
            scores.append(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
        else:
            steps.append(Ridge(alpha=100.0))
            model = make_pipeline(*steps)
            model.fit(X[train_idx], y[train_idx])
            scores.append(r2_score(y[test_idx], model.predict(X[test_idx])))
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def length_strata(length: np.ndarray, n_strata: int) -> np.ndarray:
    """Rank-based equal-count strata (robust to tied lengths)."""
    order = np.argsort(np.argsort(length, kind="stable"), kind="stable")
    n = len(length)
    return np.minimum((order * n_strata) // n, n_strata - 1)


def distribution_by_polarity(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    length = labels["length_metric"].to_numpy()
    help_y = bool_column(labels, "voter_help_label")
    hurt_y = bool_column(labels, "voter_hurt_label")
    neutral_y = ((help_y == 0) & (hurt_y == 0)).astype(int)
    top_y = bool_column(labels, "voter_signed_top_choice")
    best_y = bool_column(labels, "voter_best_pick_vote")
    for name, mask in [
        ("helped", help_y == 1),
        ("hurt", hurt_y == 1),
        ("neutral", neutral_y == 1),
        ("alloc_top_choice", top_y == 1),
        ("not_alloc_top_choice", top_y == 0),
        ("best_pick", best_y == 1),
        ("not_best_pick", best_y == 0),
        ("all", np.ones(len(length), dtype=bool)),
    ]:
        vals = length[mask]
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "mean_length": float(np.mean(vals)) if len(vals) else np.nan,
                "median_length": float(np.median(vals)) if len(vals) else np.nan,
                "std_length": float(np.std(vals)) if len(vals) else np.nan,
            }
        )
    dist = pd.DataFrame(rows)

    alloc = labels["voter_allocation"].astype(float).to_numpy()
    corr_rows = [
        {"pair": "length_vs_allocation", "pearson_r": float(np.corrcoef(length, alloc)[0, 1])},
        {"pair": "length_vs_help", "pearson_r": float(np.corrcoef(length, help_y)[0, 1])},
        {"pair": "length_vs_hurt", "pearson_r": float(np.corrcoef(length, hurt_y)[0, 1])},
        {"pair": "length_vs_best_pick", "pearson_r": float(np.corrcoef(length, best_y)[0, 1])},
        {
            "pair": "length_vs_positive_mass",
            "pearson_r": float(
                np.corrcoef(length, labels["voter_positive_mass"].astype(float))[0, 1]
            ),
        },
        {
            "pair": "length_vs_negative_mass",
            "pearson_r": float(
                np.corrcoef(length, labels["voter_negative_mass"].astype(float))[0, 1]
            ),
        },
    ]
    corr = pd.DataFrame(corr_rows)
    corr["n"] = int(len(length))
    corr = corr.rename(columns={"pearson_r": "value"})
    corr["group"] = corr["pair"]
    corr["metric"] = "pearson_r"
    dist["metric"] = "length_summary"
    return pd.concat(
        [dist, corr[["group", "metric", "value", "n"]]], ignore_index=True
    )


def length_only_and_covariate(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    length_col = labels["length_metric"].to_numpy().reshape(-1, 1)
    X_len = np.concatenate([X, length_col], axis=1)
    only_rows = []
    cov_rows = []
    for target, kind in (
        [(t, "classification") for t in BINARY_TARGETS]
        + [(t, "regression") for t in CONTINUOUS_TARGETS]
    ):
        if kind == "classification":
            y = bool_column(labels, target)
        else:
            y = labels[target].astype(float).to_numpy()
        mean_len, std_len, n_len = grouped_metric(
            length_col, y, labels["prompt_id"], kind, 0, args.logistic_c, args.seed
        )
        only_rows.append(
            {"target": target, "kind": kind, "feature": "length_only",
             "mean_metric": mean_len, "std_metric": std_len, "n_folds": n_len}
        )
        mean_act, std_act, n_act = grouped_metric(
            X, y, labels["prompt_id"], kind, args.pca_components, args.logistic_c, args.seed
        )
        mean_both, std_both, n_both = grouped_metric(
            X_len, y, labels["prompt_id"], kind, args.pca_components, args.logistic_c, args.seed
        )
        cov_rows.append(
            {
                "target": target,
                "kind": kind,
                "activation_metric": mean_act,
                "activation_plus_length_metric": mean_both,
                "delta": mean_both - mean_act,
                "n_folds": n_act,
            }
        )
    return pd.DataFrame(only_rows), pd.DataFrame(cov_rows)


def length_matched_control(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    strata = length_strata(labels["length_metric"].to_numpy(), args.n_strata)
    rows = []
    for target in BINARY_TARGETS:
        y = bool_column(labels, target)
        per_stratum = []
        for s in range(args.n_strata):
            mask = strata == s
            auc, std, n_folds = grouped_metric(
                X[mask], y[mask], labels.loc[mask, "prompt_id"], "classification",
                args.pca_components, args.logistic_c, args.seed,
            )
            per_stratum.append(auc)
            lo = float(labels.loc[mask, "length_metric"].min())
            hi = float(labels.loc[mask, "length_metric"].max())
            rows.append(
                {
                    "target": target,
                    "length_stratum": s + 1,
                    "length_low": lo,
                    "length_high": hi,
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
                "length_stratum": "pooled",
                "length_low": np.nan,
                "length_high": np.nan,
                "n_rows": int(len(y)),
                "n_positive": int(y.sum()),
                "n_folds": np.nan,
                "mean_auc": float(np.nanmean(per_stratum)),
                "std_auc": float(np.nanstd(per_stratum)),
            }
        )
    return pd.DataFrame(rows)


def residual_with_length(
    X: np.ndarray, labels: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    prompt_ids = labels["prompt_id"]
    rank_cols = ["voter_borda_points", "voter_borda_points_z", "voter_allocation_rank"]
    length_col = labels["length_metric"].to_numpy().reshape(-1, 1)
    rows = []
    for target in CONTINUOUS_TARGETS:
        y = labels[target].astype(float).to_numpy()
        for baseline_name, base_X in [
            ("rank_only", labels[rank_cols].astype(float).to_numpy()),
            (
                "rank_plus_length",
                np.concatenate([labels[rank_cols].astype(float).to_numpy(), length_col], axis=1),
            ),
            ("length_only", length_col),
        ]:
            r2s = []
            base_r2s = []
            for train_idx, test_idx in GroupKFold(
                n_splits=min(5, len(np.unique(prompt_ids.astype(str))))
            ).split(np.zeros(len(y)), groups=prompt_ids.astype(str).to_numpy()):
                base = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                base.fit(base_X[train_idx], y[train_idx])
                base_r2s.append(r2_score(y[test_idx], base.predict(base_X[test_idx])))
                y_res_train = y[train_idx] - base.predict(base_X[train_idx])
                y_res_test = y[test_idx] - base.predict(base_X[test_idx])
                steps = make_pca_steps(len(train_idx), X.shape[1], args.pca_components, args.seed)
                steps.append(Ridge(alpha=100.0))
                model = make_pipeline(*steps)
                model.fit(X[train_idx], y_res_train)
                r2s.append(r2_score(y_res_test, model.predict(X[test_idx])))
            rows.append(
                {
                    "target": target,
                    "baseline": baseline_name,
                    "baseline_r2": float(np.mean(base_r2s)),
                    "activation_residual_r2": float(np.mean(r2s)),
                    "std_residual_r2": float(np.std(r2s)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"length_controls_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    X, labels = load_inputs(base, args.layer, args.length_metric)

    dist = distribution_by_polarity(labels)
    dist.to_csv(out_dir / "length_distribution_by_polarity.csv", index=False)

    only_df, cov_df = length_only_and_covariate(X, labels, args)
    only_df.to_csv(out_dir / "length_only_baselines.csv", index=False)
    cov_df.to_csv(out_dir / "length_covariate_probes.csv", index=False)

    matched = length_matched_control(X, labels, args)
    matched.to_csv(out_dir / "length_matched_control.csv", index=False)

    residual = residual_with_length(X, labels, args)
    residual.to_csv(out_dir / "residual_beyond_rank_with_length.csv", index=False)

    run_config = pd.DataFrame(
        [
            {
                "mech_output_dir": str(base),
                "layer": args.layer,
                "pca_components": args.pca_components,
                "length_metric": args.length_metric,
                "n_strata": args.n_strata,
                "logistic_c": args.logistic_c,
                "seed": args.seed,
                "n_rows": int(len(labels)),
            }
        ]
    )
    run_config.to_csv(out_dir / "run_config.csv", index=False)

    print("Length distribution by polarity")
    print(dist.round(3).to_string(index=False))
    print("\nLength-only baselines")
    print(only_df.round(3).to_string(index=False))
    print("\nActivation vs activation+length")
    print(cov_df.round(3).to_string(index=False))
    print("\nLength-matched decodability control")
    print(matched.round(3).to_string(index=False))
    print("\nResidual beyond rank, with length")
    print(residual.round(3).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
