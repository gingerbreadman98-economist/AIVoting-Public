#!/usr/bin/env python3
"""Regression summaries for causal activation steering outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


SEMANTIC_UP = {"help", "neg_hurt"}
SEMANTIC_DOWN = {"hurt", "neg_help"}
RANDOM_DIRECTIONS = {"random", "random2", "random3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    return parser.parse_args()


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def add_constant_and_dummies(
    df: pd.DataFrame,
    numeric_cols: list[str],
    dummy_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(len(df), dtype=float)]
    names = ["const"]
    for col in numeric_cols:
        cols.append(df[col].astype(float).to_numpy())
        names.append(col)
    for col in dummy_cols or []:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
        for name in dummies.columns:
            cols.append(dummies[name].to_numpy())
            names.append(str(name))
    return np.column_stack(cols), names


def cluster_ols(
    df: pd.DataFrame,
    y_col: str,
    numeric_cols: list[str],
    dummy_cols: list[str] | None = None,
    cluster_col: str = "ballot_id",
    model_name: str = "",
) -> pd.DataFrame:
    work = df[[y_col, cluster_col] + numeric_cols + (dummy_cols or [])].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    y = work[y_col].astype(float).to_numpy()
    X, names = add_constant_and_dummies(work, numeric_cols, dummy_cols)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    groups = work[cluster_col].astype(str).to_numpy()
    for group in np.unique(groups):
        idx = groups == group
        xg = X[idx]
        ug = residual[idx][:, None]
        meat += xg.T @ ug @ ug.T @ xg
    n = X.shape[0]
    k = X.shape[1]
    g = len(np.unique(groups))
    finite = 1.0
    if g > 1 and n > k:
        finite = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    cov = finite * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / np.where(se == 0, np.nan, se)
    pvals = np.array([2.0 * (1.0 - normal_cdf(abs(t))) if np.isfinite(t) else np.nan for t in tvals])
    return pd.DataFrame(
        {
            "model": model_name,
            "term": names,
            "coef": beta,
            "cluster_se": se,
            "t": tvals,
            "p_normal": pvals,
            "n": n,
            "clusters": g,
            "r2": 1.0 - float(np.sum(residual**2)) / float(np.sum((y - y.mean()) ** 2)),
        }
    )


def format_coef(coef: float, p: float) -> str:
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{coef:+.3f}{stars}"


def build_target_frame(votes: pd.DataFrame) -> pd.DataFrame:
    baseline = votes[votes["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation", "best_pick_vote", "borda_points"]
    ].rename(
        columns={
            "allocation": "baseline_allocation",
            "best_pick_vote": "baseline_best_pick_vote",
            "borda_points": "baseline_borda_points",
        }
    )
    st = votes[votes["condition"] == "steered"].merge(
        baseline, on=["ballot_id", "candidate_id"], how="left", validate="many_to_one"
    )
    st["delta_allocation"] = st["allocation"] - st["baseline_allocation"]
    st["delta_borda"] = st["borda_points"] - st["baseline_borda_points"]
    st["delta_best_pick"] = st["best_pick_vote"] - st["baseline_best_pick_vote"]
    target = st[st["is_target_candidate"].astype(bool)].copy()
    target["direction"] = target["direction"].astype(str)
    target["top_policy"] = (target["target_policy"].astype(str) == "top").astype(float)
    target["is_random"] = target["direction"].isin(RANDOM_DIRECTIONS)
    target["semantic_sign"] = np.select(
        [target["direction"].isin(SEMANTIC_UP), target["direction"].isin(SEMANTIC_DOWN)],
        [1.0, -1.0],
        default=0.0,
    )
    target["signed_strength"] = target["semantic_sign"] * target["strength"].astype(float)
    target["random_strength"] = np.where(target["is_random"], target["strength"].astype(float), 0.0)
    target["intended_cross"] = (
        (
            target["direction"].isin(SEMANTIC_UP)
            & (target["baseline_allocation"] <= 0)
            & (target["allocation"] > 0)
        )
        | (
            target["direction"].isin(SEMANTIC_DOWN)
            & (target["baseline_allocation"] >= 0)
            & (target["allocation"] < 0)
        )
    ).astype(float)
    return target


def crossing_rates(target: pd.DataFrame) -> pd.DataFrame:
    strong = target[np.isclose(target["strength"], 0.20)].copy()
    semantic = strong[strong["direction"].isin(SEMANTIC_UP | SEMANTIC_DOWN)].copy()
    rows = (
        semantic.groupby("intervention")
        .agg(n=("ballot_id", "nunique"), intended_crossing_rate=("intended_cross", "mean"))
        .reset_index()
    )

    random_rows = []
    for policy, group in strong[strong["is_random"]].groupby("target_policy"):
        if str(policy) == "bottom":
            crossed = (group["baseline_allocation"] <= 0) & (group["allocation"] > 0)
        else:
            crossed = (group["baseline_allocation"] >= 0) & (group["allocation"] < 0)
        random_rows.append(
            {
                "intervention": f"random_{policy}_3seed_mean",
                "n": int(group["ballot_id"].nunique()),
                "intended_crossing_rate": float(crossed.mean()),
            }
        )
    return pd.concat([rows, pd.DataFrame(random_rows)], ignore_index=True)


def run_regressions(out: Path) -> None:
    votes = pd.read_csv(out / "causal_steering_vote_rows.csv")
    target = build_target_frame(votes)
    regressions: list[pd.DataFrame] = []

    regressions.append(
        cluster_ols(
            target,
            "delta_allocation",
            ["signed_strength", "random_strength", "baseline_allocation", "top_policy"],
            model_name="main_target_delta",
        )
    )
    for policy in ["top", "bottom"]:
        part = target[target["target_policy"].astype(str) == policy]
        regressions.append(
            cluster_ols(
                part,
                "delta_allocation",
                ["signed_strength", "random_strength", "baseline_allocation"],
                model_name=f"{policy}_target_delta",
            )
        )

    semantic = target[target["direction"].isin(SEMANTIC_UP | SEMANTIC_DOWN)].copy()
    regressions.append(
        cluster_ols(
            semantic,
            "intended_cross",
            ["strength", "baseline_allocation", "top_policy"],
            ["direction"],
            model_name="semantic_intended_cross",
        )
    )

    spill_path = out / "causal_steering_per_ballot_middle_spillovers.csv"
    if not spill_path.exists():
        raise FileNotFoundError(
            f"{spill_path} not found. Run analyze_causal_steering_spillovers.py first."
        )
    spill = pd.read_csv(spill_path)
    top_spill = spill[
        (spill["target_policy"].astype(str) == "top")
        & np.isclose(spill["strength"].astype(float), 0.20)
    ].copy()
    regressions.append(
        cluster_ols(
            top_spill,
            "mean_middle_sum_delta",
            ["mean_target_delta"],
            ["direction"],
            cluster_col="intervention",
            model_name="top_policy_middle_spillover_aggregated",
        )
    )

    # Per-ballot version for the same spillover, reconstructed from vote rows so
    # the coefficient can be clustered by ballot. This is the version to cite.
    baseline = votes[votes["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation"]
    ].rename(columns={"allocation": "baseline_allocation"})
    baseline["fresh_rank"] = baseline.groupby("ballot_id")["baseline_allocation"].rank(
        method="first", ascending=False
    )
    st = votes[votes["condition"] == "steered"].merge(
        baseline, on=["ballot_id", "candidate_id"], how="left", validate="many_to_one"
    )
    st["delta_allocation"] = st["allocation"] - st["baseline_allocation"]
    target_delta = st[st["is_target_candidate"].astype(bool)][
        ["ballot_id", "intervention", "direction", "target_policy", "strength", "delta_allocation"]
    ].rename(columns={"delta_allocation": "target_delta"})
    middle = st[(st["fresh_rank"].isin([2.0, 3.0])) & (~st["is_target_candidate"].astype(bool))]
    per_ballot = (
        middle.groupby(["ballot_id", "intervention", "direction", "target_policy", "strength"], dropna=False)
        .agg(middle_sum_delta=("delta_allocation", "sum"))
        .reset_index()
        .merge(
            target_delta,
            on=["ballot_id", "intervention", "direction", "target_policy", "strength"],
            how="left",
            validate="one_to_one",
        )
    )
    top_per_ballot = per_ballot[
        (per_ballot["target_policy"].astype(str) == "top")
        & np.isclose(per_ballot["strength"].astype(float), 0.20)
    ].copy()
    regressions.append(
        cluster_ols(
            top_per_ballot,
            "middle_sum_delta",
            ["target_delta"],
            ["direction"],
            model_name="top_policy_middle_spillover",
        )
    )

    reg = pd.concat(regressions, ignore_index=True)
    reg.to_csv(out / "causal_steering_regressions.csv", index=False)
    crossing = crossing_rates(target)
    crossing.to_csv(out / "causal_steering_regression_crossing_rates.csv", index=False)

    print("\nRegression coefficients")
    for model in reg["model"].unique():
        show = reg[reg["model"] == model].copy()
        print(f"\n{model}")
        print(
            show[["term", "coef", "cluster_se", "p_normal", "n", "clusters", "r2"]]
            .to_string(index=False, formatters={"coef": "{:+.3f}".format, "cluster_se": "{:.3f}".format, "p_normal": "{:.4g}".format, "r2": "{:.3f}".format})
        )
    print("\nCrossing rates at 20%")
    print(crossing.to_string(index=False, formatters={"intended_crossing_rate": "{:.3f}".format}))

    main = reg[(reg["model"] == "main_target_delta") & (reg["term"].isin(["signed_strength", "random_strength", "baseline_allocation", "top_policy"]))]
    print("\nPaper main table cells")
    for row in main.itertuples(index=False):
        print(f"{row.term}: {format_coef(row.coef, row.p_normal)} (SE {row.cluster_se:.3f})")


def main() -> None:
    args = parse_args()
    run_regressions(Path(args.output_dir))


if __name__ == "__main__":
    main()
