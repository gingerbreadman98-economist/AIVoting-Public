#!/usr/bin/env python3
"""Find non-target allocation spillovers in causal steering outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--min-abs-delta", type=float, default=0.10)
    return parser.parse_args()


def baseline_roles(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["_alloc_rank_desc"] = base.groupby("ballot_id")["baseline_allocation"].rank(
        method="first", ascending=False
    )
    base["_alloc_rank_asc"] = base.groupby("ballot_id")["baseline_allocation"].rank(
        method="first", ascending=True
    )
    base["baseline_role"] = np.select(
        [base["_alloc_rank_desc"] == 1, base["_alloc_rank_asc"] == 1],
        ["top", "bottom"],
        default="middle",
    )
    base["baseline_middle_rank"] = base.groupby("ballot_id")["baseline_allocation"].rank(
        method="first", ascending=False
    )
    return base.drop(columns=["_alloc_rank_desc", "_alloc_rank_asc"])


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    votes = pd.read_csv(out / "causal_steering_vote_rows.csv")
    base = votes[votes["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation", "best_pick_vote", "borda_points"]
    ].rename(
        columns={
            "allocation": "baseline_allocation",
            "best_pick_vote": "baseline_best_pick_vote",
            "borda_points": "baseline_borda_points",
        }
    )
    base = baseline_roles(base)
    st = votes[votes["condition"] == "steered"].merge(
        base, on=["ballot_id", "candidate_id"], how="left", validate="many_to_one"
    )
    st["delta_allocation"] = st["allocation"] - st["baseline_allocation"]
    st["delta_borda"] = st["borda_points"] - st["baseline_borda_points"]
    st["delta_best_pick"] = st["best_pick_vote"] - st["baseline_best_pick_vote"]
    st["abs_delta_allocation"] = st["delta_allocation"].abs()
    st["parse_failed"] = st["parse_error"].fillna("").astype(str) != ""

    target = st[st["is_target_candidate"].astype(bool)][
        [
            "ballot_id",
            "intervention",
            "direction",
            "target_policy",
            "strength",
            "candidate_id",
            "baseline_role",
            "baseline_allocation",
            "allocation",
            "delta_allocation",
            "delta_borda",
            "delta_best_pick",
        ]
    ].rename(
        columns={
            "candidate_id": "target_candidate_id",
            "baseline_role": "target_baseline_role",
            "baseline_allocation": "target_baseline_allocation",
            "allocation": "target_allocation",
            "delta_allocation": "target_delta_allocation",
            "delta_borda": "target_delta_borda",
            "delta_best_pick": "target_delta_best_pick",
        }
    )

    st = st.drop(columns=["target_candidate_id", "target_display_id"], errors="ignore")
    non = st[~st["is_target_candidate"].astype(bool)].merge(
        target,
        on=["ballot_id", "intervention", "direction", "target_policy", "strength"],
        how="left",
        validate="many_to_one",
    )

    summary = (
        non.groupby(["intervention", "direction", "target_policy", "strength", "baseline_role"], dropna=False)
        .agg(
            n_rows=("candidate_id", "size"),
            mean_delta_allocation=("delta_allocation", "mean"),
            mean_abs_delta_allocation=("abs_delta_allocation", "mean"),
            median_delta_allocation=("delta_allocation", "median"),
            increase_rate=("delta_allocation", lambda s: float((s > 1e-12).mean())),
            decrease_rate=("delta_allocation", lambda s: float((s < -1e-12).mean())),
            large_change_rate=("abs_delta_allocation", lambda s: float((s >= args.min_abs_delta).mean())),
            mean_delta_borda=("delta_borda", "mean"),
            best_pick_gain_rate=("delta_best_pick", lambda s: float((s > 0).mean())),
            best_pick_loss_rate=("delta_best_pick", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )

    middle_summary = summary[summary["baseline_role"] == "middle"].copy()
    strong_middle = non[
        (non["baseline_role"] == "middle")
        & (non["abs_delta_allocation"] >= args.min_abs_delta)
    ].copy()
    strong_middle = strong_middle.sort_values(
        ["abs_delta_allocation", "strength"], ascending=[False, False]
    )

    # Per steered ballot: target shift vs total middle shift.
    per_ballot_middle = (
        non[non["baseline_role"] == "middle"]
        .groupby(["intervention", "direction", "target_policy", "strength", "ballot_id"], dropna=False)
        .agg(
            middle_sum_delta=("delta_allocation", "sum"),
            middle_abs_delta=("abs_delta_allocation", "sum"),
            middle_max_abs_delta=("abs_delta_allocation", "max"),
            middle_any_large=("abs_delta_allocation", lambda s: bool((s >= args.min_abs_delta).any())),
        )
        .reset_index()
        .merge(
            target[
                [
                    "ballot_id",
                    "intervention",
                    "direction",
                    "target_policy",
                    "strength",
                    "target_candidate_id",
                    "target_delta_allocation",
                    "target_delta_borda",
                    "target_delta_best_pick",
                ]
            ],
            on=["ballot_id", "intervention", "direction", "target_policy", "strength"],
            how="left",
            validate="one_to_one",
        )
    )
    per_ballot_middle["opposite_to_target"] = (
        per_ballot_middle["middle_sum_delta"] * per_ballot_middle["target_delta_allocation"] < 0
    )
    per_ballot_summary = (
        per_ballot_middle.groupby(["intervention", "direction", "target_policy", "strength"], dropna=False)
        .agg(
            n_ballots=("ballot_id", "nunique"),
            mean_target_delta=("target_delta_allocation", "mean"),
            mean_middle_sum_delta=("middle_sum_delta", "mean"),
            mean_middle_abs_delta=("middle_abs_delta", "mean"),
            middle_any_large_rate=("middle_any_large", "mean"),
            opposite_to_target_rate=("opposite_to_target", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(out / "causal_steering_spillover_by_role.csv", index=False)
    middle_summary.to_csv(out / "causal_steering_middle_spillover_summary.csv", index=False)
    strong_middle.to_csv(out / "causal_steering_large_middle_spillovers.csv", index=False)
    per_ballot_summary.to_csv(out / "causal_steering_per_ballot_middle_spillovers.csv", index=False)

    print("\nMiddle-candidate spillover summary")
    print(middle_summary.to_string(index=False))
    print("\nPer-ballot middle spillover summary")
    print(per_ballot_summary.to_string(index=False))
    print(f"\nLarge middle spillovers abs(delta) >= {args.min_abs_delta}")
    cols = [
        "ballot_id",
        "intervention",
        "direction",
        "target_policy",
        "strength",
        "target_candidate_id",
        "target_delta_allocation",
        "candidate_id",
        "baseline_role",
        "baseline_allocation",
        "allocation",
        "delta_allocation",
        "baseline_borda_points",
        "borda_points",
        "delta_borda",
        "baseline_best_pick_vote",
        "best_pick_vote",
        "delta_best_pick",
    ]
    print(strong_middle[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
