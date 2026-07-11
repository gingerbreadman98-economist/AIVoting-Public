#!/usr/bin/env python3
"""Analyze outputs from level25_causal_activation_steering.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    return parser.parse_args()


def summarize_numeric(s: pd.Series) -> dict[str, float]:
    clean = s.dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    summary = pd.read_csv(out / "causal_steering_summary.csv")
    votes = pd.read_csv(out / "causal_steering_vote_rows.csv")
    raw = pd.read_csv(out / "causal_steering_raw_outputs.csv")
    config = pd.read_csv(out / "causal_steering_run_config.csv")
    directions = pd.read_csv(out / "causal_steering_directions.csv")

    print("\nRun config")
    print(config.to_string(index=False))
    print("\nDirections")
    print(directions.to_string(index=False))
    print("\nOriginal summary")
    print(summary.to_string(index=False))

    print("\nRow counts")
    print(
        pd.DataFrame(
            [
                {
                    "vote_rows": len(votes),
                    "raw_rows": len(raw),
                    "ballots": votes["ballot_id"].nunique(),
                    "steered_conditions": raw[raw["condition"] == "steered"].shape[0],
                    "raw_parse_error_rate": float((raw["parse_error"].fillna("").astype(str) != "").mean()),
                    "vote_parse_error_rate": float((votes["parse_error"].fillna("").astype(str) != "").mean()),
                }
            ]
        ).to_string(index=False)
    )

    baseline = votes[votes["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation", "best_pick_vote", "borda_points"]
    ].rename(
        columns={
            "allocation": "baseline_allocation",
            "best_pick_vote": "baseline_best_pick_vote",
            "borda_points": "baseline_borda_points",
        }
    )
    steered = votes[votes["condition"] == "steered"].merge(
        baseline, on=["ballot_id", "candidate_id"], how="left", validate="many_to_one"
    )
    steered["delta_allocation"] = steered["allocation"] - steered["baseline_allocation"]
    steered["delta_borda"] = steered["borda_points"] - steered["baseline_borda_points"]
    steered["delta_best_pick"] = steered["best_pick_vote"] - steered["baseline_best_pick_vote"]
    steered["parse_failed"] = steered["parse_error"].fillna("").astype(str) != ""
    steered["allocation_changed"] = steered["delta_allocation"].abs() > 1e-12
    steered["borda_changed"] = steered["delta_borda"].abs() > 1e-12
    steered["best_pick_changed"] = steered["delta_best_pick"].abs() > 1e-12

    target = steered[steered["is_target_candidate"].astype(bool)].copy()
    non_target = steered[~steered["is_target_candidate"].astype(bool)].copy()

    rows = []
    for keys, group in target.groupby(["intervention", "direction", "target_policy", "strength"], dropna=False):
        intervention, direction, target_policy, strength = keys
        nt = non_target[
            (non_target["intervention"] == intervention)
            & (non_target["direction"] == direction)
            & (non_target["target_policy"] == target_policy)
            & (non_target["strength"] == strength)
        ]
        stats = summarize_numeric(group["delta_allocation"])
        rows.append(
            {
                "intervention": intervention,
                "direction": direction,
                "target_policy": target_policy,
                "strength": strength,
                "n_ballots": group["ballot_id"].nunique(),
                "parse_error_rate": float(group["parse_failed"].mean()),
                "target_mean_delta_alloc": stats["mean"],
                "target_median_delta_alloc": stats["median"],
                "target_q25_delta_alloc": stats["q25"],
                "target_q75_delta_alloc": stats["q75"],
                "target_min_delta_alloc": stats["min"],
                "target_max_delta_alloc": stats["max"],
                "target_change_rate": float(group["allocation_changed"].mean()),
                "target_increase_rate": float((group["delta_allocation"] > 0).mean()),
                "target_decrease_rate": float((group["delta_allocation"] < 0).mean()),
                "target_mean_delta_borda": float(group["delta_borda"].mean()),
                "target_borda_change_rate": float(group["borda_changed"].mean()),
                "target_mean_delta_best_pick": float(group["delta_best_pick"].mean()),
                "target_best_pick_gain_rate": float((group["delta_best_pick"] > 0).mean()),
                "nontarget_mean_abs_delta_alloc": float(nt["delta_allocation"].abs().mean()),
                "nontarget_change_rate": float(nt["allocation_changed"].mean()),
            }
        )
    detailed = pd.DataFrame(rows)
    print("\nDetailed target and spillover summary")
    print(detailed.to_string(index=False))

    # Per-ballot total L1 movement, winner flips, and target winner changes.
    condition_rows = []
    for keys, group in steered.groupby(["intervention", "direction", "target_policy", "strength", "ballot_id"]):
        intervention, direction, target_policy, strength, ballot_id = keys
        base_group = baseline[baseline["ballot_id"] == ballot_id]
        steered_winners = set(group.loc[group["allocation"] == group["allocation"].max(), "candidate_id"].astype(str))
        base_winners = set(base_group.loc[base_group["baseline_allocation"] == base_group["baseline_allocation"].max(), "candidate_id"].astype(str))
        target_rows = group[group["is_target_candidate"].astype(bool)]
        target_id = str(target_rows["candidate_id"].iloc[0]) if len(target_rows) else ""
        condition_rows.append(
            {
                "intervention": intervention,
                "direction": direction,
                "target_policy": target_policy,
                "strength": strength,
                "ballot_id": ballot_id,
                "l1_allocation_movement": float(group["delta_allocation"].abs().sum()),
                "any_allocation_change": bool((group["delta_allocation"].abs() > 1e-12).any()),
                "winner_set_changed": steered_winners != base_winners,
                "target_became_winner": target_id in steered_winners and target_id not in base_winners,
                "target_lost_winner": target_id in base_winners and target_id not in steered_winners,
            }
        )
    movement = pd.DataFrame(condition_rows)
    movement_summary = (
        movement.groupby(["intervention", "direction", "target_policy", "strength"], dropna=False)
        .agg(
            mean_l1_allocation_movement=("l1_allocation_movement", "mean"),
            median_l1_allocation_movement=("l1_allocation_movement", "median"),
            any_allocation_change_rate=("any_allocation_change", "mean"),
            winner_set_change_rate=("winner_set_changed", "mean"),
            target_became_winner_rate=("target_became_winner", "mean"),
            target_lost_winner_rate=("target_lost_winner", "mean"),
        )
        .reset_index()
    )
    print("\nWhole-ballot movement and winner flips")
    print(movement_summary.to_string(index=False))

    # Top examples by absolute target shift.
    examples = target.reindex(target["delta_allocation"].abs().sort_values(ascending=False).index)
    cols = [
        "ballot_id",
        "intervention",
        "strength",
        "candidate_id",
        "display_id",
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
    print("\nLargest absolute target allocation shifts")
    print(examples[cols].head(20).to_string(index=False))

    detailed.to_csv(out / "causal_steering_detailed_analysis.csv", index=False)
    movement_summary.to_csv(out / "causal_steering_movement_analysis.csv", index=False)
    examples[cols].head(100).to_csv(out / "causal_steering_largest_target_shifts.csv", index=False)


if __name__ == "__main__":
    main()
