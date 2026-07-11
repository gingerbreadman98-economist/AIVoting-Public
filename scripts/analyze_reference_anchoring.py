#!/usr/bin/env python3
"""Compare paired hidden- and visible-reference Level 2.5 voting runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BALLOT_KEYS = ["prompt_id", "evaluator_id"]
CANDIDATE_KEYS = BALLOT_KEYS + ["candidate_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--visible-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-second-hidden",
        action="store_true",
        help="Allow the second run to be hidden for a hidden-vs-hidden placebo comparison.",
    )
    parser.add_argument("--comparison-label", default="hidden_vs_visible")
    return parser.parse_args()


def read_required(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def config_bool(config: pd.DataFrame, key: str) -> bool:
    if config.empty or key not in config.columns:
        raise ValueError(f"Missing {key!r} in run_config.csv")
    return str(config.iloc[0][key]).strip().lower() in {"1", "true", "yes", "y"}


def save_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> None:
    frame.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def ballot_orders(votes: pd.DataFrame) -> pd.DataFrame:
    counts = votes.groupby(BALLOT_KEYS)["candidate_display_order"].nunique()
    if counts.gt(1).any():
        raise ValueError("A run contains inconsistent candidate display orders within ballots.")
    return votes.drop_duplicates(BALLOT_KEYS)[BALLOT_KEYS + ["candidate_display_order"]]


def winner_set(group: pd.DataFrame, score_column: str) -> set[str]:
    values = pd.to_numeric(group[score_column], errors="coerce")
    maximum = float(values.max())
    return set(group.loc[np.isclose(values, maximum), "candidate_id"].astype(str))


def aggregate_prompt_winners(votes: pd.DataFrame, condition: str) -> pd.DataFrame:
    methods = {
        "best_pick": "best_pick_vote",
        "borda": "borda_points",
        "absolute_allocation": "allocation",
    }
    aggregated = (
        votes.groupby(["prompt_id", "candidate_id"], as_index=False)
        .agg(
            best_pick_vote=("best_pick_vote", "sum"),
            borda_points=("borda_points", "sum"),
            allocation=("allocation", "sum"),
        )
    )
    rows = []
    ballot_counts = votes.drop_duplicates(BALLOT_KEYS).groupby("prompt_id").size()
    for prompt_id, group in aggregated.groupby("prompt_id", sort=True):
        for method, score_column in methods.items():
            winners = winner_set(group, score_column)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "method": method,
                    f"{condition}_winners": ",".join(sorted(winners)),
                    f"{condition}_winner_count": len(winners),
                    "n_paired_ballots": int(ballot_counts.loc[prompt_id]),
                }
            )
    return pd.DataFrame(rows)


def polarity(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.sign(numeric), index=values.index, dtype=int)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def main() -> None:
    args = parse_args()
    hidden_dir = Path(args.hidden_dir)
    visible_dir = Path(args.visible_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_config = read_required(hidden_dir, "run_config.csv")
    visible_config = read_required(visible_dir, "run_config.csv")
    if config_bool(hidden_config, "self_answer_visible_reference"):
        raise ValueError("--hidden-dir is configured as visible-reference.")
    second_visible = config_bool(visible_config, "self_answer_visible_reference")
    if not second_visible and not args.allow_second_hidden:
        raise ValueError("--visible-dir is configured as hidden-reference.")

    hidden_candidates = read_required(hidden_dir, "candidates.csv")
    visible_candidates = read_required(visible_dir, "candidates.csv")
    candidate_column = "candidate" if "candidate" in hidden_candidates.columns else "candidate_id"
    candidate_view_columns = ["prompt_id", candidate_column, "candidate_answer"]
    hc = hidden_candidates[candidate_view_columns].sort_values(
        ["prompt_id", candidate_column]
    ).reset_index(drop=True)
    vc = visible_candidates[candidate_view_columns].sort_values(
        ["prompt_id", candidate_column]
    ).reset_index(drop=True)
    if not hc.equals(vc):
        raise ValueError("Hidden and visible candidate answers are not identical.")

    hidden_self = read_required(hidden_dir, "self_answers.csv")
    visible_self = read_required(visible_dir, "self_answers.csv")
    self_pairs = hidden_self[BALLOT_KEYS + ["self_answer"]].merge(
        visible_self[BALLOT_KEYS + ["self_answer"]],
        on=BALLOT_KEYS,
        how="outer",
        suffixes=("_hidden", "_visible"),
        indicator=True,
        validate="one_to_one",
    )
    common_self = self_pairs[self_pairs["_merge"] == "both"].copy()
    common_self["self_answer_exact_match"] = (
        common_self["self_answer_hidden"] == common_self["self_answer_visible"]
    )
    if len(common_self) != len(hidden_self) or len(common_self) != len(visible_self):
        raise ValueError("Hidden and visible runs do not contain the same self-answer keys.")
    if not common_self["self_answer_exact_match"].all():
        matched = int(common_self["self_answer_exact_match"].sum())
        raise ValueError(
            "Self-answer text is not exactly paired: "
            f"{matched}/{len(common_self)} rows match."
        )

    hidden_votes = read_required(hidden_dir, "direct_votes.csv")
    visible_votes = read_required(visible_dir, "direct_votes.csv")
    for frame in (hidden_votes, visible_votes):
        for column in CANDIDATE_KEYS:
            frame[column] = frame[column].astype(str)

    hidden_orders = ballot_orders(hidden_votes).rename(
        columns={"candidate_display_order": "candidate_display_order_hidden"}
    )
    visible_orders = ballot_orders(visible_votes).rename(
        columns={"candidate_display_order": "candidate_display_order_visible"}
    )
    order_pairs = hidden_orders.merge(
        visible_orders,
        on=BALLOT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    order_pairs["candidate_order_exact_match"] = (
        order_pairs["candidate_display_order_hidden"]
        == order_pairs["candidate_display_order_visible"]
    )
    if not order_pairs["candidate_order_exact_match"].all():
        raise ValueError("Candidate display order differs in one or more paired ballots.")

    common_ballots = order_pairs[BALLOT_KEYS].copy()
    hidden_paired = hidden_votes.merge(common_ballots, on=BALLOT_KEYS, how="inner")
    visible_paired = visible_votes.merge(common_ballots, on=BALLOT_KEYS, how="inner")
    candidate_pairs = hidden_paired.merge(
        visible_paired,
        on=CANDIDATE_KEYS,
        how="inner",
        suffixes=("_hidden", "_visible"),
        validate="one_to_one",
    )
    candidate_pairs["allocation_change"] = (
        candidate_pairs["allocation_visible"] - candidate_pairs["allocation_hidden"]
    )
    candidate_pairs["borda_change"] = (
        candidate_pairs["borda_points_visible"] - candidate_pairs["borda_points_hidden"]
    )
    candidate_pairs["best_pick_changed"] = (
        candidate_pairs["best_pick_vote_visible"]
        != candidate_pairs["best_pick_vote_hidden"]
    )
    candidate_pairs["polarity_hidden"] = polarity(candidate_pairs["allocation_hidden"])
    candidate_pairs["polarity_visible"] = polarity(candidate_pairs["allocation_visible"])
    candidate_pairs["polarity_changed"] = (
        candidate_pairs["polarity_hidden"] != candidate_pairs["polarity_visible"]
    )

    hidden_winners = aggregate_prompt_winners(hidden_paired, "hidden")
    visible_winners = aggregate_prompt_winners(visible_paired, "visible")
    winner_changes = hidden_winners.merge(
        visible_winners,
        on=["prompt_id", "method", "n_paired_ballots"],
        how="inner",
        validate="one_to_one",
    )
    winner_changes["winner_changed"] = (
        winner_changes["hidden_winners"] != winner_changes["visible_winners"]
    )
    winner_summary_parts = []
    for subset_name, subset in (
        ("all_paired", winner_changes),
        ("four_paired_ballots", winner_changes[winner_changes["n_paired_ballots"] == 4]),
    ):
        part = (
            subset.groupby("method", as_index=False)
            .agg(
                n_prompts=("prompt_id", "nunique"),
                n_winner_changes=("winner_changed", "sum"),
                winner_change_rate=("winner_changed", "mean"),
                mean_paired_ballots_per_prompt=("n_paired_ballots", "mean"),
                min_paired_ballots_per_prompt=("n_paired_ballots", "min"),
            )
        )
        part.insert(0, "subset", subset_name)
        winner_summary_parts.append(part)
    winner_summary = pd.concat(winner_summary_parts, ignore_index=True)
    intervals = winner_summary.apply(
        lambda row: wilson_interval(
            int(row["n_winner_changes"]),
            int(row["n_prompts"]),
        ),
        axis=1,
    )
    winner_summary["winner_change_ci95_low"] = [interval[0] for interval in intervals]
    winner_summary["winner_change_ci95_high"] = [interval[1] for interval in intervals]

    polarity_rows = []
    no_repair_mask = (
        pd.to_numeric(candidate_pairs["vote_repair_count_hidden"], errors="coerce").eq(0)
        & pd.to_numeric(candidate_pairs["vote_repair_count_visible"], errors="coerce").eq(0)
    )
    for subset_name, subset in (
        ("all_paired", candidate_pairs),
        ("both_no_repair", candidate_pairs[no_repair_mask]),
    ):
        polarity_rows.append(
            {
                "subset": subset_name,
                "n_candidate_pairs": int(len(subset)),
                "hidden_help_rate": float(subset["allocation_hidden"].gt(0).mean()),
                "hidden_neutral_rate": float(subset["allocation_hidden"].eq(0).mean()),
                "hidden_hurt_rate": float(subset["allocation_hidden"].lt(0).mean()),
                "visible_help_rate": float(subset["allocation_visible"].gt(0).mean()),
                "visible_neutral_rate": float(subset["allocation_visible"].eq(0).mean()),
                "visible_hurt_rate": float(subset["allocation_visible"].lt(0).mean()),
                "polarity_change_rate": float(subset["polarity_changed"].mean()),
                "mean_allocation_change": float(subset["allocation_change"].mean()),
                "mean_absolute_allocation_change": float(
                    subset["allocation_change"].abs().mean()
                ),
            }
        )
    polarity_summary = pd.DataFrame(polarity_rows)

    ballot_pairs = candidate_pairs.drop_duplicates(BALLOT_KEYS).copy()
    ballot_pairs["hidden_clean"] = pd.to_numeric(
        ballot_pairs["vote_repair_count_hidden"], errors="coerce"
    ).eq(0)
    ballot_pairs["visible_clean"] = pd.to_numeric(
        ballot_pairs["vote_repair_count_visible"], errors="coerce"
    ).eq(0)
    repair_pairing = (
        ballot_pairs.groupby(["hidden_clean", "visible_clean"], as_index=False)
        .size()
        .rename(columns={"size": "n_ballots"})
    )

    ballot_changes = (
        candidate_pairs.groupby(BALLOT_KEYS, as_index=False)
        .agg(
            best_pick_changed=("best_pick_changed", "max"),
            any_polarity_changed=("polarity_changed", "max"),
            allocation_l1_change=("allocation_change", lambda values: values.abs().sum()),
        )
    )
    ballot_change_summary = pd.DataFrame(
        [
            {
                "n_ballots": int(len(ballot_changes)),
                "best_pick_change_rate": float(ballot_changes["best_pick_changed"].mean()),
                "any_polarity_change_rate": float(
                    ballot_changes["any_polarity_changed"].mean()
                ),
                "exact_allocation_match_rate": float(
                    ballot_changes["allocation_l1_change"].lt(1e-12).mean()
                ),
                "mean_allocation_l1_change": float(
                    ballot_changes["allocation_l1_change"].mean()
                ),
                "median_allocation_l1_change": float(
                    ballot_changes["allocation_l1_change"].median()
                ),
            }
        ]
    )

    polarity_transitions = (
        candidate_pairs.groupby(["polarity_hidden", "polarity_visible"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    polarity_transitions["hidden_polarity_total"] = polarity_transitions.groupby(
        "polarity_hidden"
    )["n"].transform("sum")
    polarity_transitions["transition_rate_within_hidden_polarity"] = (
        polarity_transitions["n"] / polarity_transitions["hidden_polarity_total"]
    )

    pairing_summary = pd.DataFrame(
        [
            {
                "comparison_label": args.comparison_label,
                "second_run_visible_reference": second_visible,
                "hidden_self_answer_rows": int(len(hidden_self)),
                "visible_self_answer_rows": int(len(visible_self)),
                "paired_self_answers": int(len(common_self)),
                "exact_self_answer_matches": int(common_self["self_answer_exact_match"].sum()),
                "hidden_parsed_ballots": int(len(hidden_orders)),
                "visible_parsed_ballots": int(len(visible_orders)),
                "paired_parsed_ballots": int(len(order_pairs)),
                "exact_candidate_order_matches": int(
                    order_pairs["candidate_order_exact_match"].sum()
                ),
                "paired_prompts": int(order_pairs["prompt_id"].nunique()),
            }
        ]
    )

    save_table(pairing_summary, out_dir, "anchoring_pairing_summary")
    save_table(winner_summary, out_dir, "anchoring_winner_change_summary")
    save_table(winner_changes, out_dir, "anchoring_winner_changes_by_prompt")
    save_table(polarity_summary, out_dir, "anchoring_polarity_summary")
    save_table(repair_pairing, out_dir, "anchoring_repair_pairing")
    save_table(ballot_change_summary, out_dir, "anchoring_ballot_change_summary")
    save_table(polarity_transitions, out_dir, "anchoring_polarity_transitions")
    save_table(candidate_pairs, out_dir, "anchoring_candidate_pairs")

    print("\nPairing summary")
    print(pairing_summary.to_string(index=False))
    print("\nWinner-change summary")
    print(winner_summary.to_string(index=False))
    print("\nPolarity summary")
    print(polarity_summary.to_string(index=False))


if __name__ == "__main__":
    main()
