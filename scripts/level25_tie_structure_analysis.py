#!/usr/bin/env python3
"""
Tie structure of the weak ordinal (Borda-style) ballots, and how Absolute
Allocation behaves within ties.

Reported in paperMech.tex ("What Absolute Allocation Reveals Beyond Weak
Ordinal Ballots" and "Residual Beyond Borda"): the elicited ranking allows
tied groups, so the ordinal object is a weak ranking. This script quantifies
(1) how common ties are, and (2) how often Absolute Allocation distinguishes
candidates that the ranking ties -- including sign disagreements (one tied
candidate helped, the other hurt or neutral), which no rank-only ballot can
express.

Usage:
  python level25_tie_structure_analysis.py --mech-output-dir <dir>
Requires self_answer_vote_labels.csv and activation_row_index.csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_labels(base: Path) -> pd.DataFrame:
    labels = pd.read_csv(base / "self_answer_vote_labels.csv")
    row_index = pd.read_csv(base / "activation_row_index.csv")
    return row_index.merge(
        labels,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )


def analyze(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_ballots = 0
    n_ballots_with_tie = 0
    rank_level_counts: dict[int, int] = {}
    tied_pairs = 0
    tied_pairs_diff_alloc = 0
    tied_pairs_diff_sign = 0
    abs_diffs: list[float] = []
    for (_, _), group in labels.groupby(["prompt_id", "evaluator_id"]):
        borda = group["voter_borda_points"].astype(float).round(6).to_numpy()
        alloc = group["voter_allocation"].astype(float).to_numpy()
        n_ballots += 1
        n_levels = len(np.unique(borda))
        rank_level_counts[n_levels] = rank_level_counts.get(n_levels, 0) + 1
        if n_levels < len(borda):
            n_ballots_with_tie += 1
        for i in range(len(borda)):
            for j in range(i + 1, len(borda)):
                if borda[i] != borda[j]:
                    continue
                tied_pairs += 1
                if abs(alloc[i] - alloc[j]) > 1e-12:
                    tied_pairs_diff_alloc += 1
                    abs_diffs.append(abs(alloc[i] - alloc[j]))
                    if np.sign(alloc[i]) != np.sign(alloc[j]):
                        tied_pairs_diff_sign += 1
    summary = pd.DataFrame(
        [
            {
                "n_ballots": n_ballots,
                "n_ballots_with_tie": n_ballots_with_tie,
                "pct_ballots_with_tie": 100.0 * n_ballots_with_tie / n_ballots,
                "n_tied_pairs": tied_pairs,
                "n_tied_pairs_different_allocation": tied_pairs_diff_alloc,
                "pct_tied_pairs_different_allocation": (
                    100.0 * tied_pairs_diff_alloc / tied_pairs if tied_pairs else np.nan
                ),
                "n_tied_pairs_different_sign": tied_pairs_diff_sign,
                "pct_tied_pairs_different_sign": (
                    100.0 * tied_pairs_diff_sign / tied_pairs if tied_pairs else np.nan
                ),
                "mean_abs_allocation_diff_when_different": (
                    float(np.mean(abs_diffs)) if abs_diffs else np.nan
                ),
            }
        ]
    )
    levels = pd.DataFrame(
        [
            {"n_distinct_rank_levels": k, "n_ballots": v}
            for k, v in sorted(rank_level_counts.items())
        ]
    )
    return summary, levels


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"tie_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(base)
    summary, levels = analyze(labels)
    summary.to_csv(out_dir / "tie_structure_summary.csv", index=False)
    levels.to_csv(out_dir / "rank_level_distribution.csv", index=False)
    pd.DataFrame([{"mech_output_dir": str(base)}]).to_csv(
        out_dir / "run_config.csv", index=False
    )
    print("Tie structure summary")
    print(summary.round(3).to_string(index=False))
    print("\nDistinct rank levels per ballot")
    print(levels.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
