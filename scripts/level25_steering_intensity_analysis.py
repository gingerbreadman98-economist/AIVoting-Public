#!/usr/bin/env python3
"""
Intensity-level analysis of causal steering outputs (paperMech.tex, "Causal
Steering of Ballot Behavior").

Distinguishes direction-specific steering from nonspecific degradation
(regression to the mean) using allocation intensities rather than orderings:

- Degradation predicts steered targets accumulate AT zero allocation
  (unjudgeable candidates get ignored): zero-rate up, neutralization up.
- Steering predicts targets CROSS zero into active support or opposition,
  dose-responsively: sign flips up, zero-rate flat or down.

Reads causal_steering_vote_rows.csv produced by
level25_causal_activation_steering.py. Rows with parse failures (NaN
allocations) are excluded pairwise.

Usage:
  python level25_steering_intensity_analysis.py --steering-output-dir <dir>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
DEEP = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def analyze(vote_rows: pd.DataFrame) -> pd.DataFrame:
    baseline = vote_rows[vote_rows["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation"]
    ].rename(columns={"allocation": "base"})
    steered = vote_rows[
        (vote_rows["condition"] == "steered") & (vote_rows["is_target_candidate"])
    ].merge(baseline, on=["ballot_id", "candidate_id"])
    steered = steered.dropna(subset=["allocation", "base"])
    rows = []
    for (intervention, strength), group in steered.groupby(["intervention", "strength"]):
        base = group["base"].to_numpy(dtype=float)
        steer = group["allocation"].to_numpy(dtype=float)
        upward = str(intervention).endswith("to_bottom")
        if upward:
            crossed = ((base <= EPS) & (steer > EPS)).mean()
            deep = ((base < -EPS) & (steer > DEEP)).mean()
        else:
            crossed = ((base >= -EPS) & (steer < -EPS)).mean()
            deep = ((base > EPS) & (steer < -DEEP)).mean()
        rows.append(
            {
                "intervention": intervention,
                "strength": strength,
                "n": int(len(group)),
                "base_mean_allocation": float(base.mean()),
                "steered_mean_allocation": float(steer.mean()),
                "base_pct_zero": float((np.abs(base) <= EPS).mean() * 100),
                "steered_pct_zero": float((np.abs(steer) <= EPS).mean() * 100),
                "base_pct_positive": float((base > EPS).mean() * 100),
                "steered_pct_positive": float((steer > EPS).mean() * 100),
                "base_pct_negative": float((base < -EPS).mean() * 100),
                "steered_pct_negative": float((steer < -EPS).mean() * 100),
                "neutralized_pct": float(
                    ((np.abs(base) > EPS) & (np.abs(steer) <= EPS)).mean() * 100
                ),
                "predicted_sign_crossing_pct": float(crossed * 100),
                "deep_crossing_pct": float(deep * 100),
            }
        )
    return pd.DataFrame(rows).sort_values(["intervention", "strength"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    base = Path(args.steering_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"steering_intensity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    vote_rows = pd.read_csv(base / "causal_steering_vote_rows.csv")
    summary = analyze(vote_rows)
    summary.to_csv(out_dir / "steering_intensity_summary.csv", index=False)
    pd.DataFrame([{"steering_output_dir": str(base), "deep_threshold": DEEP}]).to_csv(
        out_dir / "run_config.csv", index=False
    )
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
