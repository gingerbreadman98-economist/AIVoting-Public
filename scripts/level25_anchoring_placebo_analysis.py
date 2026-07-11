#!/usr/bin/env python3
"""
Anchoring effect vs. sampling-noise placebo for Absolute Allocation elections
(paperMech.tex, "Reference-Answer Anchoring").

The reference-anchoring comparison (visible vs. hidden self-answer) mixes two
sources of winner change: the treatment (showing the reference) and pure
sampling stochasticity, because both conditions are generated independently.
This script separates them with a placebo:

  anchor  (A): hidden reference, seed s0        -> baseline winners
  visible    : visible reference, seed s1        -> treatment
  placebo (B): hidden reference, seed s1         -> same-treatment, seed change

For each aggregation method (best_pick, borda, allocation) it computes, on the
ballots parsed in *every* provided run:

  anchoring effect = winner-change rate (visible vs anchor)
  placebo floor    = winner-change rate (placebo vs anchor)
  corrected effect = effect - floor

When both visible and placebo are provided it also runs a per-prompt McNemar
test (paired over prompts) of whether the treatment changes winners more often
than the seed-only placebo.

Winner change is reported two ways:
  strict          : the tied-for-top candidate set differs at all.
  tie_disjoint    : the two top sets share no candidate (a "real" flip).

Usage:
  python level25_anchoring_placebo_analysis.py \
      --anchor-dir runA_hidden_seed7 \
      --visible-dir runVisible_seed11 \
      --placebo-dir runB_hidden_seed11 \
      [--evaluator-model Qwen/Qwen2.5-7B-Instruct]

--visible-dir and --placebo-dir are each optional; provide either or both.
Reads direct_votes.csv from each run directory.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    "best_pick": "best_pick_vote",
    "borda": "borda_points",
    "allocation": "allocation",
}
KEY = ["prompt_id", "evaluator_id"]
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True, help="Hidden reference anchor run (A).")
    parser.add_argument("--visible-dir", default="", help="Visible reference run (treatment).")
    parser.add_argument("--placebo-dir", default="", help="Hidden reference run at a new seed (placebo B).")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--evaluator-model",
        default="",
        help="Restrict to one evaluator_model. Empty uses all rows in each run.",
    )
    parser.add_argument(
        "--restrict-to-common-ballots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Aggregate winners only over (prompt, evaluator) ballots parsed in every "
            "provided run. Mirrors the paper's parsed-in-both restriction."
        ),
    )
    return parser.parse_args()


def load_votes(run_dir: Path, evaluator_model: str) -> pd.DataFrame:
    path = run_dir / "direct_votes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = set(KEY + ["candidate_id"] + list(METHODS.values()))
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if evaluator_model:
        if "evaluator_model" not in df.columns:
            raise ValueError(f"{path} has no evaluator_model column to filter on.")
        df = df[df["evaluator_model"].astype(str) == str(evaluator_model)].copy()
        if df.empty:
            raise ValueError(f"No rows for evaluator_model {evaluator_model!r} in {path}.")
    for col in KEY + ["candidate_id"]:
        df[col] = df[col].astype(str)
    for col in METHODS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ballot_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, df[KEY].drop_duplicates().to_numpy().tolist()))


def winners_by_method(df: pd.DataFrame) -> dict[str, dict[str, frozenset[str]]]:
    """Return {method: {prompt_id: frozenset(tied-top candidate ids)}}."""
    out: dict[str, dict[str, frozenset[str]]] = {method: {} for method in METHODS}
    for prompt_id, group in df.groupby("prompt_id", sort=False):
        for method, col in METHODS.items():
            sums = group.groupby("candidate_id")[col].sum()
            if sums.empty or not np.isfinite(sums.to_numpy()).any():
                continue
            top = float(sums.max())
            winners = frozenset(str(cid) for cid, val in sums.items() if float(val) >= top - EPS)
            out[method][str(prompt_id)] = winners
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def compare(
    anchor_w: dict[str, dict[str, frozenset[str]]],
    other_w: dict[str, dict[str, frozenset[str]]],
    label: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    """Per-method winner-change rates and per-prompt change indicators."""
    rows = []
    indicators: dict[str, dict[str, dict[str, int]]] = {method: {} for method in METHODS}
    for method in METHODS:
        prompts = sorted(set(anchor_w[method]) & set(other_w[method]))
        n = len(prompts)
        strict_changes = 0
        disjoint_changes = 0
        for prompt_id in prompts:
            a = anchor_w[method][prompt_id]
            b = other_w[method][prompt_id]
            strict = int(a != b)
            disjoint = int(len(a & b) == 0)
            strict_changes += strict
            disjoint_changes += disjoint
            indicators[method][prompt_id] = {"strict": strict, "disjoint": disjoint}
        s_lo, s_hi = wilson_interval(strict_changes, n)
        d_lo, d_hi = wilson_interval(disjoint_changes, n)
        rows.append(
            {
                "comparison": label,
                "method": method,
                "n_prompts": n,
                "strict_change_rate": strict_changes / n if n else float("nan"),
                "strict_wilson_lo": s_lo,
                "strict_wilson_hi": s_hi,
                "tie_disjoint_change_rate": disjoint_changes / n if n else float("nan"),
                "tie_disjoint_wilson_lo": d_lo,
                "tie_disjoint_wilson_hi": d_hi,
            }
        )
    return pd.DataFrame(rows), indicators


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    # two-sided exact binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_test(
    effect_ind: dict[str, dict[str, dict[str, int]]],
    floor_ind: dict[str, dict[str, dict[str, int]]],
    change_kind: str,
) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        prompts = sorted(set(effect_ind[method]) & set(floor_ind[method]))
        # b: treatment changed, placebo did not; c: placebo changed, treatment did not
        b = c = both = neither = 0
        for prompt_id in prompts:
            t = effect_ind[method][prompt_id][change_kind]
            p = floor_ind[method][prompt_id][change_kind]
            if t and not p:
                b += 1
            elif p and not t:
                c += 1
            elif t and p:
                both += 1
            else:
                neither += 1
        n = len(prompts)
        eff = (b + both) / n if n else float("nan")
        flr = (c + both) / n if n else float("nan")
        rows.append(
            {
                "method": method,
                "change_kind": change_kind,
                "n_prompts": n,
                "effect_rate": eff,
                "placebo_rate": flr,
                "corrected_effect": eff - flr if n else float("nan"),
                "discordant_treatment_only_b": b,
                "discordant_placebo_only_c": c,
                "mcnemar_exact_p": mcnemar_exact(b, c),
            }
        )
    return pd.DataFrame(rows)


def voter_level_changes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    comparison: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = KEY + ["candidate_id"]
    columns = keys + ["allocation", "borda_points", "best_pick_vote"]
    paired = left[columns].merge(
        right[columns], on=keys, suffixes=("_left", "_right"), validate="one_to_one"
    )
    paired["allocation_change"] = paired["allocation_right"] - paired["allocation_left"]
    paired["absolute_allocation_change"] = paired["allocation_change"].abs()
    paired["allocation_magnitude_same"] = np.isclose(
        paired["allocation_left"].abs(), paired["allocation_right"].abs()
    )
    paired["allocation_exact_same"] = np.isclose(
        paired["allocation_left"], paired["allocation_right"]
    )
    paired["polarity_changed"] = (
        np.sign(paired["allocation_left"]) != np.sign(paired["allocation_right"])
    )
    paired["borda_changed"] = ~np.isclose(
        paired["borda_points_left"], paired["borda_points_right"]
    )
    paired["best_pick_changed"] = ~np.isclose(
        paired["best_pick_vote_left"], paired["best_pick_vote_right"]
    )
    paired.insert(0, "comparison", comparison)
    ballot = (
        paired.groupby(KEY, as_index=False)
        .agg(
            any_polarity_changed=("polarity_changed", "max"),
            any_borda_changed=("borda_changed", "max"),
            any_best_pick_changed=("best_pick_changed", "max"),
            all_allocations_exact_same=("allocation_exact_same", "all"),
            all_magnitudes_same=("allocation_magnitude_same", "all"),
            allocation_l1_change=("absolute_allocation_change", "sum"),
        )
    )
    ballot.insert(0, "comparison", comparison)
    return paired, ballot


def main() -> None:
    args = parse_args()
    if not args.visible_dir and not args.placebo_dir:
        raise ValueError("Provide at least one of --visible-dir or --placebo-dir.")

    anchor_dir = Path(args.anchor_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else anchor_dir / f"anchoring_placebo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {"anchor": load_votes(anchor_dir, args.evaluator_model)}
    if args.visible_dir:
        runs["visible"] = load_votes(Path(args.visible_dir), args.evaluator_model)
    if args.placebo_dir:
        runs["placebo"] = load_votes(Path(args.placebo_dir), args.evaluator_model)

    common = None
    if args.restrict_to_common_ballots:
        common = set.intersection(*(ballot_keys(df) for df in runs.values()))
        if not common:
            raise ValueError("No (prompt, evaluator) ballots are common to all provided runs.")
        for name, df in runs.items():
            mask = df[KEY].apply(lambda r: (r["prompt_id"], r["evaluator_id"]) in common, axis=1)
            runs[name] = df[mask].copy()

    winners = {name: winners_by_method(df) for name, df in runs.items()}

    summary_frames = []
    all_indicators: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    if "visible" in winners:
        df_eff, ind_eff = compare(winners["anchor"], winners["visible"], "anchoring_effect")
        summary_frames.append(df_eff)
        all_indicators["anchoring_effect"] = ind_eff
    if "placebo" in winners:
        df_flr, ind_flr = compare(winners["anchor"], winners["placebo"], "placebo_floor")
        summary_frames.append(df_flr)
        all_indicators["placebo_floor"] = ind_flr
    if "visible" in winners and "placebo" in winners:
        df_direct, ind_direct = compare(
            winners["placebo"], winners["visible"], "visible_vs_placebo_same_seed"
        )
        summary_frames.append(df_direct)
        all_indicators["visible_vs_placebo_same_seed"] = ind_direct

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(out_dir / "winner_change_rates.csv", index=False)

    candidate_frames = []
    ballot_frames = []
    if "visible" in runs:
        candidate, ballot = voter_level_changes(runs["anchor"], runs["visible"], "anchoring_effect")
        candidate_frames.append(candidate)
        ballot_frames.append(ballot)
    if "placebo" in runs:
        candidate, ballot = voter_level_changes(runs["anchor"], runs["placebo"], "placebo_floor")
        candidate_frames.append(candidate)
        ballot_frames.append(ballot)
    if "visible" in runs and "placebo" in runs:
        candidate, ballot = voter_level_changes(
            runs["placebo"], runs["visible"], "visible_vs_placebo_same_seed"
        )
        candidate_frames.append(candidate)
        ballot_frames.append(ballot)
    candidate_changes = pd.concat(candidate_frames, ignore_index=True)
    ballot_changes = pd.concat(ballot_frames, ignore_index=True)
    candidate_changes.to_csv(out_dir / "candidate_level_changes.csv", index=False)
    ballot_changes.to_csv(out_dir / "ballot_level_changes.csv", index=False)
    candidate_summary = (
        candidate_changes.groupby("comparison", as_index=False)
        .agg(
            n_candidate_pairs=("candidate_id", "size"),
            mean_absolute_allocation_change=("absolute_allocation_change", "mean"),
            median_absolute_allocation_change=("absolute_allocation_change", "median"),
            allocation_exact_same_rate=("allocation_exact_same", "mean"),
            allocation_magnitude_same_rate=("allocation_magnitude_same", "mean"),
            polarity_change_rate=("polarity_changed", "mean"),
            borda_change_rate=("borda_changed", "mean"),
            best_pick_change_rate=("best_pick_changed", "mean"),
        )
    )
    ballot_summary = (
        ballot_changes.groupby("comparison", as_index=False)
        .agg(
            n_ballots=("prompt_id", "size"),
            any_polarity_change_rate=("any_polarity_changed", "mean"),
            any_borda_change_rate=("any_borda_changed", "mean"),
            any_best_pick_change_rate=("any_best_pick_changed", "mean"),
            all_allocations_exact_same_rate=("all_allocations_exact_same", "mean"),
            all_magnitudes_same_rate=("all_magnitudes_same", "mean"),
            mean_allocation_l1_change=("allocation_l1_change", "mean"),
            median_allocation_l1_change=("allocation_l1_change", "median"),
        )
    )
    candidate_summary.to_csv(out_dir / "candidate_level_change_summary.csv", index=False)
    ballot_summary.to_csv(out_dir / "ballot_level_change_summary.csv", index=False)

    corrected = pd.DataFrame()
    if "visible" in winners and "placebo" in winners:
        corrected = pd.concat(
            [
                paired_test(all_indicators["anchoring_effect"], all_indicators["placebo_floor"], "strict"),
                paired_test(all_indicators["anchoring_effect"], all_indicators["placebo_floor"], "disjoint"),
            ],
            ignore_index=True,
        )
        corrected.to_csv(out_dir / "corrected_effect_paired.csv", index=False)

    n_common = len(common) if common is not None else -1
    pd.DataFrame(
        [
            {
                "anchor_dir": str(anchor_dir),
                "visible_dir": args.visible_dir,
                "placebo_dir": args.placebo_dir,
                "evaluator_model": args.evaluator_model,
                "restrict_to_common_ballots": args.restrict_to_common_ballots,
                "n_common_ballots": n_common,
            }
        ]
    ).to_csv(out_dir / "run_config.csv", index=False)

    print("Winner-change rates (strict = top set differs; disjoint = no shared top)")
    print(summary.round(3).to_string(index=False))
    if not corrected.empty:
        print("\nPlacebo-corrected effect (paired over prompts; McNemar exact)")
        print(corrected.round(4).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
