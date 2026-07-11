#!/usr/bin/env python3
"""Check whether hurt labels target competitive alternatives or low rank.

Uses saved voter-level rows and raw Borda rankings. Outputs:
- ballot allocation sign pattern counts
- Borda rank by allocation sign and magnitude
- tie/rank construction diagnostics
- strict-ranking-only comparison
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", default="level25_self_answer_mech_outputs_20260709_000748")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, default=str) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def normalize_group(candidate: Any) -> str:
    return str(candidate).strip()


def parse_borda_groups(raw_output: str) -> list[list[str]]:
    parsed = extract_json_object(raw_output)
    votes = parsed.get("votes", parsed)
    groups = votes.get("borda_ranking")
    if not isinstance(groups, list):
        return []
    clean: list[list[str]] = []
    for group in groups:
        if isinstance(group, list):
            clean_group = [normalize_group(x) for x in group]
        else:
            clean_group = [normalize_group(group)]
        clean_group = [x for x in clean_group if x]
        if clean_group:
            clean.append(clean_group)
    return clean


def add_rank_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["dense_borda_rank"] = (
        rows.groupby(["prompt_id", "evaluator_id"])["voter_borda_points"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    rows["competition_rank"] = (
        rows.groupby(["prompt_id", "evaluator_id"])["voter_borda_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    group_infos = []
    for (prompt_id, evaluator_id), group in rows.groupby(["prompt_id", "evaluator_id"], sort=False):
        raw = str(group.iloc[0]["raw_output"])
        try:
            groups = parse_borda_groups(raw)
            sizes = [len(g) for g in groups]
            candidate_to_group_index = {}
            candidate_to_group_size = {}
            start_rank = 1
            candidate_to_start_rank = {}
            for group_index, ranked_group in enumerate(groups, start=1):
                for candidate_id in ranked_group:
                    candidate_to_group_index[candidate_id] = group_index
                    candidate_to_group_size[candidate_id] = len(ranked_group)
                    candidate_to_start_rank[candidate_id] = start_rank
                start_rank += len(ranked_group)
            pattern = "-".join(str(s) for s in sizes)
            strict = sizes == [1, 1, 1, 1]
            parse_error = ""
        except Exception as exc:
            groups = []
            pattern = ""
            strict = False
            parse_error = repr(exc)
            candidate_to_group_index = {}
            candidate_to_group_size = {}
            candidate_to_start_rank = {}
        for row in group.itertuples(index=False):
            cid = str(row.candidate_id)
            group_infos.append(
                {
                    "prompt_id": prompt_id,
                    "evaluator_id": evaluator_id,
                    "candidate_id": cid,
                    "borda_group_index_raw": candidate_to_group_index.get(cid, np.nan),
                    "borda_group_size": candidate_to_group_size.get(cid, np.nan),
                    "borda_start_rank_raw": candidate_to_start_rank.get(cid, np.nan),
                    "borda_group_pattern": pattern,
                    "borda_is_strict_ranking": strict,
                    "borda_parse_error": parse_error,
                }
            )
    info = pd.DataFrame(group_infos)
    return rows.merge(info, on=["prompt_id", "evaluator_id", "candidate_id"], how="left", validate="one_to_one")


def ballot_pattern_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(["prompt_id", "evaluator_id"], sort=False):
        alloc = group["voter_allocation"].astype(float)
        rows.append(
            {
                "n_positive": int((alloc > 0).sum()),
                "n_negative": int((alloc < 0).sum()),
                "n_zero": int((alloc == 0).sum()),
                "pattern": f"{int((alloc > 0).sum())} pos, {int((alloc < 0).sum())} neg, {int((alloc == 0).sum())} zero",
            }
        )
    out = pd.DataFrame(rows)
    counts = out.value_counts(["n_positive", "n_negative", "n_zero", "pattern"]).reset_index(name="n_ballots")
    counts["pct_ballots"] = counts["n_ballots"] / counts["n_ballots"].sum()
    return counts.sort_values(["n_ballots"], ascending=False)


def sign_rank_table(df: pd.DataFrame, rank_col: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["sign"] = np.select(
        [work["voter_allocation"] > 0, work["voter_allocation"] < 0],
        ["help", "hurt"],
        default="neutral",
    )
    counts = (
        work.groupby([rank_col, "sign"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["help", "neutral", "hurt"], fill_value=0)
        .reset_index()
        .rename(columns={rank_col: "rank"})
    )
    counts.insert(0, "rank_definition", prefix)
    pct = counts.copy()
    totals = pct[["help", "neutral", "hurt"]].sum(axis=1)
    for col in ["help", "neutral", "hurt"]:
        pct[col] = pct[col] / totals
    magnitude = (
        work.groupby(rank_col)["voter_allocation"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
        .rename(columns={rank_col: "rank"})
    )
    hurt_mag = (
        work[work["voter_allocation"] < 0]
        .groupby(rank_col)["voter_allocation"]
        .agg(hurt_count="count", mean_hurt_allocation="mean", median_hurt_allocation="median")
        .reset_index()
        .rename(columns={rank_col: "rank"})
    )
    magnitude = magnitude.merge(hurt_mag, on="rank", how="left")
    magnitude.insert(0, "rank_definition", prefix)
    return counts, pct, magnitude


def tie_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ballot_level = (
        df.drop_duplicates(["prompt_id", "evaluator_id"])
        [["prompt_id", "evaluator_id", "borda_group_pattern", "borda_is_strict_ranking", "borda_parse_error"]]
        .copy()
    )
    pattern_counts = ballot_level.value_counts(["borda_group_pattern", "borda_is_strict_ranking"]).reset_index(name="n_ballots")
    pattern_counts["pct_ballots"] = pattern_counts["n_ballots"] / len(ballot_level)
    rank2 = df[df["dense_borda_rank"] == 2].copy()
    rank2_summary = (
        rank2.groupby(["borda_group_size", "borda_group_pattern"])
        .agg(
            n_candidates=("candidate_id", "size"),
            hurt_rate=("voter_hurt_label", lambda s: float(s.astype(bool).mean())),
            neutral_rate=("voter_allocation", lambda s: float((s == 0).mean())),
            mean_allocation=("voter_allocation", "mean"),
        )
        .reset_index()
        .sort_values(["n_candidates"], ascending=False)
    )
    return pattern_counts, rank2_summary


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base / f"competitive_opposition_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base / "self_answer_vote_rows_with_text.csv")
    df = add_rank_diagnostics(df)

    patterns = ballot_pattern_counts(df)
    save_table(patterns, out_dir, "ballot_allocation_sign_patterns")

    all_tables = []
    pct_tables = []
    mag_tables = []
    for rank_col, prefix in [
        ("dense_borda_rank", "dense_points_rank"),
        ("competition_rank", "competition_min_rank"),
        ("borda_start_rank_raw", "raw_group_start_rank"),
    ]:
        counts, pct, mag = sign_rank_table(df, rank_col, prefix)
        all_tables.append(counts)
        pct_tables.append(pct)
        mag_tables.append(mag)
    counts_all = pd.concat(all_tables, ignore_index=True)
    pct_all = pd.concat(pct_tables, ignore_index=True)
    mag_all = pd.concat(mag_tables, ignore_index=True)
    save_table(counts_all, out_dir, "borda_rank_sign_counts")
    save_table(pct_all, out_dir, "borda_rank_sign_proportions")
    save_table(mag_all, out_dir, "borda_rank_allocation_magnitudes")

    strict = df[df["borda_is_strict_ranking"]].copy()
    strict_counts, strict_pct, strict_mag = sign_rank_table(strict, "dense_borda_rank", "strict_rank_only") if len(strict) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    save_table(strict_counts, out_dir, "strict_rank_sign_counts")
    save_table(strict_pct, out_dir, "strict_rank_sign_proportions")
    save_table(strict_mag, out_dir, "strict_rank_allocation_magnitudes")

    pattern_counts, rank2_summary = tie_diagnostics(df)
    save_table(pattern_counts, out_dir, "borda_group_pattern_counts")
    save_table(rank2_summary, out_dir, "dense_rank2_by_tie_pattern")

    config = pd.DataFrame([{"mech_output_dir": str(base), "n_rows": len(df), "n_ballots": df.groupby(["prompt_id", "evaluator_id"]).ngroups}])
    save_table(config, out_dir, "run_config")
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)

    print("\nBallot allocation sign patterns")
    print(patterns.to_string(index=False))
    print("\nDense Borda rank sign proportions")
    print(pct_all[pct_all["rank_definition"] == "dense_points_rank"].to_string(index=False))
    print("\nDense Borda rank allocation magnitudes")
    print(mag_all[mag_all["rank_definition"] == "dense_points_rank"].to_string(index=False))
    print("\nBorda group pattern counts")
    print(pattern_counts.to_string(index=False))
    print("\nDense rank 2 by tie pattern")
    print(rank2_summary.to_string(index=False))
    print("\nStrict ranking only proportions")
    print(strict_pct.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive}")


if __name__ == "__main__":
    main()
