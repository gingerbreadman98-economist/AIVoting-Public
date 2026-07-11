"""Recover strict-allocation failures as a labeled sensitivity analysis.

The strict vLLM voting script rejects ballots whose raw integer cents do not
sum to exactly 100 in absolute value. This utility does not change those
primary outputs. It creates separate recovery files for ballots whose final
raw JSON is parseable and whose signed-allocation cents can be normalized by
their observed L1 total.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from level1_direct_vote_eval import (
    borda_points_from_ranked_groups,
    extract_json,
    normalize_best_pick,
    normalize_candidate_id,
    translate_display_ids,
)


EXPECTED_CANDIDATES = ["A", "B", "C", "D"]
KEY_COLUMNS = ["prompt_id", "evaluator_id"]
PIPELINE_PROTOCOL_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover parseable strict-allocation failures by normalizing raw "
            "signed cents to L1=1. Writes separate sensitivity outputs."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or .zip archives containing Level 2.5 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="allocation_recovery_sensitivity",
        help="Directory for recovered ballot tables and summaries.",
    )
    return parser.parse_args()


def read_csv_from_run(run_path: Path, filename: str) -> pd.DataFrame:
    if run_path.is_dir():
        return pd.read_csv(run_path / filename)
    with zipfile.ZipFile(run_path) as zf:
        matches = [
            name
            for name in zf.namelist()
            if name == filename or name.endswith("/" + filename)
        ]
        if not matches:
            raise FileNotFoundError(f"{filename} not found in {run_path}")
        with zf.open(matches[0]) as fh:
            return pd.read_csv(fh)


def safe_json_loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite cents")
    return result


def get_votes(parsed: dict[str, Any]) -> dict[str, Any]:
    votes = parsed.get("votes")
    if isinstance(votes, dict):
        return votes
    if any(
        key in parsed
        for key in ("best_pick", "borda_ranking", "signed_allocation_cents")
    ):
        return parsed
    raise ValueError("votes missing or not an object")


def recover_one(row: pd.Series, strict_borda: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = extract_json(str(row["final_raw_output"]))
    display_map = safe_json_loads(row.get("display_to_candidate_json", ""), {})
    if not display_map:
        raise ValueError(
            "display-to-candidate mapping was not archived; shuffled ballot cannot be recovered reliably"
        )
    parsed = translate_display_ids(parsed, {str(k): str(v) for k, v in display_map.items()})
    votes = get_votes(parsed)
    allocation_items = votes.get("signed_allocation_cents")
    if not isinstance(allocation_items, list):
        raise ValueError("signed_allocation_cents missing or not a list")

    cents_by_candidate: dict[str, float] = {}
    for item in allocation_items:
        if not isinstance(item, dict):
            raise ValueError("non-object allocation item")
        candidate_id = normalize_candidate_id(item.get("candidate_id", ""))
        if candidate_id not in EXPECTED_CANDIDATES:
            raise ValueError(f"unexpected allocation candidate {candidate_id!r}")
        if candidate_id in cents_by_candidate:
            raise ValueError(f"duplicate allocation candidate {candidate_id!r}")
        if "cents" not in item:
            raise ValueError(f"missing cents for {candidate_id}")
        cents_by_candidate[candidate_id] = finite_float(item["cents"])

    if sorted(cents_by_candidate) != EXPECTED_CANDIDATES:
        raise ValueError(
            "allocation does not contain every candidate exactly once: "
            + ",".join(sorted(cents_by_candidate))
        )

    raw_l1 = sum(abs(value) for value in cents_by_candidate.values())
    if raw_l1 <= 0:
        raise ValueError("allocation absolute total is zero")

    normalized = {
        candidate_id: cents / raw_l1
        for candidate_id, cents in cents_by_candidate.items()
    }

    best_pick = normalize_best_pick(votes.get("best_pick", ""))
    if best_pick not in EXPECTED_CANDIDATES:
        raise ValueError(f"invalid best_pick {best_pick!r}")
    borda_points, borda_repairs = borda_points_from_ranked_groups(
        votes.get("borda_ranking"),
        EXPECTED_CANDIDATES,
        strict=strict_borda,
    )

    common = {
        "prompt_id": row["prompt_id"],
        "domain": row.get("domain", ""),
        "evaluator_model": row.get("evaluator_model", ""),
        "evaluator": row.get("evaluator", ""),
        "evaluator_id": row["evaluator_id"],
        "recovery_source": "normalized_raw_cents_l1",
        "raw_absolute_total": raw_l1,
        "normalization_factor": 1.0 / raw_l1,
        "reported_absolute_cents_total": votes.get("absolute_cents_total", ""),
        "final_validation_errors_json": row.get("final_validation_errors_json", "[]"),
        "retry_count": row.get("retry_count", ""),
        "attempt_count": row.get("attempt_count", ""),
        "raw_output": row.get("final_raw_output", ""),
        "candidate_display_order": row.get("candidate_display_order", ""),
        "ballot_field_order": row.get("ballot_field_order", ""),
        "borda_repair_notes_json": json.dumps(borda_repairs),
    }
    recovered_rows = []
    for candidate_id in EXPECTED_CANDIDATES:
        recovered_rows.append(
            {
                **common,
                "candidate_id": candidate_id,
                "best_pick_vote": int(candidate_id == best_pick),
                "borda_points": borda_points[candidate_id],
                "allocation": normalized[candidate_id],
                "raw_cents": cents_by_candidate[candidate_id],
            }
        )
    ballot_row = {
        **common,
        "candidate_ids": ",".join(EXPECTED_CANDIDATES),
        "best_pick": best_pick,
        "raw_cents_json": json.dumps(cents_by_candidate, sort_keys=True),
        "normalized_allocations_json": json.dumps(normalized, sort_keys=True),
    }
    return recovered_rows, ballot_row


def run_label(path: Path) -> str:
    name = path.name
    if name.endswith(".zip"):
        name = name[:-4]
    return name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_ballots: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for raw_run in args.runs:
        run_path = Path(raw_run)
        label = run_label(run_path)
        diagnostics = read_csv_from_run(run_path, "allocation_validation_diagnostics.csv")
        run_config = read_csv_from_run(run_path, "run_config.csv")
        strict_borda = (
            "strict_borda" in run_config.columns
            and str(run_config.iloc[0]["strict_borda"]).strip().lower() in {"1", "true", "yes"}
        )
        invalid = diagnostics[diagnostics["final_valid"] != True].copy()  # noqa: E712

        recovered_ballots = 0
        run_recovered_rows: list[dict[str, Any]] = []
        for _, row in invalid.iterrows():
            try:
                recovered_rows, ballot_row = recover_one(row, strict_borda=strict_borda)
                recovered_ballots += 1
                for item in recovered_rows:
                    item["run"] = label
                    all_rows.append(item)
                    run_recovered_rows.append(item)
                ballot_row["run"] = label
                all_ballots.append(ballot_row)
            except Exception as exc:
                all_failures.append(
                    {
                        "run": label,
                        "prompt_id": row.get("prompt_id", ""),
                        "evaluator_id": row.get("evaluator_id", ""),
                        "recovery_error": str(exc),
                        "final_validation_errors_json": row.get(
                            "final_validation_errors_json", "[]"
                        ),
                        "final_raw_absolute_total": row.get(
                            "final_raw_absolute_total", ""
                        ),
                    }
                )

        summary_rows.append(
            {
                "run": label,
                "diagnostic_ballots": len(diagnostics),
                "strict_valid_ballots": int((diagnostics["final_valid"] == True).sum()),  # noqa: E712
                "strict_invalid_ballots": len(invalid),
                "recovered_invalid_ballots": recovered_ballots,
                "unrecovered_invalid_ballots": len(invalid) - recovered_ballots,
                "recovered_candidate_rows": recovered_ballots * len(EXPECTED_CANDIDATES),
            }
        )

        # A merged sensitivity run lets downstream analyses use strict-valid
        # ballots plus explicitly labeled normalized recoveries without ever
        # mutating the primary run directory.
        strict_votes = read_csv_from_run(run_path, "direct_votes.csv")
        strict_votes["allocation_recovered_from_invalid"] = False
        recovered_frame = pd.DataFrame(run_recovered_rows)
        if not recovered_frame.empty:
            recovered_frame["allocation_recovered_from_invalid"] = True
            recovered_frame["vote_repair_count"] = pd.to_numeric(
                recovered_frame.get("retry_count"), errors="coerce"
            )
            recovered_frame["allocation_first_pass_valid"] = False
            recovered_frame["allocation_retry_count"] = pd.to_numeric(
                recovered_frame.get("retry_count"), errors="coerce"
            )
            recovered_frame["vote_repairs_json"] = recovered_frame[
                "final_validation_errors_json"
            ]
            recovered_frame["reason"] = ""
            recovered_frame["placeholder_reason"] = False
            for column in strict_votes.columns:
                if column not in recovered_frame.columns:
                    recovered_frame[column] = pd.NA
            recovered_frame = recovered_frame[strict_votes.columns]
            merged = pd.concat([strict_votes, recovered_frame], ignore_index=True)
        else:
            merged = strict_votes.copy()
        merged_dir = output_dir / label
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_dir / "direct_votes.csv", index=False)

    pd.DataFrame(all_rows).to_csv(output_dir / "recovered_direct_votes.csv", index=False)
    pd.DataFrame(all_ballots).to_csv(output_dir / "recovered_ballots.csv", index=False)
    pd.DataFrame(all_failures).to_csv(output_dir / "unrecovered_ballots.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "recovery_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote recovery outputs to {output_dir}")


if __name__ == "__main__":
    main()
