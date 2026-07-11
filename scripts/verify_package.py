#!/usr/bin/env python3
"""Verify package completeness and headline values using only stdlib."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"missing required file: {relative}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find(data: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in data if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def close(actual: str | float, expected: float, tolerance: float = 5e-4) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, found {value}")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest() -> int:
    manifest = ROOT / "metadata" / "MANIFEST.sha256"
    if not manifest.is_file():
        raise AssertionError("missing metadata/MANIFEST.sha256")
    checked = 0
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / Path(relative)
        if not path.is_file():
            raise AssertionError(f"manifest file missing: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"manifest hash mismatch: {relative}")
        checked += 1
    return checked


def verify_paper_assets() -> None:
    paper = ROOT / "paper" / "paperMechConcise.tex"
    source = paper.read_text(encoding="utf-8")
    if not (ROOT / "paper" / "references.bib").is_file():
        raise AssertionError("missing paper/references.bib")
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        if not (paper.parent / relative).is_file():
            raise AssertionError(f"missing paper figure: {relative}")


def main() -> int:
    required = [
        "paper/paperMechConcise.tex",
        "paper/references.bib",
        "data/inputs/candidates_qwen_corpus.csv",
        "data/primary_mechanistic/self_answer_activations.npz",
        "data/steering/causal_steering_vote_rows.csv",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            raise AssertionError(f"missing required file: {relative}")

    manifest_files = verify_manifest()
    verify_paper_assets()

    probes = rows("data/primary_mechanistic/self_geometry_best_probe_layers.csv")
    close(find(probes, target="voter_best_pick_vote")["selection_value"], 0.7762475711)
    close(find(probes, target="voter_signed_top_choice")["selection_value"], 0.8291265709)
    close(find(probes, target="voter_allocation_z")["selection_value"], 0.2419139024)

    residual = rows(
        "data/primary_mechanistic/extra_offline_layer16/residual_beyond_borda_summary.csv"
    )
    close(find(residual, target="voter_allocation", baseline="borda_only")["mean_residual_r2"], -0.035705337)
    close(find(residual, target="voter_allocation_z", baseline="borda_only")["mean_residual_r2"], 0.002942572)

    dimensions = rows("data/primary_mechanistic/dim_layer16/score_dimensionality.csv")
    close(find(dimensions, test="hurt_from_full")["mean_auc"], 0.659208233)
    close(find(dimensions, test="hurt_from_help_score_only")["mean_auc"], 0.628743701)
    close(find(dimensions, test="help_from_hurt_score_only")["mean_auc"], 0.646308994)

    reference = rows("data/reference_display/winner_change_rates.csv")
    close(find(reference, comparison="visible_vs_placebo_same_seed", method="best_pick")["strict_change_rate"], 0.45)
    close(find(reference, comparison="placebo_floor", method="best_pick")["strict_change_rate"], 0.09)

    regressions = rows("data/steering/causal_steering_regressions.csv")
    signed = find(regressions, model="main_target_delta", term="signed_strength")
    random = find(regressions, model="main_target_delta", term="random_strength")
    close(signed["coef"], 0.182466, 1e-3)
    close(signed["cluster_se"], 0.048912, 1e-3)
    close(random["coef"], -0.028439, 1e-3)
    close(random["cluster_se"], 0.038976, 1e-3)
    if int(float(signed["clusters"])) != 50 or int(float(signed["n"])) != 1873:
        raise AssertionError("unexpected steering effective sample size")

    raw = rows("data/steering/causal_steering_raw_outputs.csv")
    steered = [row for row in raw if row.get("condition") == "steered"]
    native = sum(row.get("strict_allocation_valid", "").lower() == "true" for row in steered)
    normalized = sum(
        row.get("allocation_recovered_from_invalid", "").lower() == "true" for row in steered
    )
    unusable = len(steered) - native - normalized
    if (len(steered), native, normalized, unusable) != (2100, 213, 1660, 227):
        raise AssertionError(
            f"unexpected steering validity counts: {(len(steered), native, normalized, unusable)}"
        )

    noise = rows("data/steering/causal_steering_baseline_noise.csv")[0]
    close(noise["mean_abs_delta_allocation"], 0.041161, 1e-3)

    print("PASS: package files and headline values verified")
    print(f"  manifest: {manifest_files} SHA-256 hashes verified")
    print("  probes: best-pick 0.776, allocation-top 0.829, allocation-z R2 0.242")
    print("  steering: signed +0.182 (SE 0.049), random -0.028 (SE 0.039)")
    print("  steering validity: 213 native, 1660 normalized, 227 unusable")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
