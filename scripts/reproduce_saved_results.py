#!/usr/bin/env python3
"""Rerun CPU analyses from packaged votes and activations."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(args: list[str]) -> None:
    command = [sys.executable, *args]
    print("RUN", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reproduced")
    parser.add_argument("--fast", action="store_true", help="Skip slower sensitivity sweeps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (ROOT / args.output_dir).resolve()
    if ROOT not in output.parents:
        raise ValueError("output directory must remain inside the package")
    output.mkdir(parents=True, exist_ok=True)

    votes = ROOT / "data" / "primary_votes"
    anchoring = output / "reference_display"
    run(
        [
            str(SCRIPTS / "level25_anchoring_placebo_analysis.py"),
            "--anchor-dir", str(votes / "hidden_anchor"),
            "--visible-dir", str(votes / "visible_reference"),
            "--placebo-dir", str(votes / "hidden_placebo"),
            "--evaluator-model", "Qwen/Qwen2.5-7B-Instruct",
            "--output-dir", str(anchoring),
        ]
    )

    steering_source = ROOT / "data" / "steering"
    steering_output = output / "steering"
    steering_output.mkdir(parents=True, exist_ok=True)
    for path in steering_source.glob("causal_steering_*.csv"):
        shutil.copy2(path, steering_output / path.name)
    run([str(SCRIPTS / "analyze_causal_steering_outputs.py"), str(steering_output)])
    run([str(SCRIPTS / "analyze_causal_steering_spillovers.py"), str(steering_output)])
    run([str(SCRIPTS / "analyze_causal_steering_regressions.py"), str(steering_output)])
    run(
        [
            str(SCRIPTS / "level25_steering_intensity_analysis.py"),
            "--steering-output-dir", str(steering_output),
            "--output-dir", str(steering_output / "intensity"),
        ]
    )

    mech = ROOT / "data" / "primary_mechanistic"
    mech_output = output / "mechanistic"
    run(
        [
            str(SCRIPTS / "level25_extra_offline_analysis.py"),
            "--mech-output-dir", str(mech),
            "--output-dir", str(mech_output / "extra_offline_layer16"),
            "--layer", "16",
        ]
    )
    run(
        [
            str(SCRIPTS / "level25_dimensionality_tests.py"),
            "--mech-output-dir", str(mech),
            "--output-dir", str(mech_output / "dim_layer16"),
            "--layer", "16",
        ]
    )
    run(
        [
            str(SCRIPTS / "level25_direction_position_controls.py"),
            "--mech-output-dir", str(mech),
            "--output-dir", str(mech_output / "dirpos_layer16"),
            "--layer", "16",
        ]
    )
    run(
        [
            str(SCRIPTS / "level25_length_controls.py"),
            "--mech-output-dir", str(mech),
            "--output-dir", str(mech_output / "length_layer16"),
            "--layer", "16",
        ]
    )
    if not args.fast:
        run(
            [
                str(SCRIPTS / "level25_cosine_reliability_sensitivity.py"),
                "--mech-output-dir", str(mech),
                "--output-dir", str(mech_output / "cosine_layer16"),
                "--layer", "16",
                "--repeats", "50",
            ]
        )

    run([str(SCRIPTS / "verify_package.py")])
    print(f"Saved reproduced analyses to {output}")


if __name__ == "__main__":
    main()

