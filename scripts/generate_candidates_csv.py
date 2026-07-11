#!/usr/bin/env python3
"""
Generate candidate-answer CSVs for the direct-voting experiments.

This is a small wrapper around the same candidate-generation logic used by
level1_direct_vote_eval.py. It takes built-in prompts or a prompts CSV and writes
pre-generated candidates that can later be passed to:

    level1_direct_vote_eval.py --candidates-csv <output>/candidates.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from level1_direct_vote_eval import (
    generate_candidates,
    load_model,
    load_prompts,
    release_model,
    save_table,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--prompts-csv",
        default="",
        help="CSV with prompt_id, domain, and user_prompt columns. Empty uses built-ins.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create a zip archive of the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"candidate_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_csv, args.max_prompts)
    save_table(prompts, out_dir, "prompts")
    save_table(
        pd.DataFrame(
            [
                {
                    "candidate_model": args.candidate_model,
                    "fallback_model": args.fallback_model,
                    "prompts_csv": args.prompts_csv,
                    "num_candidates": args.num_candidates,
                    "max_prompts": args.max_prompts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                }
            ]
        ),
        out_dir,
        "run_config",
    )

    model = load_model(args.candidate_model, args.fallback_model)
    try:
        candidates = generate_candidates(prompts, model, args)
    finally:
        release_model(model)

    save_table(candidates, out_dir, "candidates")

    if not args.no_zip:
        archive = shutil.make_archive(str(out_dir), "zip", out_dir)
        print(f"Created archive {archive}")

    print(f"Saved candidate outputs to {out_dir}")
    print(f"Use with: --candidates-csv \"{out_dir / 'candidates.csv'}\"")


if __name__ == "__main__":
    main()
