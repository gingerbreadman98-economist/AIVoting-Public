#!/usr/bin/env python3
"""Sensitivity checks for help/hurt direction cosine reliability.

This reruns the split-half direction analysis under several regularization
strengths and reports both standardized-coordinate and raw-activation-coordinate
cosines. The latter divides logistic coefficients by the StandardScaler scale.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logistic-cs", default="0.01,0.03,0.1,0.3,1.0")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_inputs(base: Path, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    npz = np.load(base / "self_answer_activations.npz")
    layers = npz["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} not in saved layers: {layers}")
    X = npz["candidate_activations"][:, layers.index(layer), :].astype(np.float64)
    labels = pd.read_csv(base / "self_answer_vote_labels.csv")
    row_index = pd.read_csv(base / "activation_row_index.csv")
    labels = row_index.merge(
        labels,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    return X, labels


def bool_column(labels: pd.DataFrame, column: str) -> np.ndarray:
    return labels[column].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def fit_direction(X: np.ndarray, y: np.ndarray, c: float, seed: int, coord: str) -> np.ndarray:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        ),
    )
    pipe.fit(X, y)
    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    clf: LogisticRegression = pipe.named_steps["logisticregression"]
    coef = clf.coef_.reshape(-1).astype(np.float64)
    if coord == "raw_activation":
        coef = coef / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    elif coord != "standardized":
        raise ValueError(coord)
    return coef


def summarize_condition(
    X: np.ndarray,
    labels: pd.DataFrame,
    c: float,
    coord: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    help_y = bool_column(labels, "voter_help_label")
    hurt_y = bool_column(labels, "voter_hurt_label")
    prompt_ids = labels["prompt_id"].astype(str).to_numpy()
    unique_prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)

    full_help = fit_direction(X, help_y, c, seed, coord)
    full_hurt = fit_direction(X, hurt_y, c, seed, coord)
    raw_cos = cosine(full_help, full_hurt)

    help_rel: list[float] = []
    hurt_rel: list[float] = []
    cross: list[float] = []
    corrected: list[float] = []
    invalid_corrections = 0
    for repeat in range(repeats):
        perm = rng.permutation(unique_prompts)
        half_a = set(perm[: len(perm) // 2])
        mask_a = np.isin(prompt_ids, list(half_a))
        mask_b = ~mask_a
        h_a = fit_direction(X[mask_a], help_y[mask_a], c, seed + repeat, coord)
        h_b = fit_direction(X[mask_b], help_y[mask_b], c, seed + repeat, coord)
        t_a = fit_direction(X[mask_a], hurt_y[mask_a], c, seed + repeat, coord)
        t_b = fit_direction(X[mask_b], hurt_y[mask_b], c, seed + repeat, coord)
        rel_h = cosine(h_a, h_b)
        rel_t = cosine(t_a, t_b)
        cross_cos = 0.5 * (cosine(h_a, t_b) + cosine(h_b, t_a))
        help_rel.append(rel_h)
        hurt_rel.append(rel_t)
        cross.append(cross_cos)
        if rel_h > 0 and rel_t > 0:
            corrected.append(cross_cos / np.sqrt(rel_h * rel_t))
        else:
            invalid_corrections += 1

    return {
        "coord": coord,
        "logistic_c": c,
        "repeats": repeats,
        "raw_help_hurt_cosine": raw_cos,
        "mean_help_reliability": float(np.mean(help_rel)),
        "std_help_reliability": float(np.std(help_rel)),
        "mean_hurt_reliability": float(np.mean(hurt_rel)),
        "std_hurt_reliability": float(np.std(hurt_rel)),
        "mean_cross_half_cosine": float(np.mean(cross)),
        "std_cross_half_cosine": float(np.std(cross)),
        "mean_corrected_cosine": float(np.mean(corrected)) if corrected else np.nan,
        "median_corrected_cosine": float(np.median(corrected)) if corrected else np.nan,
        "std_corrected_cosine": float(np.std(corrected)) if corrected else np.nan,
        "min_corrected_cosine": float(np.min(corrected)) if corrected else np.nan,
        "max_corrected_cosine": float(np.max(corrected)) if corrected else np.nan,
        "n_valid_corrections": len(corrected),
        "n_invalid_corrections": invalid_corrections,
    }


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"cosine_reliability_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    X, labels = load_inputs(base, args.layer)
    rows = []
    for c in parse_float_list(args.logistic_cs):
        for coord in ["standardized", "raw_activation"]:
            rows.append(summarize_condition(X, labels, c, coord, args.repeats, args.seed))
    summary = pd.DataFrame(rows)
    summary.insert(0, "layer", args.layer)
    summary.to_csv(out_dir / "cosine_reliability_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "mech_output_dir": str(base),
                "layer": args.layer,
                "repeats": args.repeats,
                "seed": args.seed,
                "logistic_cs": args.logistic_cs,
            }
        ]
    ).to_csv(out_dir / "run_config.csv", index=False)
    with (out_dir / "cosine_reliability_sensitivity.jsonl").open("w", encoding="utf-8") as f:
        for row in summary.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
