#!/usr/bin/env python3
"""Create local figures and scalar geometry checks for paperMech.tex.

This script uses already-saved CSV outputs. It does not require model inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-mech-dir", default="level25_self_answer_mech_outputs_20260709_000748")
    parser.add_argument(
        "--followup-dir",
        default="level25_self_answer_mech_outputs_20260709_000748/local_followup_analysis_20260709_005126",
    )
    parser.add_argument("--visible-small-mech-dir", default="level25_self_answer_mech_outputs_20260617_010546")
    parser.add_argument("--unknown-older-mech-dir", default="level25_self_answer_mech_outputs_20260617_143704")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    return parser.parse_args()


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, default=str) + "\n")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_rank_polarity(followup_dir: Path, out_dir: Path) -> None:
    df = read_csv(followup_dir / "borda_rank_allocation_polarity.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    colors = {"help_rate": "#2f7d5b", "neutral_rate": "#b8b8b8", "hurt_rate": "#b84a4a"}
    labels = {"help_rate": "Helped", "neutral_rate": "Neutral", "hurt_rate": "Hurt"}
    bottom = np.zeros(len(df))
    for col in ["hurt_rate", "neutral_rate", "help_rate"]:
        ax.bar(x, df[col], bottom=bottom, color=colors[col], label=labels[col], width=0.72)
        bottom += df[col].to_numpy()
    ax.set_xticks(x, [str(int(v)) for v in df["borda_rank"]])
    ax.set_xlabel("Borda rank")
    ax.set_ylabel("Fraction of candidates")
    ax.set_ylim(0, 1)
    ax.set_title("Absolute Allocation Polarity Within Borda Rank")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, out_dir, "fig_borda_rank_polarity")


def plot_position_polarity(followup_dir: Path, out_dir: Path) -> None:
    df = read_csv(followup_dir / "position_allocation_polarity.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    colors = {"help_rate": "#2f7d5b", "neutral_rate": "#b8b8b8", "hurt_rate": "#b84a4a"}
    labels = {"help_rate": "Helped", "neutral_rate": "Neutral", "hurt_rate": "Hurt"}
    bottom = np.zeros(len(df))
    for col in ["hurt_rate", "neutral_rate", "help_rate"]:
        ax.bar(x, df[col], bottom=bottom, color=colors[col], label=labels[col], width=0.72)
        bottom += df[col].to_numpy()
    ax.set_xticks(x, [str(int(v)) for v in df["shown_position"]])
    ax.set_xlabel("Shown position")
    ax.set_ylabel("Fraction of candidates")
    ax.set_ylim(0, 1)
    ax.set_title("Position Bias in Signed Treatment")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, out_dir, "fig_position_polarity")


def plot_rank_matched_auc(followup_dir: Path, out_dir: Path) -> None:
    df = read_csv(followup_dir / "rank_matched_probe_summary.csv")
    df = df[(df["borda_rank"] == 2) & (df["feature_set"].isin(["position_only", "activation"]))].copy()
    df["target_label"] = df["target"].str.replace("_within_rank", "", regex=False).str.title()
    targets = ["Help", "Hurt", "Neutral"]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    width = 0.36
    x = np.arange(len(targets))
    for offset, feature, label, color in [
        (-width / 2, "position_only", "Position only", "#777777"),
        (width / 2, "activation", "Activation", "#376fa3"),
    ]:
        vals = []
        for target in targets:
            row = df[(df["target_label"] == target) & (df["feature_set"] == feature)]
            vals.append(float(row["mean_auc"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + offset, vals, width=width, label=label, color=color)
    ax.axhline(0.5, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, targets)
    ax.set_ylim(0.45, 0.68)
    ax.set_ylabel("Grouped CV AUC")
    ax.set_title("Within Rank 2, Activations Add Signal")
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "fig_rank2_matched_auc")


def plot_self_geometry(hidden_mech_dir: Path, out_dir: Path, layer: int) -> pd.DataFrame:
    geom = read_csv(hidden_mech_dir / "candidate_to_self_geometry.csv")
    g = geom[geom["layer_index"].astype(int) == layer].copy()
    g["polarity"] = np.select(
        [g["voter_help_label"].astype(bool), g["voter_hurt_label"].astype(bool)],
        ["Helped", "Hurt"],
        default="Neutral",
    )
    order = ["Helped", "Neutral", "Hurt"]
    summary = (
        g.groupby("polarity")
        .agg(
            n=("candidate_id", "size"),
            mean_distance_to_self=("distance_to_self_answer", "mean"),
            se_distance_to_self=("distance_to_self_answer", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
            mean_cosine_to_self=("cosine_to_self_answer", "mean"),
            mean_allocation=("voter_allocation", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_distance_to_self"], yerr=summary["se_distance_to_self"], color=["#2f7d5b", "#b8b8b8", "#b84a4a"], capsize=3)
    ax.set_xticks(x, summary["polarity"])
    ax.set_ylabel("Mean distance to self-answer")
    ax.set_title("Self-Answer Geometry Is Weakly Ordered")
    save_fig(fig, out_dir, "fig_self_distance_by_polarity")
    return summary


def condition_name(mech_dir: Path, fallback: str) -> str:
    cfg_path = mech_dir / "run_config.csv"
    if not cfg_path.exists():
        return fallback
    cfg = pd.read_csv(cfg_path)
    if "self_answer_visible_reference" in cfg.columns:
        val = str(cfg.iloc[0]["self_answer_visible_reference"]).strip().lower()
        if val in {"true", "1"}:
            return "visible"
        if val in {"false", "0"}:
            return "hidden"
    source = str(cfg.iloc[0].get("level25_output_dir", ""))
    if "hidden" in source.lower():
        return "hidden"
    return fallback


def scalar_geometry_condition_summary(mech_dirs: list[tuple[str, Path]], layer: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fallback_name, mech_dir in mech_dirs:
        geom_path = mech_dir / "candidate_to_self_geometry.csv"
        if not geom_path.exists():
            continue
        condition = condition_name(mech_dir, fallback_name)
        geom = read_csv(geom_path)
        g = geom[geom["layer_index"].astype(int) == layer].copy()
        allocation_col = "voter_allocation" if "voter_allocation" in g.columns else "mean_allocation"
        help_col = "voter_help_label" if "voter_help_label" in g.columns else "help_label"
        hurt_col = "voter_hurt_label" if "voter_hurt_label" in g.columns else "hurt_label"
        top_col = "voter_signed_top_choice" if "voter_signed_top_choice" in g.columns else "signed_allocation_winner"
        rows.append(
            {
                "condition": condition,
                "mech_dir": str(mech_dir),
                "n_rows_layer": int(len(g)),
                "n_prompt_evaluator": int(g[["prompt_id", "evaluator_id"]].drop_duplicates().shape[0]) if "evaluator_id" in g.columns else np.nan,
                "mean_distance_helped": float(g.loc[g[help_col].astype(bool), "distance_to_self_answer"].mean()),
                "mean_distance_neutral": float(g.loc[~g[help_col].astype(bool) & ~g[hurt_col].astype(bool), "distance_to_self_answer"].mean()),
                "mean_distance_hurt": float(g.loc[g[hurt_col].astype(bool), "distance_to_self_answer"].mean()),
                "mean_distance_top": float(g.loc[g[top_col].astype(bool), "distance_to_self_answer"].mean()),
                "mean_distance_not_top": float(g.loc[~g[top_col].astype(bool), "distance_to_self_answer"].mean()),
                "corr_distance_allocation": float(g["distance_to_self_answer"].corr(g[allocation_col])),
                "corr_cosine_allocation": float(g["cosine_to_self_answer"].corr(g[allocation_col])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    hidden_mech_dir = Path(args.hidden_mech_dir)
    followup_dir = Path(args.followup_dir)
    out_dir = Path(args.output_dir) if args.output_dir else hidden_mech_dir / f"local_figures_geometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    plot_rank_polarity(followup_dir, out_dir)
    plot_position_polarity(followup_dir, out_dir)
    plot_rank_matched_auc(followup_dir, out_dir)
    self_summary = plot_self_geometry(hidden_mech_dir, out_dir, args.layer)
    save_table(self_summary, out_dir, "hidden_self_distance_by_polarity")

    condition_summary = scalar_geometry_condition_summary(
        [
            ("visible_small", Path(args.visible_small_mech_dir)),
            ("older_condition_unknown", Path(args.unknown_older_mech_dir)),
            ("hidden_full", hidden_mech_dir),
        ],
        args.layer,
    )
    save_table(condition_summary, out_dir, "scalar_geometry_condition_summary")
    pd.DataFrame(
        [
            {
                "hidden_mech_dir": str(hidden_mech_dir),
                "followup_dir": str(followup_dir),
                "visible_small_mech_dir": args.visible_small_mech_dir,
                "unknown_older_mech_dir": args.unknown_older_mech_dir,
                "layer": args.layer,
            }
        ]
    ).to_csv(out_dir / "run_config.csv", index=False)
    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("Self geometry summary")
    print(self_summary.to_string(index=False))
    print("\nScalar geometry condition summary")
    print(condition_summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
