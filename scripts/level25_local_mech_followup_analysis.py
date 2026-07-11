#!/usr/bin/env python3
"""
Offline follow-up analyses for Absolute Allocation voter-level mech results.

These analyses require only saved Level 2.5/voter-level outputs:
- self_answer_vote_labels.csv
- candidate_to_self_geometry.csv
- activation_row_index.csv
- self_answer_activations.npz

They add the local analyses requested for the paper:
- Borda rank vs helped/neutral/hurt rates.
- Rank-matched probes: within a fixed Borda rank, can activations distinguish
  helped, hurt, or neutral candidates?
- Neutrality analysis.
- Candidate-self geometry summaries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--min-positive", type=int, default=8)
    return parser.parse_args()


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_borda_rank(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["voter_borda_rank_dense"] = (
        labels.groupby(["prompt_id", "evaluator_id"])["voter_borda_points"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    labels["voter_neutral_label"] = labels["voter_allocation"].astype(float).abs() < 1e-12
    labels["allocation_polarity"] = np.select(
        [
            labels["voter_allocation"].astype(float) > 0,
            labels["voter_allocation"].astype(float) < 0,
        ],
        ["helped", "hurt"],
        default="neutral",
    )
    return labels


def one_hot_position(labels: pd.DataFrame) -> np.ndarray:
    def pos(row: pd.Series) -> int:
        order = [x.strip() for x in str(row.get("candidate_display_order", "")).split(",") if x.strip()]
        cid = str(row["candidate_id"])
        return order.index(cid) + 1 if cid in order else 0

    positions = labels.apply(pos, axis=1).astype(int).to_numpy()
    X = np.zeros((len(labels), 4), dtype=float)
    for i, p in enumerate(positions):
        if 1 <= p <= 4:
            X[i, p - 1] = 1.0
    return X


def shown_position_series(labels: pd.DataFrame) -> pd.Series:
    def pos(row: pd.Series) -> int | float:
        order = [x.strip() for x in str(row.get("candidate_display_order", "")).split(",") if x.strip()]
        cid = str(row["candidate_id"])
        return order.index(cid) + 1 if cid in order else np.nan

    return labels.apply(pos, axis=1)


def evaluate_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    pca_components: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=int)
    unique_groups = pd.Series(groups).astype(str).nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {"n_folds": 0, "mean_auc": np.nan, "std_auc": np.nan, "mean_accuracy": np.nan}
    aucs: list[float] = []
    accs: list[float] = []
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
        if pca_components and pca_components > 0:
            n_comp = min(pca_components, len(train_idx) - 1, X.shape[1])
            if n_comp >= 2:
                steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
        steps.append(
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                    random_state=seed,
                ),
            )
        )
        pipe = make_pipeline(*(step for _, step in steps))
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(float(roc_auc_score(y[test_idx], proba)))
        accs.append(float(accuracy_score(y[test_idx], pred)))
    return {
        "n_folds": len(aucs),
        "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
        "std_auc": float(np.std(aucs)) if aucs else np.nan,
        "mean_accuracy": float(np.mean(accs)) if accs else np.nan,
    }


def load_activation_features(base: Path, labels: pd.DataFrame, layer: int) -> tuple[np.ndarray, list[int]]:
    row_index = read_csv(base, "activation_row_index.csv")
    npz = np.load(base / "self_answer_activations.npz")
    layers = npz["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} not found in saved layers {layers}")
    layer_pos = layers.index(layer)
    merged = row_index.merge(
        labels[["prompt_id", "evaluator_id", "candidate_id"]].reset_index(names="label_row"),
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["label_row"].isna().any():
        raise ValueError("Activation row index does not align with labels.")
    order = merged["label_row"].astype(int).to_numpy()
    X_saved = np.asarray(npz["candidate_activations"], dtype=np.float32)[:, layer_pos, :]
    X = np.empty_like(X_saved)
    X[order] = X_saved
    return X, layers


def rank_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, group in labels.groupby("voter_borda_rank_dense", sort=True):
        n = len(group)
        helped = group["voter_help_label"].astype(bool)
        hurt = group["voter_hurt_label"].astype(bool)
        neutral = group["voter_neutral_label"].astype(bool)
        rows.append(
            {
                "borda_rank": int(rank),
                "n": int(n),
                "help_rate": float(helped.mean()),
                "neutral_rate": float(neutral.mean()),
                "hurt_rate": float(hurt.mean()),
                "mean_allocation": float(group["voter_allocation"].mean()),
                "mean_positive_mass": float(group["voter_positive_mass"].mean()),
                "mean_negative_mass": float(group["voter_negative_mass"].mean()),
                "best_pick_rate": float(group["voter_best_pick_vote"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def position_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["shown_position"] = shown_position_series(labels)
    rows = []
    for pos, group in labels.groupby("shown_position", sort=True):
        rows.append(
            {
                "shown_position": int(pos),
                "n": int(len(group)),
                "help_rate": float(group["voter_help_label"].astype(bool).mean()),
                "neutral_rate": float(group["voter_neutral_label"].astype(bool).mean()),
                "hurt_rate": float(group["voter_hurt_label"].astype(bool).mean()),
                "mean_allocation": float(group["voter_allocation"].mean()),
            }
        )
    return pd.DataFrame(rows)


def field_order_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    if "ballot_field_order" not in labels.columns:
        return pd.DataFrame()
    rows = []
    for field_order, group in labels.groupby("ballot_field_order", sort=True):
        rows.append(
            {
                "ballot_field_order": field_order,
                "n_ballots": int(group[["prompt_id", "evaluator_id"]].drop_duplicates().shape[0]),
                "n_candidate_rows": int(len(group)),
                "help_rate": float(group["voter_help_label"].astype(bool).mean()),
                "neutral_rate": float(group["voter_neutral_label"].astype(bool).mean()),
                "hurt_rate": float(group["voter_hurt_label"].astype(bool).mean()),
                "best_pick_rate": float(group["voter_best_pick_vote"].astype(bool).mean()),
                "mean_allocation": float(group["voter_allocation"].astype(float).mean()),
                "mean_repair_count": float(group["vote_repair_count"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def rank_matched_probes(
    labels: pd.DataFrame,
    X: np.ndarray,
    pca_components: int,
    seed: int,
    min_rows: int,
    min_positive: int,
) -> pd.DataFrame:
    X_pos = one_hot_position(labels)
    rows = []
    targets = [
        ("help_within_rank", "voter_help_label"),
        ("hurt_within_rank", "voter_hurt_label"),
        ("neutral_within_rank", "voter_neutral_label"),
    ]
    for rank, group in labels.groupby("voter_borda_rank_dense", sort=True):
        idx = group.index.to_numpy()
        if len(idx) < min_rows:
            continue
        for target_name, col in targets:
            y = group[col].astype(bool).astype(int).to_numpy()
            positives = int(y.sum())
            negatives = int(len(y) - positives)
            if positives < min_positive or negatives < min_positive:
                continue
            for feature_name, feat, pca in [
                ("position_only", X_pos[idx], 0),
                ("activation", X[idx], pca_components),
                ("activation_plus_position", np.concatenate([X[idx], X_pos[idx]], axis=1), pca_components),
            ]:
                result = evaluate_auc(feat, y, group["prompt_id"], pca, seed)
                rows.append(
                    {
                        "borda_rank": int(rank),
                        "target": target_name,
                        "feature_set": feature_name,
                        "n": int(len(y)),
                        "positive_rate": float(y.mean()),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def neutrality_summary(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["shown_position"] = shown_position_series(labels)
    neutral = labels[labels["voter_neutral_label"]]
    nonneutral = labels[~labels["voter_neutral_label"]]
    rows = [
        {
            "slice": "all",
            "n": int(len(labels)),
            "neutral_rate": float(labels["voter_neutral_label"].mean()),
            "mean_allocation_if_neutral": float(neutral["voter_allocation"].mean()) if len(neutral) else np.nan,
            "mean_abs_allocation_if_nonneutral": float(nonneutral["voter_allocation"].abs().mean()) if len(nonneutral) else np.nan,
            "mean_borda_rank_if_neutral": float(neutral["voter_borda_rank_dense"].mean()) if len(neutral) else np.nan,
            "mean_position_if_neutral": float(neutral["shown_position"].mean()) if len(neutral) else np.nan,
        }
    ]
    for rank, group in labels.groupby("voter_borda_rank_dense", sort=True):
        rows.append(
            {
                "slice": f"borda_rank_{int(rank)}",
                "n": int(len(group)),
                "neutral_rate": float(group["voter_neutral_label"].mean()),
                "mean_allocation_if_neutral": 0.0,
                "mean_abs_allocation_if_nonneutral": float(group.loc[~group["voter_neutral_label"], "voter_allocation"].abs().mean()),
                "mean_borda_rank_if_neutral": float(rank),
                "mean_position_if_neutral": float(group.loc[group["voter_neutral_label"], "shown_position"].mean()),
            }
        )
    return pd.DataFrame(rows)


def candidate_self_summary(geometry: pd.DataFrame, layer: int) -> pd.DataFrame:
    g = geometry[geometry["layer_index"].astype(int) == layer].copy()
    g["neutral_label"] = g["voter_allocation"].astype(float).abs() < 1e-12
    rows = []
    for label, group in [
        ("helped", g[g["voter_help_label"].astype(bool)]),
        ("neutral", g[g["neutral_label"]]),
        ("hurt", g[g["voter_hurt_label"].astype(bool)]),
        ("top_choice", g[g["voter_signed_top_choice"].astype(bool)]),
        ("not_top_choice", g[~g["voter_signed_top_choice"].astype(bool)]),
    ]:
        rows.append(
            {
                "layer": layer,
                "slice": label,
                "n": int(len(group)),
                "mean_distance_to_self": float(group["distance_to_self_answer"].mean()) if len(group) else np.nan,
                "mean_cosine_to_self": float(group["cosine_to_self_answer"].mean()) if len(group) else np.nan,
                "mean_self_rank_closest": float(group["distance_to_self_rank_closest"].mean()) if len(group) else np.nan,
                "mean_allocation": float(group["voter_allocation"].mean()) if len(group) else np.nan,
            }
        )
    rows.append(
        {
            "layer": layer,
            "slice": "correlations",
            "n": int(len(g)),
            "mean_distance_to_self": np.nan,
            "mean_cosine_to_self": np.nan,
            "mean_self_rank_closest": np.nan,
            "mean_allocation": np.nan,
            "corr_distance_allocation": float(g["distance_to_self_answer"].corr(g["voter_allocation"])),
            "corr_distance_positive_mass": float(g["distance_to_self_answer"].corr(g["voter_positive_mass"])),
            "corr_distance_negative_mass": float(g["distance_to_self_answer"].corr(g["voter_negative_mass"])),
            "corr_cosine_allocation": float(g["cosine_to_self_answer"].corr(g["voter_allocation"])),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base / f"local_followup_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = add_borda_rank(read_csv(base, "self_answer_vote_labels.csv"))
    labels = labels.reset_index(drop=True)
    X, layers = load_activation_features(base, labels, args.layer)
    geometry = read_csv(base, "candidate_to_self_geometry.csv")

    rank_table = rank_distribution(labels)
    pos_table = position_distribution(labels)
    field_order_table = field_order_distribution(labels)
    neutral_table = neutrality_summary(labels)
    rank_probe_table = rank_matched_probes(
        labels,
        X,
        args.pca_components,
        args.seed,
        args.min_rows,
        args.min_positive,
    )
    self_table = candidate_self_summary(geometry, args.layer)

    save_table(rank_table, out_dir, "borda_rank_allocation_polarity")
    save_table(pos_table, out_dir, "position_allocation_polarity")
    save_table(field_order_table, out_dir, "field_order_allocation_polarity")
    save_table(neutral_table, out_dir, "neutrality_summary")
    save_table(rank_probe_table, out_dir, "rank_matched_probe_summary")
    save_table(self_table, out_dir, "candidate_self_geometry_summary")
    save_table(
        pd.DataFrame(
            [
                {
                    "mech_output_dir": str(base),
                    "layer": args.layer,
                    "layers_available_json": json.dumps(layers),
                    "pca_components": args.pca_components,
                    "seed": args.seed,
                    "min_rows": args.min_rows,
                    "min_positive": args.min_positive,
                }
            ]
        ),
        out_dir,
        "local_followup_run_config",
    )
    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)

    print("\nBorda rank allocation polarity")
    print(rank_table.to_string(index=False))
    print("\nRank-matched probe summary")
    print(rank_probe_table.to_string(index=False))
    print("\nNeutrality summary")
    print(neutral_table.to_string(index=False))
    print("\nBallot field-order allocation polarity")
    print(field_order_table.to_string(index=False) if not field_order_table.empty else "Unavailable")
    print("\nCandidate-self geometry summary")
    print(self_table.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
