import argparse
import itertools
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BINARY_TARGETS = [
    "voter_help_label",
    "voter_hurt_label",
    "voter_signed_top_choice",
    "voter_best_pick_vote",
]

CONTINUOUS_TARGETS = [
    "voter_allocation",
    "voter_allocation_centered",
    "voter_allocation_z",
    "voter_positive_mass",
    "voter_negative_mass",
    "voter_signed_abs_share",
    "voter_allocation_rank",
    "voter_borda_points",
    "voter_borda_points_centered",
    "voter_borda_points_z",
]

FIELD_ORDER_CATEGORIES = [
    ",".join(order)
    for order in itertools.permutations(
        ["best_pick", "borda_ranking", "signed_allocation"]
    )
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-pairwise-rows", type=int, default=0)
    return parser.parse_args()


def read_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")


def shown_position(row: pd.Series) -> int:
    order = str(row.get("candidate_display_order", "")).split(",")
    candidate = str(row["candidate_id"])
    try:
        return order.index(candidate) + 1
    except ValueError:
        return np.nan


def group_splits(prompt_ids: pd.Series, n_splits: int = 5):
    groups = prompt_ids.astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError("Need at least two prompt groups for grouped CV.")
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def choose_pca_components(n_train: int, n_features: int, requested: int) -> int:
    if requested <= 0:
        return 0
    return max(1, min(requested, n_train - 1, n_features))


def make_model(kind: str, n_train: int, n_features: int, pca_components: int, seed: int):
    steps = [StandardScaler()]
    n_pca = choose_pca_components(n_train, n_features, pca_components)
    if n_pca > 0:
        steps.append(PCA(n_components=n_pca, random_state=seed))
    if kind == "classification":
        steps.append(
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            )
        )
    else:
        steps.append(Ridge(alpha=100.0))
    return make_pipeline(*steps)


def one_hot_position(positions: pd.Series) -> np.ndarray:
    enc = OneHotEncoder(categories=[[1, 2, 3, 4]], sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(positions.astype(int).to_numpy().reshape(-1, 1))


def one_hot_field_order(labels: pd.DataFrame) -> tuple[np.ndarray, bool]:
    if "ballot_field_order" not in labels.columns:
        return np.empty((len(labels), 0), dtype=float), False
    orders = labels["ballot_field_order"].fillna("").astype(str).str.strip()
    if orders.eq("").all():
        return np.empty((len(labels), 0), dtype=float), False
    invalid = sorted(set(orders) - set(FIELD_ORDER_CATEGORIES))
    if invalid:
        raise ValueError(f"Unexpected ballot_field_order value(s): {invalid}")
    enc = OneHotEncoder(
        categories=[FIELD_ORDER_CATEGORIES],
        sparse_output=False,
        handle_unknown="ignore",
    )
    return enc.fit_transform(orders.to_numpy().reshape(-1, 1)), True


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: pd.Series,
    kind: str,
    pca_components: int,
    seed: int,
) -> dict:
    aucs = []
    r2s = []
    n_folds = 0
    for train_idx, test_idx in group_splits(prompt_ids):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if kind == "classification":
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
        model = make_model(kind, len(train_idx), X.shape[1], pca_components, seed)
        model.fit(X[train_idx], y_train)
        if kind == "classification":
            score = model.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_test, score))
        else:
            pred = model.predict(X[test_idx])
            r2s.append(r2_score(y_test, pred))
        n_folds += 1
    if kind == "classification":
        return {
            "n_folds": n_folds,
            "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
            "std_auc": float(np.std(aucs)) if aucs else np.nan,
        }
    return {
        "n_folds": n_folds,
        "mean_r2": float(np.mean(r2s)) if r2s else np.nan,
        "std_r2": float(np.std(r2s)) if r2s else np.nan,
    }


def coefficient_direction(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: pd.Series,
    seed: int,
) -> np.ndarray:
    coefs = []
    for train_idx, test_idx in group_splits(prompt_ids):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        model.fit(X[train_idx], y_train)
        coefs.append(model.named_steps["logisticregression"].coef_.reshape(-1))
    return np.mean(np.stack(coefs, axis=0), axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def residual_beyond_borda(
    X: np.ndarray,
    labels: pd.DataFrame,
    target: str,
    pca_components: int,
    seed: int,
    context_controls: np.ndarray | None,
    baseline_name: str,
) -> dict:
    y = labels[target].astype(float).to_numpy()
    base_cols = ["voter_borda_points", "voter_borda_points_z", "voter_allocation_rank"]
    base_X = labels[base_cols].astype(float).to_numpy()
    if context_controls is not None and context_controls.shape[1] > 0:
        base_X = np.concatenate([base_X, context_controls], axis=1)
    prompt_ids = labels["prompt_id"]
    r2s = []
    for train_idx, test_idx in group_splits(prompt_ids):
        base = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        base.fit(base_X[train_idx], y[train_idx])
        y_res_train = y[train_idx] - base.predict(base_X[train_idx])
        y_res_test = y[test_idx] - base.predict(base_X[test_idx])
        model = make_model("regression", len(train_idx), X.shape[1], pca_components, seed)
        model.fit(X[train_idx], y_res_train)
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(y_res_test, pred))
    return {
        "target": target,
        "baseline": baseline_name,
        "mean_residual_r2": float(np.mean(r2s)),
        "std_residual_r2": float(np.std(r2s)),
    }


def field_order_behavior_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if "ballot_field_order" not in labels.columns:
        return pd.DataFrame()
    rows = []
    for field_order, group in labels.groupby("ballot_field_order", sort=True):
        allocation = group["voter_allocation"].astype(float)
        rows.append(
            {
                "ballot_field_order": field_order,
                "n_ballots": int(group[["prompt_id", "evaluator_id"]].drop_duplicates().shape[0]),
                "n_candidate_rows": int(len(group)),
                "help_rate": float((allocation > 0).mean()),
                "neutral_rate": float((allocation == 0).mean()),
                "hurt_rate": float((allocation < 0).mean()),
                "best_pick_rate": float(group["voter_best_pick_vote"].astype(bool).mean()),
                "mean_allocation": float(allocation.mean()),
                "mean_absolute_allocation": float(allocation.abs().mean()),
                "mean_repair_count": float(group["vote_repair_count"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_rows(labels: pd.DataFrame, activations: np.ndarray, layer_pos: int) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    rows = []
    ys = []
    groups = []
    # labels and activations are aligned by row_index order.
    for (_, evaluator_id), group in labels.groupby(["prompt_id", "evaluator_id"], sort=False):
        idx = group.index.to_list()
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                i = idx[a_pos]
                j = idx[b_pos]
                yi = float(labels.loc[i, "voter_allocation"])
                yj = float(labels.loc[j, "voter_allocation"])
                if math.isclose(yi, yj):
                    continue
                rows.append(activations[i, layer_pos, :] - activations[j, layer_pos, :])
                ys.append(1 if yi > yj else 0)
                groups.append(labels.loc[i, "prompt_id"])
                rows.append(activations[j, layer_pos, :] - activations[i, layer_pos, :])
                ys.append(1 if yj > yi else 0)
                groups.append(labels.loc[i, "prompt_id"])
    return np.stack(rows, axis=0), np.array(ys, dtype=int), pd.Series(groups)


def main() -> None:
    args = parse_args()
    base = Path(args.mech_output_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base / f"extra_offline_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv(base, "self_answer_vote_labels.csv")
    row_index = read_csv(base, "activation_row_index.csv")
    npz = np.load(base / "self_answer_activations.npz")
    candidate_activations = npz["candidate_activations"]
    layers = npz["layers"].astype(int).tolist()
    if args.layer not in layers:
        raise ValueError(f"Layer {args.layer} not in saved layers: {layers}")
    layer_pos = layers.index(args.layer)

    labels = row_index.merge(
        labels,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    labels["shown_position"] = labels.apply(shown_position, axis=1)
    if labels["shown_position"].isna().any():
        raise ValueError("Could not recover shown_position for every row.")
    X = candidate_activations[:, layer_pos, :]
    X_pos = one_hot_position(labels["shown_position"])
    X_field, field_order_available = one_hot_field_order(labels)
    X_with_pos = np.concatenate([X, X_pos], axis=1)
    X_context = np.concatenate([X_pos, X_field], axis=1)
    X_with_field = np.concatenate([X, X_field], axis=1)
    X_with_context = np.concatenate([X, X_context], axis=1)
    no_repair_mask = labels["vote_repair_count"].astype(int).to_numpy() == 0

    rows = []
    for target in BINARY_TARGETS:
        y = labels[target].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()
        feature_sets = [
            ("position_only", X_pos, 0),
            ("activation_plus_position", X_with_pos, args.pca_components),
            ("activation_layer", X, args.pca_components),
        ]
        if field_order_available:
            feature_sets.extend(
                [
                    ("field_order_only", X_field, 0),
                    ("position_plus_field_order", X_context, 0),
                    ("activation_plus_field_order", X_with_field, args.pca_components),
                    ("activation_plus_position_plus_field_order", X_with_context, args.pca_components),
                ]
            )
        for name, feat, pca in feature_sets:
            result = evaluate_cv(feat, y, labels["prompt_id"], "classification", pca, args.seed)
            rows.append({"target": target, "test": name, "kind": "classification", **result})
        result = evaluate_cv(
            X[no_repair_mask],
            y[no_repair_mask],
            labels.loc[no_repair_mask, "prompt_id"],
            "classification",
            args.pca_components,
            args.seed,
        )
        rows.append({"target": target, "test": "activation_layer_no_repair_ballots", "kind": "classification", **result})

    for target in CONTINUOUS_TARGETS:
        y = labels[target].astype(float).to_numpy()
        feature_sets = [
            ("position_only", X_pos, 0),
            ("activation_plus_position", X_with_pos, args.pca_components),
            ("activation_layer", X, args.pca_components),
        ]
        if field_order_available:
            feature_sets.extend(
                [
                    ("field_order_only", X_field, 0),
                    ("position_plus_field_order", X_context, 0),
                    ("activation_plus_field_order", X_with_field, args.pca_components),
                    ("activation_plus_position_plus_field_order", X_with_context, args.pca_components),
                ]
            )
        for name, feat, pca in feature_sets:
            result = evaluate_cv(feat, y, labels["prompt_id"], "regression", pca, args.seed)
            rows.append({"target": target, "test": name, "kind": "regression", **result})
        result = evaluate_cv(
            X[no_repair_mask],
            y[no_repair_mask],
            labels.loc[no_repair_mask, "prompt_id"],
            "regression",
            args.pca_components,
            args.seed,
        )
        rows.append({"target": target, "test": "activation_layer_no_repair_ballots", "kind": "regression", **result})

    extra_summary = pd.DataFrame(rows)
    save_table(extra_summary, out_dir, "extra_probe_summary")

    residual_rows = []
    for target in [
        "voter_allocation",
        "voter_allocation_centered",
        "voter_allocation_z",
        "voter_positive_mass",
        "voter_negative_mass",
    ]:
        residual_rows.append(
            residual_beyond_borda(
                X, labels, target, args.pca_components, args.seed, None, "borda_only"
            )
        )
        residual_rows.append(
            residual_beyond_borda(
                X, labels, target, args.pca_components, args.seed, X_pos, "borda_plus_position"
            )
        )
        if field_order_available:
            residual_rows.append(
                residual_beyond_borda(
                    X,
                    labels,
                    target,
                    args.pca_components,
                    args.seed,
                    X_context,
                    "borda_plus_position_and_field_order",
                )
            )
    residual_summary = pd.DataFrame(residual_rows)
    save_table(residual_summary, out_dir, "residual_beyond_borda_summary")
    save_table(field_order_behavior_summary(labels), out_dir, "field_order_behavior_summary")

    help_y = labels["voter_help_label"].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()
    hurt_y = labels["voter_hurt_label"].astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()
    help_dir = coefficient_direction(X, help_y, labels["prompt_id"], args.seed)
    hurt_dir = coefficient_direction(X, hurt_y, labels["prompt_id"], args.seed)
    direction_summary = pd.DataFrame(
        [
            {
                "layer": args.layer,
                "help_hurt_direction_cosine": cosine(help_dir, hurt_dir),
                "help_vs_negative_hurt_cosine": cosine(help_dir, -hurt_dir),
                "help_direction_norm": float(np.linalg.norm(help_dir)),
                "hurt_direction_norm": float(np.linalg.norm(hurt_dir)),
            }
        ]
    )
    save_table(direction_summary, out_dir, "help_hurt_direction_summary")

    pair_X, pair_y, pair_groups = pairwise_rows(labels, candidate_activations, layer_pos)
    if args.max_pairwise_rows and len(pair_y) > args.max_pairwise_rows:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(pair_y), size=args.max_pairwise_rows, replace=False)
        pair_X = pair_X[keep]
        pair_y = pair_y[keep]
        pair_groups = pair_groups.iloc[keep].reset_index(drop=True)
    pair_result = evaluate_cv(pair_X, pair_y, pair_groups, "classification", args.pca_components, args.seed)
    pairwise_summary = pd.DataFrame(
        [
            {
                "target": "pairwise_allocation_preference",
                "feature_set": f"layer_{args.layer}_candidate_difference",
                "n_rows": int(len(pair_y)),
                **pair_result,
            }
        ]
    )
    save_table(pairwise_summary, out_dir, "pairwise_contrast_summary")
    save_table(
        pd.DataFrame(
            [
                {
                    "mech_output_dir": str(base),
                    "layer": args.layer,
                    "field_order_available": field_order_available,
                    "field_order_categories_json": json.dumps(FIELD_ORDER_CATEGORIES),
                }
            ]
        ),
        out_dir,
        "context_control_run_config",
    )

    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nExtra probe summary")
    print(extra_summary.to_string(index=False))
    print("\nResidual beyond Borda")
    print(residual_summary.to_string(index=False))
    print("\nHelp/hurt directions")
    print(direction_summary.to_string(index=False))
    print("\nPairwise contrast")
    print(pairwise_summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
