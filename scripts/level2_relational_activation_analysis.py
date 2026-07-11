#!/usr/bin/env python3
"""
Level 2 relational mechanistic activation analysis for direct-vote aggregation.

This script consumes a Level 1 direct-vote output directory and asks:

1. Are candidate-level hidden states linearly predictive of direct signed
   allocation outcomes such as help, hurt, severity, and aggregate winner?
2. Do those signals improve when candidate activations are represented
   relative to the other candidates in the same prompt/election?
3. Which layers and feature constructions carry the strongest signal?
4. Do contrast vectors between aggregate winners and losers align with the
   same dimensions that separate helped from hurt candidates?

It does not call hosted APIs. It uses a local Hugging Face causal LM and reads
the already-generated Level 1 candidates/votes/judge outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CANDIDATE_LABELS = ["A", "B", "C", "D"]


@dataclass
class CandidateSpan:
    candidate_id: str
    char_start: int
    char_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level1-output-dir",
        required=True,
        help="Directory containing Level 1 outputs such as candidates.csv and direct_votes.csv.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--layers",
        default="every4",
        help=(
            "Layer selection: all, every4, last, or comma-separated zero-based "
            "hidden-state indices. Hidden state 0 is embeddings; transformer "
            "layers start at 1."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="How to pool candidate answer span activations.",
    )
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Reuse activations.npz in --output-dir if present.",
    )
    parser.add_argument(
        "--save-activation-matrix",
        action="store_true",
        help="Save the full activation tensor to activations.npz. Can be large.",
    )
    parser.add_argument(
        "--feature-modes",
        default="raw,prompt_centered,raw_plus_centered",
        help=(
            "Comma-separated feature modes for probes. Options: raw, "
            "prompt_centered, prompt_zscore, raw_plus_centered, "
            "raw_plus_centered_plus_prompt_mean."
        ),
    )
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def read_csv_required(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def parse_tie_selection(selection: Any) -> set[str]:
    text = str(selection)
    if text.startswith("TIE:"):
        return {part.strip() for part in text[4:].split(",") if part.strip()}
    return {text.strip()}


def tie_aware_winner(scores: dict[str, float], tol: float = 1e-12) -> str:
    max_score = max(scores.values())
    winners = sorted(label for label, score in scores.items() if abs(score - max_score) <= tol)
    if len(winners) == 1:
        return winners[0]
    return "TIE:" + ",".join(winners)


def build_candidate_metrics(base: Path, max_prompts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = read_csv_required(base, "prompts.csv")
    candidates = read_csv_required(base, "candidates.csv")
    votes = read_csv_required(base, "direct_votes.csv")

    if max_prompts > 0:
        prompt_ids = prompts["prompt_id"].drop_duplicates().head(max_prompts).tolist()
        prompts = prompts[prompts["prompt_id"].isin(prompt_ids)].copy()
        candidates = candidates[candidates["prompt_id"].isin(prompt_ids)].copy()
        votes = votes[votes["prompt_id"].isin(prompt_ids)].copy()

    candidate_col = "candidate" if "candidate" in candidates.columns else "candidate_id"
    candidates = candidates.rename(columns={candidate_col: "candidate_id"})

    vote_metrics = (
        votes.groupby(["prompt_id", "candidate_id"], as_index=False)
        .agg(
            mean_allocation=("allocation", "mean"),
            sum_allocation=("allocation", "sum"),
            positive_rate=("allocation", lambda s: float((s > 0).mean())),
            negative_rate=("allocation", lambda s: float((s < 0).mean())),
            positive_mass=("allocation", lambda s: float(s[s > 0].sum())),
            negative_mass=("allocation", lambda s: float((-s[s < 0]).sum())),
            mean_positive_severity=(
                "allocation",
                lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            ),
            mean_negative_severity=(
                "allocation",
                lambda s: float((-s[s < 0]).mean()) if (s < 0).any() else 0.0,
            ),
            zero_rate=("allocation", lambda s: float((s == 0).mean())),
            best_pick_rate=("best_pick_vote", "mean"),
            mean_borda=("borda_points", "mean"),
            n_vote_rows=("allocation", "count"),
        )
    )

    aggregate_rows = []
    for prompt_id, group in vote_metrics.groupby("prompt_id"):
        alloc_scores = dict(zip(group["candidate_id"], group["sum_allocation"]))
        best_scores = dict(zip(group["candidate_id"], group["best_pick_rate"]))
        borda_scores = dict(zip(group["candidate_id"], group["mean_borda"]))
        alloc_winners = parse_tie_selection(tie_aware_winner(alloc_scores))
        best_winners = parse_tie_selection(tie_aware_winner(best_scores))
        borda_winners = parse_tie_selection(tie_aware_winner(borda_scores))
        for candidate_id in group["candidate_id"]:
            aggregate_rows.append(
                {
                    "prompt_id": prompt_id,
                    "candidate_id": candidate_id,
                    "signed_allocation_winner": candidate_id in alloc_winners,
                    "best_pick_winner": candidate_id in best_winners,
                    "borda_winner": candidate_id in borda_winners,
                }
            )
    aggregate_flags = pd.DataFrame(aggregate_rows)

    metrics = candidates.merge(vote_metrics, on=["prompt_id", "candidate_id"], how="left")
    metrics = metrics.merge(aggregate_flags, on=["prompt_id", "candidate_id"], how="left")

    judge_path = base / "external_judge_results.csv"
    if judge_path.exists():
        judge = pd.read_csv(judge_path)
        judge_flags = []
        for row in judge.itertuples(index=False):
            winners = parse_tie_selection(getattr(row, "judge_winner"))
            for candidate_id in CANDIDATE_LABELS:
                judge_flags.append(
                    {
                        "prompt_id": row.prompt_id,
                        "candidate_id": candidate_id,
                        "judge_winner": candidate_id in winners,
                        "judge_winner_text": getattr(row, "judge_winner"),
                        "judge_consensus_share": getattr(row, "judge_consensus_share", np.nan),
                    }
                )
        metrics = metrics.merge(pd.DataFrame(judge_flags), on=["prompt_id", "candidate_id"], how="left")
    else:
        metrics["judge_winner"] = False
        metrics["judge_winner_text"] = ""
        metrics["judge_consensus_share"] = np.nan

    metrics["help_label"] = metrics["mean_allocation"] > 0
    metrics["hurt_label"] = metrics["mean_allocation"] < 0
    metrics["positive_share"] = metrics["positive_mass"] / metrics["n_vote_rows"].clip(lower=1)
    metrics["negative_share"] = metrics["negative_mass"] / metrics["n_vote_rows"].clip(lower=1)
    metrics["signed_abs_share"] = (
        metrics["positive_mass"] + metrics["negative_mass"]
    ) / metrics["n_vote_rows"].clip(lower=1)
    metrics["strong_help_label"] = metrics["mean_allocation"] >= metrics["mean_allocation"].quantile(0.75)
    metrics["strong_hurt_label"] = metrics["mean_allocation"] <= metrics["mean_allocation"].quantile(0.25)

    relative_targets = [
        "mean_allocation",
        "positive_rate",
        "negative_rate",
        "positive_mass",
        "negative_mass",
        "positive_share",
        "negative_share",
        "signed_abs_share",
        "mean_positive_severity",
        "mean_negative_severity",
        "mean_borda",
        "best_pick_rate",
    ]
    for target in relative_targets:
        if target not in metrics.columns:
            continue
        prompt_mean = metrics.groupby("prompt_id")[target].transform("mean")
        prompt_std = metrics.groupby("prompt_id")[target].transform("std").replace(0, np.nan)
        metrics[f"{target}_centered"] = metrics[target] - prompt_mean
        metrics[f"{target}_z_within_prompt"] = (
            (metrics[target] - prompt_mean) / prompt_std
        ).fillna(0.0)

    metrics["allocation_rank_within_prompt"] = metrics.groupby("prompt_id")["mean_allocation"].rank(
        method="average",
        ascending=False,
    )
    return prompts, metrics


def build_mechanistic_prompt(
    user_prompt: str,
    candidate_rows: pd.DataFrame,
) -> tuple[str, list[CandidateSpan]]:
    parts = [
        "You are evaluating candidate answers for a direct-voting election.",
        "",
        "Original user prompt:",
        user_prompt,
        "",
        "Candidate answers:",
    ]
    spans: list[CandidateSpan] = []
    text = "\n".join(parts)
    for row in candidate_rows.sort_values("candidate_id").itertuples(index=False):
        header = f"\n\nCandidate {row.candidate_id}:\n"
        text += header
        start = len(text)
        answer = str(row.candidate_answer)
        text += answer
        end = len(text)
        spans.append(CandidateSpan(row.candidate_id, start, end))
    text += (
        "\n\nThink about which candidates deserve positive support, negative "
        "opposition, or neutrality under signed allocation."
    )
    return text, spans


def parse_layers(layer_spec: str, n_hidden_states: int) -> list[int]:
    if layer_spec == "all":
        return list(range(n_hidden_states))
    if layer_spec == "last":
        return [n_hidden_states - 1]
    if layer_spec.startswith("every"):
        step = int(layer_spec.replace("every", "") or "4")
        layers = list(range(0, n_hidden_states, step))
        if layers[-1] != n_hidden_states - 1:
            layers.append(n_hidden_states - 1)
        return layers
    layers = [int(part.strip()) for part in layer_spec.split(",") if part.strip()]
    return [layer if layer >= 0 else n_hidden_states + layer for layer in layers]


def load_model_and_tokenizer(
    model_name: str,
    dtype: str,
    device: str,
    attn_implementation: str = "",
    revision: str = "",
):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[dtype],
        "device_map": device,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    revision_kwargs = {"revision": revision} if revision else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            **revision_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, **revision_kwargs)
    except Exception as causal_exc:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            **revision_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        multimodal_cls = (
            getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "AutoModelForMultimodalLM", None)
            or getattr(transformers, "Gemma3ForConditionalGeneration", None)
        )
        if multimodal_cls is None:
            raise causal_exc
        model = multimodal_cls.from_pretrained(model_name, **kwargs, **revision_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def char_span_to_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if tok_end <= start or tok_start >= end:
            continue
        token_indices.append(idx)
    return token_indices


def extract_activations(
    prompts: pd.DataFrame,
    metrics: pd.DataFrame,
    model_name: str,
    dtype: str,
    device: str,
    layer_spec: str,
    pooling: str,
    max_model_len: int,
) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    import torch

    model, tokenizer = load_model_and_tokenizer(model_name, dtype, device)

    prompt_lookup = prompts.set_index("prompt_id")["user_prompt"].to_dict()
    rows = []
    activation_rows = []
    selected_layers: list[int] | None = None

    for prompt_id, group in tqdm(metrics.groupby("prompt_id"), desc="Extracting activations"):
        prompt_text, spans = build_mechanistic_prompt(prompt_lookup[prompt_id], group)
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_model_len,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states
        if selected_layers is None:
            selected_layers = parse_layers(layer_spec, len(hidden_states))
        assert selected_layers is not None

        for span in spans:
            token_indices = char_span_to_token_indices(offsets, span.char_start, span.char_end)
            if not token_indices:
                continue
            if pooling == "last":
                pooled_indices = [token_indices[-1]]
            else:
                pooled_indices = token_indices
            layer_vectors = []
            for layer_idx in selected_layers:
                vec = hidden_states[layer_idx][0, pooled_indices, :].mean(dim=0)
                layer_vectors.append(vec.detach().float().cpu().numpy())
            activation_rows.append(np.stack(layer_vectors, axis=0))
            rows.append({"prompt_id": prompt_id, "candidate_id": span.candidate_id})

        del output, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not activation_rows:
        raise RuntimeError("No activation rows were extracted. Check truncation/max-model-len.")
    activations = np.stack(activation_rows, axis=0)
    row_index = pd.DataFrame(rows)
    return activations, row_index, selected_layers or []


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def prompt_group_folds(prompt_ids: pd.Series, n_folds: int = 5, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(prompt_ids.unique()))
    rng.shuffle(unique)
    folds = np.array_split(unique, min(n_folds, len(unique)))
    result = []
    prompt_arr = prompt_ids.to_numpy()
    for fold in folds:
        test_mask = np.isin(prompt_arr, fold)
        train_mask = ~test_mask
        result.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return result


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def parse_feature_modes(feature_modes: str) -> list[str]:
    valid = {
        "raw",
        "prompt_centered",
        "prompt_zscore",
        "raw_plus_centered",
        "raw_plus_centered_plus_prompt_mean",
    }
    modes = [part.strip() for part in feature_modes.split(",") if part.strip()]
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"Unknown feature mode(s): {', '.join(unknown)}")
    return modes or ["raw"]


def build_feature_matrix(
    layer_activations: np.ndarray,
    dataset: pd.DataFrame,
    feature_mode: str,
) -> np.ndarray:
    prompt_ids = dataset["prompt_id"].to_numpy()
    prompt_mean = np.zeros_like(layer_activations)
    prompt_std = np.ones_like(layer_activations)
    for prompt_id in sorted(dataset["prompt_id"].unique()):
        mask = prompt_ids == prompt_id
        values = layer_activations[mask]
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        prompt_mean[mask] = mean
        prompt_std[mask] = std

    centered = layer_activations - prompt_mean
    zscore = centered / prompt_std
    if feature_mode == "raw":
        return layer_activations
    if feature_mode == "prompt_centered":
        return centered
    if feature_mode == "prompt_zscore":
        return zscore
    if feature_mode == "raw_plus_centered":
        return np.concatenate([layer_activations, centered], axis=1)
    if feature_mode == "raw_plus_centered_plus_prompt_mean":
        return np.concatenate([layer_activations, centered, prompt_mean], axis=1)
    raise ValueError(f"Unhandled feature mode: {feature_mode}")


def run_layerwise_probes(
    activations: np.ndarray,
    dataset: pd.DataFrame,
    layers: list[int],
    seed: int,
    feature_modes: list[str],
    logistic_c: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

    targets = [
        ("help_label", "classification"),
        ("hurt_label", "classification"),
        ("signed_allocation_winner", "classification"),
        ("judge_winner", "classification"),
        ("mean_allocation", "regression"),
        ("positive_rate", "regression"),
        ("negative_rate", "regression"),
        ("positive_mass", "regression"),
        ("negative_mass", "regression"),
        ("positive_share", "regression"),
        ("negative_share", "regression"),
        ("signed_abs_share", "regression"),
        ("mean_positive_severity", "regression"),
        ("mean_negative_severity", "regression"),
        ("mean_allocation_centered", "regression"),
        ("positive_mass_centered", "regression"),
        ("negative_mass_centered", "regression"),
        ("positive_share_centered", "regression"),
        ("negative_share_centered", "regression"),
        ("signed_abs_share_centered", "regression"),
        ("mean_positive_severity_centered", "regression"),
        ("mean_negative_severity_centered", "regression"),
        ("mean_allocation_z_within_prompt", "regression"),
        ("positive_mass_z_within_prompt", "regression"),
        ("negative_mass_z_within_prompt", "regression"),
        ("signed_abs_share_z_within_prompt", "regression"),
        ("allocation_rank_within_prompt", "regression"),
    ]
    folds = prompt_group_folds(dataset["prompt_id"], n_folds=5, seed=seed)
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        layer_activations = activations[:, layer_pos, :]
        for feature_mode in feature_modes:
            X = build_feature_matrix(layer_activations, dataset, feature_mode)
            for target, kind in targets:
                if target not in dataset.columns:
                    continue
                y_raw = dataset[target]
                if y_raw.isna().all():
                    continue
                fold_metrics = []
                for train_idx, test_idx in folds:
                    y_train = y_raw.iloc[train_idx].to_numpy()
                    y_test = y_raw.iloc[test_idx].to_numpy()
                    if kind == "classification":
                        y_train = y_train.astype(bool).astype(int)
                        y_test = y_test.astype(bool).astype(int)
                        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                            continue
                        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
                        clf = LogisticRegression(
                            C=logistic_c,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=seed,
                        )
                        clf.fit(X_train, y_train)
                        probs = clf.predict_proba(X_test)[:, 1]
                        preds = (probs >= 0.5).astype(int)
                        fold_metrics.append(
                            {
                                "accuracy": accuracy_score(y_test, preds),
                                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                                "auc": safe_auc(y_test, probs),
                            }
                        )
                    else:
                        y_train = y_train.astype(float)
                        y_test = y_test.astype(float)
                        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
                        reg = Ridge(alpha=ridge_alpha)
                        reg.fit(X_train, y_train)
                        preds = reg.predict(X_test)
                        fold_metrics.append(
                            {
                                "rmse": math.sqrt(mean_squared_error(y_test, preds)),
                                "r2": r2_score(y_test, preds),
                            }
                        )
                if not fold_metrics:
                    continue
                metric_names = sorted({key for metric in fold_metrics for key in metric})
                row = {
                    "layer_index": layer_idx,
                    "layer_position": layer_pos,
                    "feature_mode": feature_mode,
                    "feature_dim": int(X.shape[1]),
                    "target": target,
                    "kind": kind,
                    "n_folds": len(fold_metrics),
                }
                for metric_name in metric_names:
                    values = [metric[metric_name] for metric in fold_metrics if metric_name in metric]
                    row[f"mean_{metric_name}"] = float(np.nanmean(values))
                    row[f"std_{metric_name}"] = float(np.nanstd(values))
                rows.append(row)
    return pd.DataFrame(rows)


def run_within_prompt_winner_probes(
    activations: np.ndarray,
    dataset: pd.DataFrame,
    layers: list[int],
    seed: int,
    feature_modes: list[str],
    logistic_c: float,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    targets = [
        "signed_allocation_winner",
        "judge_winner",
        "best_pick_winner",
        "borda_winner",
    ]
    folds = prompt_group_folds(dataset["prompt_id"], n_folds=5, seed=seed)
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        layer_activations = activations[:, layer_pos, :]
        for feature_mode in feature_modes:
            X = build_feature_matrix(layer_activations, dataset, feature_mode)
            for target in targets:
                if target not in dataset.columns:
                    continue
                y_raw = dataset[target]
                fold_values = []
                for train_idx, test_idx in folds:
                    y_train = y_raw.iloc[train_idx].astype(bool).astype(int).to_numpy()
                    if len(set(y_train.tolist())) < 2:
                        continue
                    X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
                    clf = LogisticRegression(
                        C=logistic_c,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    )
                    clf.fit(X_train, y_train)
                    scores = clf.predict_proba(X_test)[:, 1]
                    test_frame = dataset.iloc[test_idx][["prompt_id", "candidate_id", target]].copy()
                    test_frame["score"] = scores
                    prompt_hits = []
                    for _, group in test_frame.groupby("prompt_id"):
                        predicted = set(group.loc[group["score"] == group["score"].max(), "candidate_id"])
                        actual = set(group.loc[group[target].astype(bool), "candidate_id"])
                        if not actual:
                            continue
                        prompt_hits.append(bool(predicted & actual))
                    if prompt_hits:
                        fold_values.append(float(np.mean(prompt_hits)))
                if fold_values:
                    rows.append(
                        {
                            "layer_index": layer_idx,
                            "layer_position": layer_pos,
                            "feature_mode": feature_mode,
                            "target": target,
                            "n_folds": len(fold_values),
                            "mean_prompt_top1_match": float(np.mean(fold_values)),
                            "std_prompt_top1_match": float(np.std(fold_values)),
                        }
                    )
    return pd.DataFrame(rows)


def allocation_pattern_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id, group in dataset.groupby("prompt_id"):
        positives = int((group["mean_allocation"] > 0).sum())
        negatives = int((group["mean_allocation"] < 0).sum())
        zeros = int((group["mean_allocation"] == 0).sum())
        rows.append(
            {
                "prompt_id": prompt_id,
                "positive_candidates": positives,
                "negative_candidates": negatives,
                "zero_candidates": zeros,
                "pattern": f"+{positives}/-{negatives}/0{zeros}",
            }
        )
    return pd.DataFrame(rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def row_id_map(dataset: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(row.prompt_id), str(row.candidate_id)): idx
        for idx, row in enumerate(dataset.itertuples(index=False))
    }


def activation_path_long_table(
    activations: np.ndarray,
    dataset: pd.DataFrame,
    layers: list[int],
) -> pd.DataFrame:
    """Candidate-level trajectory features for every layer.

    This is the most literal "activation path" output: each candidate has one
    row per layer with distances to the prompt centroid, signed winner centroid,
    judge winner centroid, and its previous-layer movement.
    """
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        X = activations[:, layer_pos, :]
        prev_X = activations[:, layer_pos - 1, :] if layer_pos > 0 else None
        for prompt_id, group in dataset.groupby("prompt_id", sort=True):
            idx = group.index.to_numpy()
            values = X[idx]
            centroid = values.mean(axis=0)

            signed_mask = group["signed_allocation_winner"].astype(bool).to_numpy()
            judge_mask = group["judge_winner"].astype(bool).to_numpy()
            signed_centroid = values[signed_mask].mean(axis=0) if signed_mask.any() else None
            judge_centroid = values[judge_mask].mean(axis=0) if judge_mask.any() else None

            for local_pos, row_idx in enumerate(idx):
                row = dataset.loc[row_idx]
                vec = X[row_idx]
                centered = vec - centroid
                out = {
                    "prompt_id": prompt_id,
                    "domain": row.get("domain", ""),
                    "candidate_id": row["candidate_id"],
                    "layer_index": layer_idx,
                    "layer_position": layer_pos,
                    "activation_norm": float(np.linalg.norm(vec)),
                    "centered_norm": float(np.linalg.norm(centered)),
                    "distance_to_prompt_centroid": float(np.linalg.norm(centered)),
                    "cosine_to_prompt_centroid": cosine_similarity(vec, centroid),
                    "mean_allocation": float(row.get("mean_allocation", np.nan)),
                    "positive_mass": float(row.get("positive_mass", np.nan)),
                    "negative_mass": float(row.get("negative_mass", np.nan)),
                    "signed_abs_share": float(row.get("signed_abs_share", np.nan)),
                    "allocation_rank_within_prompt": float(row.get("allocation_rank_within_prompt", np.nan)),
                    "help_label": bool(row.get("help_label", False)),
                    "hurt_label": bool(row.get("hurt_label", False)),
                    "signed_allocation_winner": bool(row.get("signed_allocation_winner", False)),
                    "judge_winner": bool(row.get("judge_winner", False)),
                }
                if signed_centroid is not None:
                    out["distance_to_signed_winner_centroid"] = float(np.linalg.norm(vec - signed_centroid))
                    out["cosine_to_signed_winner_centroid"] = cosine_similarity(vec, signed_centroid)
                else:
                    out["distance_to_signed_winner_centroid"] = np.nan
                    out["cosine_to_signed_winner_centroid"] = np.nan
                if judge_centroid is not None:
                    out["distance_to_judge_winner_centroid"] = float(np.linalg.norm(vec - judge_centroid))
                    out["cosine_to_judge_winner_centroid"] = cosine_similarity(vec, judge_centroid)
                else:
                    out["distance_to_judge_winner_centroid"] = np.nan
                    out["cosine_to_judge_winner_centroid"] = np.nan
                if prev_X is not None:
                    prev_vec = prev_X[row_idx]
                    out["step_distance_from_previous_layer"] = float(np.linalg.norm(vec - prev_vec))
                    out["step_cosine_from_previous_layer"] = cosine_similarity(prev_vec, vec)
                else:
                    out["step_distance_from_previous_layer"] = np.nan
                    out["step_cosine_from_previous_layer"] = np.nan
                rows.append(out)
    return pd.DataFrame(rows)


def pairwise_distance_table(
    activations: np.ndarray,
    dataset: pd.DataFrame,
    layers: list[int],
) -> pd.DataFrame:
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        X = activations[:, layer_pos, :]
        for prompt_id, group in dataset.groupby("prompt_id", sort=True):
            group = group.sort_values("candidate_id")
            idx = group.index.to_numpy()
            records = group.to_dict(orient="records")
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    vec_i = X[idx[i]]
                    vec_j = X[idx[j]]
                    alloc_i = float(records[i].get("mean_allocation", np.nan))
                    alloc_j = float(records[j].get("mean_allocation", np.nan))
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "domain": records[i].get("domain", ""),
                            "layer_index": layer_idx,
                            "layer_position": layer_pos,
                            "candidate_i": records[i]["candidate_id"],
                            "candidate_j": records[j]["candidate_id"],
                            "euclidean_distance": float(np.linalg.norm(vec_i - vec_j)),
                            "cosine_similarity": cosine_similarity(vec_i, vec_j),
                            "allocation_i": alloc_i,
                            "allocation_j": alloc_j,
                            "allocation_difference": alloc_i - alloc_j,
                            "abs_allocation_difference": abs(alloc_i - alloc_j),
                            "same_help_hurt_sign": np.sign(alloc_i) == np.sign(alloc_j),
                            "i_signed_winner": bool(records[i].get("signed_allocation_winner", False)),
                            "j_signed_winner": bool(records[j].get("signed_allocation_winner", False)),
                            "i_judge_winner": bool(records[i].get("judge_winner", False)),
                            "j_judge_winner": bool(records[j].get("judge_winner", False)),
                        }
                    )
    return pd.DataFrame(rows)


def layer_geometry_summary(
    path_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for layer_idx, group in path_df.groupby("layer_index", sort=True):
        pair_group = pairwise_df[pairwise_df["layer_index"] == layer_idx]
        helped = group[group["help_label"]]
        hurt = group[group["hurt_label"]]
        signed_winners = group[group["signed_allocation_winner"]]
        signed_losers = group[~group["signed_allocation_winner"]]
        judge_winners = group[group["judge_winner"]]
        judge_losers = group[~group["judge_winner"]]
        rows.append(
            {
                "layer_index": layer_idx,
                "n_candidates": int(len(group)),
                "mean_distance_to_centroid": float(group["distance_to_prompt_centroid"].mean()),
                "mean_distance_to_centroid_helped": float(helped["distance_to_prompt_centroid"].mean()),
                "mean_distance_to_centroid_hurt": float(hurt["distance_to_prompt_centroid"].mean()),
                "mean_distance_to_centroid_signed_winners": float(
                    signed_winners["distance_to_prompt_centroid"].mean()
                ),
                "mean_distance_to_centroid_signed_losers": float(
                    signed_losers["distance_to_prompt_centroid"].mean()
                ),
                "mean_distance_to_centroid_judge_winners": float(
                    judge_winners["distance_to_prompt_centroid"].mean()
                ),
                "mean_distance_to_centroid_judge_losers": float(
                    judge_losers["distance_to_prompt_centroid"].mean()
                ),
                "corr_centroid_distance_mean_allocation": safe_corr(
                    group["distance_to_prompt_centroid"].to_numpy(),
                    group["mean_allocation"].to_numpy(),
                ),
                "corr_centroid_distance_abs_allocation": safe_corr(
                    group["distance_to_prompt_centroid"].to_numpy(),
                    np.abs(group["mean_allocation"].to_numpy()),
                ),
                "corr_centroid_distance_positive_mass": safe_corr(
                    group["distance_to_prompt_centroid"].to_numpy(),
                    group["positive_mass"].to_numpy(),
                ),
                "corr_centroid_distance_negative_mass": safe_corr(
                    group["distance_to_prompt_centroid"].to_numpy(),
                    group["negative_mass"].to_numpy(),
                ),
                "corr_distance_to_signed_winner_mean_allocation": safe_corr(
                    group["distance_to_signed_winner_centroid"].to_numpy(),
                    group["mean_allocation"].to_numpy(),
                ),
                "corr_distance_to_judge_winner_mean_allocation": safe_corr(
                    group["distance_to_judge_winner_centroid"].to_numpy(),
                    group["mean_allocation"].to_numpy(),
                ),
                "mean_pairwise_distance": float(pair_group["euclidean_distance"].mean())
                if not pair_group.empty
                else np.nan,
                "corr_pairwise_distance_abs_allocation_difference": safe_corr(
                    pair_group["euclidean_distance"].to_numpy(),
                    pair_group["abs_allocation_difference"].to_numpy(),
                )
                if not pair_group.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prompt_geometry_summary(
    path_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (prompt_id, layer_idx), group in path_df.groupby(["prompt_id", "layer_index"], sort=True):
        pair_group = pairwise_df[
            (pairwise_df["prompt_id"] == prompt_id) & (pairwise_df["layer_index"] == layer_idx)
        ]
        domain = group["domain"].iloc[0] if "domain" in group.columns and len(group) else ""
        rows.append(
            {
                "prompt_id": prompt_id,
                "domain": domain,
                "layer_index": layer_idx,
                "mean_distance_to_centroid": float(group["distance_to_prompt_centroid"].mean()),
                "max_distance_to_centroid": float(group["distance_to_prompt_centroid"].max()),
                "distance_spread_to_centroid": float(
                    group["distance_to_prompt_centroid"].max()
                    - group["distance_to_prompt_centroid"].min()
                ),
                "mean_pairwise_distance": float(pair_group["euclidean_distance"].mean())
                if not pair_group.empty
                else np.nan,
                "max_pairwise_distance": float(pair_group["euclidean_distance"].max())
                if not pair_group.empty
                else np.nan,
                "allocation_spread": float(group["mean_allocation"].max() - group["mean_allocation"].min()),
                "signed_abs_share_sum": float(group["signed_abs_share"].sum()),
                "corr_candidate_distance_allocation": safe_corr(
                    group["distance_to_prompt_centroid"].to_numpy(),
                    group["mean_allocation"].to_numpy(),
                ),
                "corr_pair_distance_alloc_difference": safe_corr(
                    pair_group["euclidean_distance"].to_numpy(),
                    pair_group["abs_allocation_difference"].to_numpy(),
                )
                if not pair_group.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def contrast_analysis(activations: np.ndarray, dataset: pd.DataFrame, layers: list[int]) -> pd.DataFrame:
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        X = activations[:, layer_pos, :]
        helped = dataset["mean_allocation"].to_numpy() > 0
        hurt = dataset["mean_allocation"].to_numpy() < 0
        signed_winner = dataset["signed_allocation_winner"].astype(bool).to_numpy()
        judge_winner = dataset["judge_winner"].astype(bool).to_numpy()

        def mean_vec(mask: np.ndarray) -> np.ndarray | None:
            if mask.sum() == 0:
                return None
            return X[mask].mean(axis=0)

        help_vec = mean_vec(helped)
        hurt_vec = mean_vec(hurt)
        signed_vec = mean_vec(signed_winner)
        signed_non_vec = mean_vec(~signed_winner)
        judge_vec = mean_vec(judge_winner)
        judge_non_vec = mean_vec(~judge_winner)

        if help_vec is None or hurt_vec is None:
            continue
        help_minus_hurt = help_vec - hurt_vec
        row = {
            "layer_index": layer_idx,
            "help_minus_hurt_norm": float(np.linalg.norm(help_minus_hurt)),
        }
        if signed_vec is not None and signed_non_vec is not None:
            signed_contrast = signed_vec - signed_non_vec
            row["signed_winner_contrast_norm"] = float(np.linalg.norm(signed_contrast))
            row["cos_help_hurt_vs_signed_winner"] = cosine_similarity(
                help_minus_hurt,
                signed_contrast,
            )
        if judge_vec is not None and judge_non_vec is not None:
            judge_contrast = judge_vec - judge_non_vec
            row["judge_winner_contrast_norm"] = float(np.linalg.norm(judge_contrast))
            row["cos_help_hurt_vs_judge_winner"] = cosine_similarity(
                help_minus_hurt,
                judge_contrast,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    level1_dir = Path(args.level1_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"level2_mech_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, metrics = build_candidate_metrics(level1_dir, args.max_prompts)
    save_table(prompts, out_dir, "prompts_used")
    save_table(metrics, out_dir, "candidate_mechanistic_labels")

    activation_path = out_dir / "activations.npz"
    if args.skip_activations and activation_path.exists():
        archive = np.load(activation_path, allow_pickle=True)
        activations = archive["activations"]
        row_index = pd.DataFrame(archive["row_index"].tolist())
        layers = archive["layers"].astype(int).tolist()
    else:
        activations, row_index, layers = extract_activations(
            prompts,
            metrics,
            args.model,
            args.dtype,
            args.device,
            args.layers,
            args.pooling,
            args.max_model_len,
        )
        if args.save_activation_matrix:
            np.savez_compressed(
                activation_path,
                activations=activations,
                row_index=row_index.to_dict(orient="records"),
                layers=np.array(layers, dtype=int),
            )

    dataset = row_index.merge(
        metrics,
        on=["prompt_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    save_table(dataset.drop(columns=["candidate_answer"], errors="ignore"), out_dir, "activation_row_index")

    feature_modes = parse_feature_modes(args.feature_modes)
    probe_summary = run_layerwise_probes(
        activations,
        dataset,
        layers,
        args.seed,
        feature_modes,
        args.logistic_c,
        args.ridge_alpha,
    )
    within_prompt_probe_summary = run_within_prompt_winner_probes(
        activations,
        dataset,
        layers,
        args.seed,
        feature_modes,
        args.logistic_c,
    )
    contrast_summary = contrast_analysis(activations, dataset, layers)
    activation_paths = activation_path_long_table(activations, dataset, layers)
    pairwise_distances = pairwise_distance_table(activations, dataset, layers)
    geometry_by_layer = layer_geometry_summary(activation_paths, pairwise_distances)
    geometry_by_prompt_layer = prompt_geometry_summary(activation_paths, pairwise_distances)
    save_table(probe_summary, out_dir, "probe_summary")
    save_table(within_prompt_probe_summary, out_dir, "within_prompt_winner_probe_summary")
    save_table(contrast_summary, out_dir, "contrast_summary")
    save_table(activation_paths, out_dir, "activation_paths_long")
    save_table(pairwise_distances, out_dir, "pairwise_candidate_distances")
    save_table(geometry_by_layer, out_dir, "geometry_by_layer")
    save_table(geometry_by_prompt_layer, out_dir, "geometry_by_prompt_layer")
    save_table(allocation_pattern_summary(dataset), out_dir, "prompt_allocation_pattern_summary")

    best_rows = []
    if not probe_summary.empty:
        for target, group in probe_summary.groupby("target"):
            if "mean_auc" in group.columns and group["mean_auc"].notna().any():
                best = group.sort_values("mean_auc", ascending=False).iloc[0]
                metric = "mean_auc"
            elif "mean_balanced_accuracy" in group.columns and group["mean_balanced_accuracy"].notna().any():
                best = group.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
                metric = "mean_balanced_accuracy"
            elif "mean_r2" in group.columns and group["mean_r2"].notna().any():
                best = group.sort_values("mean_r2", ascending=False).iloc[0]
                metric = "mean_r2"
            else:
                continue
            best_rows.append(
                {
                    "target": target,
                    "best_layer_index": int(best["layer_index"]),
                    "best_feature_mode": str(best.get("feature_mode", "")),
                    "selection_metric": metric,
                    "selection_value": float(best[metric]),
                }
            )
    best_probe_layers = pd.DataFrame(best_rows)
    save_table(best_probe_layers, out_dir, "best_probe_layers")

    run_config = pd.DataFrame(
        [
            {
                "level1_output_dir": str(level1_dir),
                "model": args.model,
                "layers": args.layers,
                "feature_modes": args.feature_modes,
                "selected_layers_json": json.dumps(layers),
                "pooling": args.pooling,
                "max_model_len": args.max_model_len,
                "dtype": args.dtype,
                "device": args.device,
                "logistic_c": args.logistic_c,
                "ridge_alpha": args.ridge_alpha,
                "max_prompts": args.max_prompts,
                "n_activation_rows": int(activations.shape[0]),
                "n_layers": int(activations.shape[1]),
                "hidden_size": int(activations.shape[2]),
                "save_activation_matrix": args.save_activation_matrix,
            }
        ]
    )
    save_table(run_config, out_dir, "run_config")

    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nBest probe layers")
    print(best_probe_layers.to_string(index=False) if not best_probe_layers.empty else "No probes fit.")
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
