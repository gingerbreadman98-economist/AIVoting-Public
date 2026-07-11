#!/usr/bin/env python3
"""
Causal activation steering for Absolute Allocation voter-level experiments.

This script trains simple linear directions from saved Level 2.5/voter-level
activation outputs, then reruns a held-out subset of voting prompts while
adding those directions to one candidate span inside the voting context.

Typical use:

python level25_causal_activation_steering.py \
  --level25-output-dir level25_hidden_self_answer_outputs_20260709_000300 \
  --mech-output-dir level25_self_answer_mech_outputs_20260709_000748 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layer 16 \
  --max-ballots 40 \
  --strengths 0.05,0.10,0.20 \
  --interventions help_to_bottom,neg_hurt_to_bottom,pairwise_to_bottom,hurt_to_top

Control-grid example:

python level25_causal_activation_steering.py \
  --level25-output-dir level25_hidden_self_answer_outputs_20260709_000300 \
  --mech-output-dir level25_self_answer_mech_outputs_20260709_000748 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layer 16 \
  --max-ballots 80 \
  --strengths 0.05,0.10,0.20 \
  --interventions help_to_top,hurt_to_bottom,neg_help_to_bottom,neg_help_to_top,random_to_top,random_to_bottom,random2_to_top,random2_to_bottom,random3_to_top,random3_to_bottom

The script intentionally reports baseline and steered parsed ballots, rather
than only aggregate deltas, so failed JSON/repair behavior is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from level1_direct_vote_eval import (
    extract_json,
    save_table,
    translate_display_ids,
    validate_direct_votes,
)
from level25_ballot_prompt import (
    VOTE_FIELDS,
    self_answer_evaluator_prompt as canonical_self_answer_evaluator_prompt,
)

PIPELINE_PROTOCOL_VERSION = 2

DEFAULT_INTERVENTIONS = (
    "help_to_bottom,neg_hurt_to_bottom,pairwise_to_bottom,hurt_to_top"
)


@dataclass(frozen=True)
class Direction:
    name: str
    vector: np.ndarray
    train_n: int
    train_positive_rate: float | None = None
    train_target_mean: float | None = None


@dataclass(frozen=True)
class BallotJob:
    prompt_id: str
    evaluator_id: str
    domain: str
    target_candidate_id: str
    target_display_id: str
    target_policy: str
    shown_order: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train activation directions from voter-level mech outputs and "
            "test whether adding them to candidate spans changes Absolute "
            "Allocation ballots."
        )
    )
    parser.add_argument("--level25-output-dir", required=True)
    parser.add_argument("--mech-output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--pooling", choices=["mean"], default="mean")
    parser.add_argument("--interventions", default=DEFAULT_INTERVENTIONS)
    parser.add_argument(
        "--strengths",
        default="0.05,0.10,0.20",
        help=(
            "Comma-separated steering strengths. By default these are fractions "
            "of the median candidate activation norm at the selected layer."
        ),
    )
    parser.add_argument(
        "--strength-mode",
        choices=["norm_fraction", "absolute"],
        default="norm_fraction",
    )
    parser.add_argument("--max-prompts", type=int, default=25)
    parser.add_argument("--max-ballots", type=int, default=80)
    parser.add_argument(
        "--baseline-replicates",
        type=int,
        default=1,
        help=(
            "Total unsteered baseline generations per ballot. Replicate 0 is the "
            "primary baseline used for target selection and deltas; additional "
            "replicates are recorded as condition=baseline_replicate and used only "
            "to estimate per-ballot generation noise "
            "(causal_steering_baseline_noise.csv)."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--no-vote-reasons", action="store_true")
    parser.add_argument(
        "--strict-borda",
        action="store_true",
        help="Require Borda rankings to be strict singleton ranks with no ties.",
    )
    parser.add_argument(
        "--target-source",
        choices=["saved", "fresh_baseline"],
        default="fresh_baseline",
        help="Choose top/bottom targets from saved vote rows or from the paired fresh baseline.",
    )
    parser.add_argument(
        "--training-repair-policy",
        choices=["all", "no_repair"],
        default="all",
        help=(
            "Select direction-training labels from all parsed ballots (the "
            "Qwen-compatible default) or only ballots requiring no repair."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default="",
        choices=["", "eager", "sdpa", "flash_attention_2"],
        help="Optional Hugging Face attention implementation override.",
    )
    parser.add_argument(
        "--visible-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether the self-answer is visible in the voting prompt. "
            "If omitted, the value is read from run_config.csv."
        ),
    )
    parser.add_argument(
        "--install-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the Hugging Face model.",
    )
    return parser.parse_args()


def parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in parse_list(text)]


def read_csv_any(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_bool_config(run_dir: Path, key: str, default: bool) -> bool:
    path = run_dir / "run_config.csv"
    if not path.exists():
        return default
    cfg = pd.read_csv(path)
    if cfg.empty or key not in cfg.columns:
        return default
    value = cfg.iloc[0][key]
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def read_int_config(run_dir: Path, key: str, default: int) -> int:
    path = run_dir / "run_config.csv"
    if not path.exists():
        return default
    cfg = pd.read_csv(path)
    if cfg.empty or key not in cfg.columns:
        return default
    try:
        return int(cfg.iloc[0][key])
    except Exception:
        return default


def parse_ballot_field_order(value: Any) -> list[str] | None:
    fields = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(fields) == len(VOTE_FIELDS) and set(fields) == set(VOTE_FIELDS):
        return fields
    return None


def bool_series(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int).to_numpy()


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero/nonfinite direction.")
    return vec / norm


def train_logistic_direction(X: np.ndarray, y: np.ndarray, seed: int, name: str) -> Direction:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name} target has only one class in training data.")
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )
    pipe.fit(X, y)
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    raw_coef = clf.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    return Direction(
        name=name,
        vector=normalize(raw_coef),
        train_n=int(len(y)),
        train_positive_rate=float(y.mean()),
    )


def train_ridge_direction(X: np.ndarray, y: np.ndarray, name: str) -> Direction:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    pipe = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    pipe.fit(X, y)
    scaler = pipe.named_steps["standardscaler"]
    reg = pipe.named_steps["ridgecv"]
    raw_coef = reg.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    return Direction(
        name=name,
        vector=normalize(raw_coef),
        train_n=int(len(y)),
        train_target_mean=float(np.mean(y)),
    )


def train_pairwise_direction(
    labels: pd.DataFrame,
    X: np.ndarray,
    seed: int,
    max_rows: int = 20000,
) -> Direction:
    rows: list[np.ndarray] = []
    y: list[int] = []
    for _, group in labels.groupby(["prompt_id", "evaluator_id"], sort=False):
        idx = group.index.to_list()
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                i = idx[a_pos]
                j = idx[b_pos]
                ai = float(labels.loc[i, "voter_allocation"])
                aj = float(labels.loc[j, "voter_allocation"])
                if abs(ai - aj) < 1e-12:
                    continue
                if ai > aj:
                    rows.append(X[i] - X[j])
                    y.append(1)
                    rows.append(X[j] - X[i])
                    y.append(0)
                else:
                    rows.append(X[j] - X[i])
                    y.append(1)
                    rows.append(X[i] - X[j])
                    y.append(0)
    if not rows:
        raise ValueError("No non-tied pairwise allocation rows found.")
    pair_X = np.stack(rows, axis=0)
    pair_y = np.asarray(y, dtype=int)
    if len(pair_y) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pair_y), size=max_rows, replace=False)
        pair_X = pair_X[keep]
        pair_y = pair_y[keep]
    return train_logistic_direction(pair_X, pair_y, seed, "pairwise")


def load_training_data(
    mech_dir: Path,
    layer: int,
    eval_prompt_ids: set[str],
    seed: int,
    repair_policy: str = "all",
) -> tuple[pd.DataFrame, np.ndarray, list[int], dict[str, Direction], float, pd.DataFrame]:
    labels = read_csv_any(mech_dir, "self_answer_vote_labels.csv")
    row_index = read_csv_any(mech_dir, "activation_row_index.csv")
    npz_path = mech_dir / "self_answer_activations.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Rerun activation analysis with --save-activation-matrix."
        )
    npz = np.load(npz_path)
    layers = npz["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} not available; saved layers are {layers}.")
    layer_pos = layers.index(layer)
    candidate_activations = np.asarray(npz["candidate_activations"], dtype=np.float64)
    labels = row_index.merge(
        labels,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    labels["prompt_id"] = labels["prompt_id"].astype(str)
    labels["evaluator_id"] = labels["evaluator_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    X = candidate_activations[:, layer_pos, :]
    train_mask = ~labels["prompt_id"].isin(eval_prompt_ids).to_numpy()
    if repair_policy == "no_repair":
        if "vote_repair_count" not in labels.columns:
            raise ValueError(
                "--training-repair-policy no_repair requires vote_repair_count labels."
            )
        repair_count = pd.to_numeric(labels["vote_repair_count"], errors="coerce")
        train_mask &= repair_count.eq(0).to_numpy()
    elif repair_policy != "all":
        raise ValueError(f"Unknown training repair policy: {repair_policy}")
    if train_mask.sum() < 20:
        raise ValueError("Too few training rows after holding out eval prompts.")
    train_labels = labels.loc[train_mask].reset_index(drop=True)
    train_X = X[train_mask]
    help_labels = bool_series(train_labels["voter_help_label"])
    hurt_labels = bool_series(train_labels["voter_hurt_label"])
    training_summary = pd.DataFrame(
        [
            {
                "training_repair_policy": repair_policy,
                "n_prompts": int(train_labels["prompt_id"].nunique()),
                "n_ballots": int(
                    train_labels.drop_duplicates(["prompt_id", "evaluator_id"]).shape[0]
                ),
                "n_candidate_rows": int(len(train_labels)),
                "n_help": int(help_labels.sum()),
                "n_hurt": int(hurt_labels.sum()),
                "n_neutral": int(len(train_labels) - help_labels.sum() - hurt_labels.sum()),
                "help_rate": float(help_labels.mean()),
                "hurt_rate": float(hurt_labels.mean()),
            }
        ]
    )
    median_norm = float(np.median(np.linalg.norm(train_X, axis=1)))
    directions = {
        "help": train_logistic_direction(
            train_X,
            help_labels,
            seed,
            "help",
        ),
        "hurt": train_logistic_direction(
            train_X,
            hurt_labels,
            seed,
            "hurt",
        ),
        "positive_mass": train_ridge_direction(
            train_X,
            train_labels["voter_positive_mass"].astype(float).to_numpy(),
            "positive_mass",
        ),
        "negative_mass": train_ridge_direction(
            train_X,
            train_labels["voter_negative_mass"].astype(float).to_numpy(),
            "negative_mass",
        ),
        "allocation": train_ridge_direction(
            train_X,
            train_labels["voter_allocation"].astype(float).to_numpy(),
            "allocation",
        ),
        "pairwise": train_pairwise_direction(train_labels, train_X, seed),
    }
    directions["neg_hurt"] = Direction(
        name="neg_hurt",
        vector=-directions["hurt"].vector,
        train_n=directions["hurt"].train_n,
        train_positive_rate=directions["hurt"].train_positive_rate,
    )
    directions["neg_help"] = Direction(
        name="neg_help",
        vector=-directions["help"].vector,
        train_n=directions["help"].train_n,
        train_positive_rate=directions["help"].train_positive_rate,
    )
    rng = np.random.default_rng(seed)
    dim = directions["help"].vector.shape[0]
    for random_name in ["random", "random2", "random3"]:
        directions[random_name] = Direction(
            name=random_name,
            vector=normalize(rng.normal(size=dim)),
            train_n=0,
        )
    return labels, X, layers, directions, median_norm, training_summary


def shown_order_for_group(group: pd.DataFrame) -> list[str]:
    text = str(group.iloc[0].get("candidate_display_order", "")).strip()
    order = [part.strip() for part in text.split(",") if part.strip()]
    ids = set(group["candidate_id"].astype(str))
    if order and set(order) == ids:
        return order
    return sorted(ids)


def candidate_rows_in_shown_order(group: pd.DataFrame) -> list[pd.Series]:
    by_id = {str(row.candidate_id): row for row in group.itertuples(index=False)}
    rows = []
    for candidate_id in shown_order_for_group(group):
        rows.append(by_id[candidate_id])
    return rows


def build_display_maps(group: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    display_to_candidate: dict[str, str] = {}
    candidate_to_display: dict[str, str] = {}
    fallback_labels = ["1", "2", "3", "4"]
    for pos, row in enumerate(candidate_rows_in_shown_order(group)):
        candidate_id = str(row.candidate_id)
        display_id = str(getattr(row, "display_id", "") or fallback_labels[pos])
        display_to_candidate[display_id] = candidate_id
        candidate_to_display[candidate_id] = display_id
    return display_to_candidate, candidate_to_display


def build_voting_messages(
    group: pd.DataFrame,
    visible_reference: bool,
    include_reason: bool,
    strict_borda: bool = False,
    show_candidate_labels: bool = True,
    strict_absolute_allocation: bool = False,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str], list[str]]:
    first = group.iloc[0]
    fixed_order = shown_order_for_group(group)
    ballot_field_order = parse_ballot_field_order(first.get("ballot_field_order", ""))
    evaluator = {
        "name": str(first.get("evaluator", "normal")),
        "mode": "normal",
    }
    messages, display_to_candidate, shown_order = canonical_self_answer_evaluator_prompt(
        str(first.user_prompt),
        str(first.self_answer),
        group,
        evaluator,
        shuffle_candidates=False,
        include_reason=include_reason,
        show_candidate_labels=show_candidate_labels,
        self_answer_visible_reference=visible_reference,
        strict_borda=strict_borda,
        fixed_candidate_order=fixed_order,
        strict_absolute_allocation=strict_absolute_allocation,
        vote_field_order=ballot_field_order,
    )
    candidate_to_display = {
        str(candidate): str(display)
        for display, candidate in display_to_candidate.items()
    }
    return messages, display_to_candidate, candidate_to_display, shown_order_for_group(group)


def render_prompt_and_spans(
    tokenizer: Any,
    group: pd.DataFrame,
    visible_reference: bool,
    include_reason: bool,
    strict_borda: bool = False,
    show_candidate_labels: bool = True,
    strict_absolute_allocation: bool = False,
) -> tuple[str, dict[str, tuple[int, int]], list[dict[str, str]], dict[str, str], dict[str, str]]:
    messages, display_to_candidate, candidate_to_display, _ = build_voting_messages(
        group,
        visible_reference,
        include_reason,
        strict_borda,
        show_candidate_labels,
        strict_absolute_allocation,
    )
    return render_messages_and_spans(tokenizer, group, messages, candidate_to_display, display_to_candidate)


def render_messages_and_spans(
    tokenizer: Any,
    group: pd.DataFrame,
    messages: list[dict[str, str]],
    candidate_to_display: dict[str, str],
    display_to_candidate: dict[str, str],
) -> tuple[str, dict[str, tuple[int, int]], list[dict[str, str]], dict[str, str], dict[str, str]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    spans: dict[str, tuple[int, int]] = {}
    search_from = 0
    for row in candidate_rows_in_shown_order(group):
        candidate_id = str(row.candidate_id)
        display_id = candidate_to_display[candidate_id]
        answer = str(row.candidate_answer)
        header = f"Candidate {display_id}:\n"
        header_start = rendered.find(header, search_from)
        if header_start < 0:
            raise ValueError(f"Could not locate candidate header {header!r} in rendered prompt.")
        answer_start = header_start + len(header)
        answer_end = answer_start + len(answer)
        if rendered[answer_start:answer_end] != answer:
            alt_start = rendered.find(answer, header_start)
            if alt_start < 0:
                raise ValueError(f"Could not locate answer text for candidate {candidate_id}.")
            answer_start = alt_start
            answer_end = alt_start + len(answer)
        spans[candidate_id] = (answer_start, answer_end)
        search_from = answer_end
    return rendered, spans, messages, display_to_candidate, candidate_to_display


def char_span_to_token_indices(
    tokenizer: Any,
    text: str,
    span: tuple[int, int],
    max_model_len: int,
) -> tuple[dict[str, Any], list[int]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_model_len,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    start, end = span
    token_indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end
    ]
    if not token_indices:
        raise ValueError("Candidate span had no token overlap after tokenization/truncation.")
    return encoded, token_indices


def get_transformer_layers(model: Any) -> Any:
    candidates = [
        ("model.model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        (
            "model.language_model.model.layers",
            lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "model", None)
            and getattr(getattr(getattr(m, "model", None), "language_model", None).model, "layers", None),
        ),
        (
            "language_model.model.layers",
            lambda m: getattr(getattr(getattr(m, "language_model", None), "model", None), "layers", None),
        ),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            print(f"Using transformer layers at {name}; n_layers={len(layers)}")
            return layers
    print(model)
    raise AttributeError("Could not find transformer layers on model.")


def torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return "auto"


def load_hf_model(
    model_name: str,
    dtype: str,
    device: str,
    trust_remote_code: bool,
    attn_implementation: str = "",
    revision: str = "",
) -> tuple[Any, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype(dtype),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    revision_kwargs = {"revision": revision} if revision else {}
    if device == "auto":
        kwargs["device_map"] = "auto"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, **revision_kwargs)
    except Exception as causal_exc:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device != "auto":
        model.to(device)
    model.eval()
    layers = get_transformer_layers(model)
    print(f"Requested hidden_state layer will map to block index layer - 1.")
    print(f"Model has {len(layers)} transformer blocks.")
    torch.set_grad_enabled(False)
    return model, tokenizer


def generate_with_optional_steering(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    char_span: tuple[int, int] | None,
    direction: np.ndarray | None,
    layer: int,
    strength: float,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    if char_span is None or direction is None or abs(strength) < 1e-12:
        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_model_len,
        )
        token_indices: list[int] = []
    else:
        encoded, token_indices = char_span_to_token_indices(
            tokenizer, rendered_prompt, char_span, max_model_len
        )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    handle = None
    if token_indices and direction is not None:
        layers = get_transformer_layers(model)
        block_index = layer - 1
        if block_index < 0 or block_index >= len(layers):
            raise ValueError(
                f"Layer {layer} maps to block {block_index}, but model has {len(layers)} blocks."
            )
        delta = torch.tensor(direction * strength, dtype=next(model.parameters()).dtype, device=device)
        token_idx = torch.tensor(token_indices, dtype=torch.long, device=device)

        def hook_fn(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                if hidden.shape[1] >= int(token_idx.max().item()) + 1:
                    hidden[:, token_idx, :] = hidden[:, token_idx, :] + delta
                return (hidden,) + output[1:]
            hidden = output.clone()
            if hidden.shape[1] >= int(token_idx.max().item()) + 1:
                hidden[:, token_idx, :] = hidden[:, token_idx, :] + delta
            return hidden

        handle = layers[block_index].register_forward_hook(hook_fn)
    try:
        gen_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output_ids = model.generate(**gen_kwargs)
    finally:
        if handle is not None:
            handle.remove()
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def contains_forbidden_reference_vote(value: Any) -> bool:
    forbidden = {
        "SELF",
        "REFERENCE",
        "PRIVATE_REFERENCE",
        "PRIVATE REFERENCE",
        "SELF_ANSWER",
        "SELF ANSWER",
        "YOUR PRIVATE REFERENCE ANSWER",
    }
    if isinstance(value, dict):
        return any(contains_forbidden_reference_vote(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_reference_vote(item) for item in value)
    if isinstance(value, str):
        return value.strip().upper() in forbidden
    return False


def validate_exact_allocation_raw(
    output: str,
    display_to_candidate: dict[str, str],
    expected_candidates: list[str],
) -> tuple[bool, list[str], int | None]:
    errors: list[str] = []
    raw_total: int | None = None
    try:
        parsed = translate_display_ids(extract_json(output), display_to_candidate)
    except Exception as exc:
        return False, [f"invalid_json:{type(exc).__name__}"], raw_total
    if contains_forbidden_reference_vote(parsed):
        errors.append("private_reference_used_as_candidate")

    payload = parsed.get("votes", parsed)
    if not isinstance(payload, dict):
        return False, errors + ["votes_not_object"], raw_total
    allocation = payload.get("signed_allocation_cents")
    if not isinstance(allocation, list):
        return False, errors + ["signed_allocation_cents_not_list"], raw_total
    if len(allocation) != len(expected_candidates):
        errors.append(f"expected_{len(expected_candidates)}_allocation_items_got_{len(allocation)}")

    cents_by_candidate: dict[str, int] = {}
    for item in allocation:
        if not isinstance(item, dict):
            errors.append("allocation_item_not_object")
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id not in expected_candidates:
            errors.append(f"unexpected_candidate:{candidate_id}")
            continue
        if candidate_id in cents_by_candidate:
            errors.append(f"duplicate_candidate:{candidate_id}")
            continue
        cents = item.get("cents")
        if isinstance(cents, bool) or not isinstance(cents, (int, float)):
            errors.append(f"non_numeric_cents:{candidate_id}")
            continue
        if not math.isfinite(float(cents)) or int(cents) != float(cents):
            errors.append(f"non_integer_cents:{candidate_id}")
            continue
        cents = int(cents)
        if not -100 <= cents <= 100:
            errors.append(f"cents_out_of_range:{candidate_id}")
            continue
        cents_by_candidate[candidate_id] = cents

    missing = [candidate for candidate in expected_candidates if candidate not in cents_by_candidate]
    if missing:
        errors.append("missing_candidates:" + ",".join(missing))
    if len(cents_by_candidate) == len(expected_candidates):
        raw_total = sum(abs(value) for value in cents_by_candidate.values())
        if raw_total != 100:
            errors.append(f"absolute_total:{raw_total}")

    reported = payload.get("absolute_cents_total")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        errors.append("absolute_cents_total_not_numeric")
    elif not math.isfinite(float(reported)) or int(reported) != float(reported):
        errors.append("absolute_cents_total_not_integer")
    elif int(reported) != 100:
        errors.append(f"reported_absolute_total:{int(reported)}")

    return not errors, errors, raw_total


def allocation_correction_messages(
    original_messages: list[dict[str, str]],
    invalid_output: str,
    errors: list[str],
    raw_total: int | None,
) -> list[dict[str, str]]:
    total_note = (
        f"Your computed absolute total was {raw_total}."
        if raw_total is not None
        else "Your allocation could not be validated."
    )
    error_note = "; ".join(errors[:4])
    correction = (
        "Your previous ballot is invalid and cannot be accepted. "
        f"{total_note} There is one shared 100-cent budget: "
        "sum(abs(cents)) across all four candidates must equal exactly 100. "
        "Positive and negative cents are not separate budgets. "
        f"Validation issue(s): {error_note}. "
        "Return a complete replacement JSON object with votes, best_pick, "
        "borda_ranking, signed_allocation_cents, and absolute_cents_total. "
        "Use four integer cents values, one per candidate, and set "
        "absolute_cents_total to 100. Return JSON only."
    )
    return original_messages + [
        {"role": "assistant", "content": str(invalid_output)},
        {"role": "user", "content": correction},
    ]


def generate_vote_with_retries(
    model: Any,
    tokenizer: Any,
    group: pd.DataFrame,
    messages: list[dict[str, str]],
    candidate_to_display: dict[str, str],
    display_to_candidate: dict[str, str],
    target_candidate_id: str | None,
    direction: np.ndarray | None,
    layer: int,
    strength: float,
    labels_for_vote: list[str],
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enforce_exact_allocation: bool,
    allocation_max_retries: int,
) -> tuple[str, bool | None, int, list[str], int | None]:
    attempt_messages = messages
    output = ""
    valid: bool | None = None
    errors: list[str] = []
    raw_total: int | None = None
    for attempt in range(allocation_max_retries + 1):
        rendered, spans, _, _, _ = render_messages_and_spans(
            tokenizer,
            group,
            attempt_messages,
            candidate_to_display,
            display_to_candidate,
        )
        char_span = (
            spans[target_candidate_id]
            if target_candidate_id is not None and target_candidate_id in spans
            else None
        )
        output = generate_with_optional_steering(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            char_span=char_span,
            direction=direction,
            layer=layer,
            strength=strength,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if not enforce_exact_allocation:
            return output, None, attempt, [], None
        valid, errors, raw_total = validate_exact_allocation_raw(
            output,
            display_to_candidate,
            labels_for_vote,
        )
        if valid:
            return output, valid, attempt, errors, raw_total
        if attempt < allocation_max_retries:
            attempt_messages = allocation_correction_messages(
                messages,
                output,
                errors,
                raw_total,
            )
    return output, valid, allocation_max_retries, errors, raw_total


def parse_vote_output(
    output: str,
    display_to_candidate: dict[str, str],
    labels: list[str],
    include_reason: bool,
    strict_borda: bool = False,
) -> tuple[pd.DataFrame, str, int]:
    parsed = translate_display_ids(extract_json(output), display_to_candidate)
    rows = validate_direct_votes(
        parsed,
        labels,
        include_reason,
        strict_borda=strict_borda,
    )
    return pd.DataFrame(rows), "", 0


def safe_parse_vote_output(
    output: str,
    display_to_candidate: dict[str, str],
    labels: list[str],
    include_reason: bool,
    strict_borda: bool = False,
) -> tuple[pd.DataFrame, str]:
    try:
        frame, _, _ = parse_vote_output(
            output,
            display_to_candidate,
            labels,
            include_reason,
            strict_borda=strict_borda,
        )
        return frame, ""
    except Exception as exc:
        return pd.DataFrame(), repr(exc)


def parse_after_strict_gate(
    output: str,
    display_to_candidate: dict[str, str],
    labels: list[str],
    include_reason: bool,
    strict_borda: bool,
    enforce_exact_allocation: bool,
    strict_valid: bool | None,
    strict_errors: list[str],
) -> tuple[pd.DataFrame, str, bool]:
    """Parse a vote after strict validation, recovering invalid L1 totals.

    Strict-invalid but parseable ballots are kept with an explicit recovery
    indicator. Downstream analyses can filter them out for strict-only results.
    """
    if not enforce_exact_allocation or strict_valid:
        parsed, parse_error = safe_parse_vote_output(
            output,
            display_to_candidate,
            labels,
            include_reason,
            strict_borda,
        )
        return parsed, parse_error, False

    parsed, parse_error = safe_parse_vote_output(
        output,
        display_to_candidate,
        labels,
        include_reason,
        strict_borda,
    )
    if parsed.empty:
        return (
            parsed,
            "strict_allocation_invalid_after_retries:" + ";".join(strict_errors)
            + (f";parse_error:{parse_error}" if parse_error else ""),
            False,
        )
    return (
        parsed,
        "strict_allocation_invalid_recovered:" + ";".join(strict_errors),
        True,
    )


def choose_target(group: pd.DataFrame, policy: str) -> str:
    work = group.copy()
    shown_order = shown_order_for_group(work)
    order_rank = {candidate_id: pos for pos, candidate_id in enumerate(shown_order)}
    work["_shown_rank"] = work["candidate_id"].astype(str).map(order_rank).fillna(999)
    if policy == "top":
        row = work.sort_values(["allocation", "_shown_rank"], ascending=[False, True]).iloc[0]
    elif policy == "bottom":
        row = work.sort_values(["allocation", "_shown_rank"], ascending=[True, True]).iloc[0]
    elif policy == "first_shown":
        return shown_order[0]
    elif policy == "last_shown":
        return shown_order[-1]
    else:
        raise ValueError(f"Unknown target policy: {policy}")
    return str(row.candidate_id)


def choose_target_from_baseline(
    baseline_by_candidate: dict[str, dict[str, Any]],
    policy: str,
    shown_order: list[str],
) -> str:
    order_rank = {str(cid): pos for pos, cid in enumerate(shown_order)}
    rows = []
    for cid, values in baseline_by_candidate.items():
        alloc = values.get("allocation", math.nan)
        try:
            alloc = float(alloc)
        except Exception:
            continue
        if not math.isfinite(alloc):
            continue
        rows.append((str(cid), alloc, order_rank.get(str(cid), 999)))

    if not rows:
        raise ValueError("Could not choose target from fresh baseline; no finite allocations.")

    if policy == "top":
        rows.sort(key=lambda x: (-x[1], x[2]))
        return rows[0][0]
    if policy == "bottom":
        rows.sort(key=lambda x: (x[1], x[2]))
        return rows[0][0]
    if policy == "first_shown":
        return shown_order[0]
    if policy == "last_shown":
        return shown_order[-1]
    raise ValueError(f"Unknown target policy: {policy}")


def fresh_baseline_ranks(
    baseline_by_candidate: dict[str, dict[str, Any]]
) -> dict[str, int]:
    vals = []
    for cid, values in baseline_by_candidate.items():
        alloc = values.get("allocation", math.nan)
        try:
            alloc = float(alloc)
        except Exception:
            continue
        if math.isfinite(alloc):
            vals.append((str(cid), alloc))
    vals.sort(key=lambda x: x[1], reverse=True)
    return {cid: rank + 1 for rank, (cid, _) in enumerate(vals)}


def intervention_to_direction_and_policy(name: str) -> tuple[str, str]:
    table = {
        "help_to_bottom": ("help", "bottom"),
        "help_to_top": ("help", "top"),
        "neg_help_to_bottom": ("neg_help", "bottom"),
        "neg_help_to_top": ("neg_help", "top"),
        "neg_hurt_to_bottom": ("neg_hurt", "bottom"),
        "neg_hurt_to_top": ("neg_hurt", "top"),
        "hurt_to_top": ("hurt", "top"),
        "hurt_to_bottom": ("hurt", "bottom"),
        "positive_mass_to_bottom": ("positive_mass", "bottom"),
        "negative_mass_to_top": ("negative_mass", "top"),
        "pairwise_to_bottom": ("pairwise", "bottom"),
        "pairwise_to_top": ("pairwise", "top"),
        "allocation_to_bottom": ("allocation", "bottom"),
        "allocation_to_top": ("allocation", "top"),
        "random_to_bottom": ("random", "bottom"),
        "random_to_top": ("random", "top"),
        "random2_to_bottom": ("random2", "bottom"),
        "random2_to_top": ("random2", "top"),
        "random3_to_bottom": ("random3", "bottom"),
        "random3_to_top": ("random3", "top"),
    }
    if name not in table:
        raise ValueError(f"Unknown intervention {name!r}. Valid: {', '.join(sorted(table))}")
    return table[name]


def select_eval_prompt_ids(rows: pd.DataFrame, max_prompts: int, seed: int) -> set[str]:
    prompt_ids = sorted(rows["prompt_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    if max_prompts and max_prompts > 0:
        prompt_ids = prompt_ids[:max_prompts]
    return set(prompt_ids)


def select_ballot_groups(rows: pd.DataFrame, eval_prompt_ids: set[str], max_ballots: int, seed: int) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    filtered = rows[rows["prompt_id"].astype(str).isin(eval_prompt_ids)].copy()
    groups = list(filtered.groupby(["prompt_id", "evaluator_id"], sort=False))
    rng = random.Random(seed)
    rng.shuffle(groups)
    if max_ballots and max_ballots > 0:
        groups = groups[:max_ballots]
    return groups


def direction_summary_rows(directions: dict[str, Direction], layer: int, median_norm: float) -> list[dict[str, Any]]:
    rows = []
    for name, direction in sorted(directions.items()):
        rows.append(
            {
                "direction": name,
                "layer": layer,
                "direction_norm": float(np.linalg.norm(direction.vector)),
                "train_n": direction.train_n,
                "train_positive_rate": direction.train_positive_rate,
                "train_target_mean": direction.train_target_mean,
                "median_candidate_activation_norm": median_norm,
            }
        )
    return rows


def vote_rows_from_parsed(
    parsed: pd.DataFrame,
    candidate_ids: list[str],
) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        by_candidate[candidate_id] = {
            "best_pick_vote": 0,
            "borda_points": math.nan,
            "allocation": math.nan,
            "vote_repair_count": math.nan,
            "vote_repairs_json": "",
        }
    if parsed.empty:
        return by_candidate
    for row in parsed.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        by_candidate[candidate_id] = {
            "best_pick_vote": int(row.best_pick_vote),
            "borda_points": float(row.borda_points),
            "allocation": float(row.allocation),
            "vote_repair_count": int(row.vote_repair_count),
            "vote_repairs_json": str(row.vote_repairs_json),
        }
    return by_candidate


def baseline_noise_summary(vote_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-ballot generation noise between baseline replicates and the primary
    baseline. This is the null against which steered deltas can be compared:
    it is produced with zero intervention at the same temperature."""
    if vote_rows.empty or "baseline_replicate_index" not in vote_rows.columns:
        return pd.DataFrame()
    primary = vote_rows[vote_rows["condition"] == "baseline"]
    replicates = vote_rows[vote_rows["condition"] == "baseline_replicate"]
    if replicates.empty:
        return pd.DataFrame()
    key = ["ballot_id", "candidate_id"]
    merged = replicates.merge(
        primary[key + ["allocation", "best_pick_vote", "borda_points"]],
        on=key,
        suffixes=("", "_primary"),
    ).dropna(subset=["allocation", "allocation_primary"])
    if merged.empty:
        return pd.DataFrame()
    delta = merged["allocation"].astype(float) - merged["allocation_primary"].astype(float)
    sign_flip = np.sign(merged["allocation"].astype(float).round(6)) != np.sign(
        merged["allocation_primary"].astype(float).round(6)
    )
    borda_delta = (
        merged["borda_points"].astype(float) - merged["borda_points_primary"].astype(float)
    ).abs()
    best_pick_flips = []
    top_flips = []
    for _, group in merged.groupby(["ballot_id", "baseline_replicate_index"]):
        picked = set(group.loc[group["best_pick_vote"] == 1, "candidate_id"])
        picked_primary = set(group.loc[group["best_pick_vote_primary"] == 1, "candidate_id"])
        if picked and picked_primary:
            best_pick_flips.append(int(picked != picked_primary))
        top_flips.append(
            int(
                str(group.loc[group["allocation"].astype(float).idxmax(), "candidate_id"])
                != str(group.loc[group["allocation_primary"].astype(float).idxmax(), "candidate_id"])
            )
        )
    return pd.DataFrame(
        [
            {
                "n_replicate_ballots": int(
                    merged[["ballot_id", "baseline_replicate_index"]].drop_duplicates().shape[0]
                ),
                "n_candidate_pairs": int(len(merged)),
                "mean_abs_delta_allocation": float(delta.abs().mean()),
                "p90_abs_delta_allocation": float(delta.abs().quantile(0.9)),
                "allocation_sign_flip_rate": float(sign_flip.mean()),
                "mean_abs_delta_borda": float(borda_delta.mean()),
                "best_pick_flip_rate": float(np.mean(best_pick_flips))
                if best_pick_flips
                else float("nan"),
                "allocation_top_flip_rate": float(np.mean(top_flips)) if top_flips else float("nan"),
            }
        ]
    )


def summarize_results(vote_rows: pd.DataFrame) -> pd.DataFrame:
    if vote_rows.empty:
        return pd.DataFrame()
    baseline = vote_rows[vote_rows["condition"] == "baseline"][
        ["ballot_id", "candidate_id", "allocation", "best_pick_vote", "borda_points"]
    ].rename(
        columns={
            "allocation": "baseline_allocation",
            "best_pick_vote": "baseline_best_pick_vote",
            "borda_points": "baseline_borda_points",
        }
    )
    target_rows = vote_rows[
        (vote_rows["condition"] == "steered") & (vote_rows["is_target_candidate"])
    ].copy()
    steered = target_rows[target_rows["condition"] == "steered"].merge(
        baseline,
        on=["ballot_id", "candidate_id"],
        how="left",
    )
    if steered.empty:
        return pd.DataFrame()
    steered["delta_allocation"] = steered["allocation"] - steered["baseline_allocation"]
    steered["delta_best_pick_vote"] = steered["best_pick_vote"] - steered["baseline_best_pick_vote"]
    steered["delta_borda_points"] = steered["borda_points"] - steered["baseline_borda_points"]
    summary = (
        steered.groupby(["intervention", "direction", "target_policy", "strength"], dropna=False)
        .agg(
            n_ballots=("ballot_id", "nunique"),
            parse_error_rate=("parse_error", lambda s: float(s.fillna("").astype(str).ne("").mean())),
            mean_target_delta_allocation=("delta_allocation", "mean"),
            median_target_delta_allocation=("delta_allocation", "median"),
            mean_target_delta_borda=("delta_borda_points", "mean"),
            mean_target_delta_best_pick=("delta_best_pick_vote", "mean"),
            target_allocation_increase_rate=("delta_allocation", lambda s: float((s > 0).mean())),
            target_best_pick_gain_rate=("delta_best_pick_vote", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    level25_dir = Path(args.level25_output_dir)
    mech_dir = Path(args.mech_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"level25_causal_steering_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = mech_dir / "self_answer_vote_rows_with_text.csv"
    if not rows_path.exists():
        raise FileNotFoundError(
            f"{rows_path} not found. Rerun level25_self_answer_activation_analysis.py first."
        )
    vote_rows_text = pd.read_csv(rows_path)
    for col in ["prompt_id", "evaluator_id", "candidate_id"]:
        vote_rows_text[col] = vote_rows_text[col].astype(str)
    eval_prompt_ids = select_eval_prompt_ids(vote_rows_text, args.max_prompts, args.seed)
    labels, _X, layers, directions, median_norm, training_summary = load_training_data(
        mech_dir,
        args.layer,
        eval_prompt_ids,
        args.seed,
        args.training_repair_policy,
    )
    visible_reference = (
        bool(args.visible_reference)
        if args.visible_reference is not None
        else read_bool_config(level25_dir, "self_answer_visible_reference", False)
    )
    include_reason = not read_bool_config(level25_dir, "no_vote_reasons", args.no_vote_reasons)
    strict_borda = bool(args.strict_borda) or read_bool_config(level25_dir, "strict_borda", False)
    show_candidate_labels = read_bool_config(level25_dir, "show_candidate_labels", True)
    enforce_exact_allocation = read_bool_config(level25_dir, "enforce_exact_allocation", False)
    allocation_max_retries = read_int_config(level25_dir, "allocation_max_retries", 0)
    interventions = parse_list(args.interventions)
    strengths = parse_float_list(args.strengths)
    effective_strengths = [
        strength * median_norm if args.strength_mode == "norm_fraction" else strength
        for strength in strengths
    ]

    model, tokenizer = load_hf_model(
        args.model,
        args.dtype,
        args.device,
        args.install_trust_remote_code,
        args.attn_implementation,
        args.model_revision,
    )
    print(f"Requested hidden_state layer {args.layer}; steering block index {args.layer - 1}")

    ballot_groups = select_ballot_groups(
        vote_rows_text,
        eval_prompt_ids,
        args.max_ballots,
        args.seed,
    )
    all_vote_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for ballot_idx, ((prompt_id, evaluator_id), group) in enumerate(ballot_groups):
        group = group.copy()
        group["candidate_id"] = group["candidate_id"].astype(str)
        _, _, _messages, display_to_candidate, candidate_to_display = render_prompt_and_spans(
            tokenizer,
            group,
            visible_reference,
            include_reason,
            strict_borda,
            show_candidate_labels,
            enforce_exact_allocation,
        )
        shown_order = shown_order_for_group(group)
        labels_for_vote = sorted(group["candidate_id"].astype(str).unique())
        ballot_id = f"{prompt_id}::{evaluator_id}"
        baseline_output, baseline_strict_valid, baseline_retry_count, baseline_strict_errors, baseline_raw_total = generate_vote_with_retries(
            model=model,
            tokenizer=tokenizer,
            group=group,
            messages=_messages,
            candidate_to_display=candidate_to_display,
            display_to_candidate=display_to_candidate,
            target_candidate_id=None,
            direction=None,
            layer=args.layer,
            strength=0.0,
            labels_for_vote=labels_for_vote,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enforce_exact_allocation=enforce_exact_allocation,
            allocation_max_retries=allocation_max_retries,
        )
        baseline_parsed, baseline_error, baseline_recovered = parse_after_strict_gate(
            baseline_output,
            display_to_candidate,
            labels_for_vote,
            include_reason,
            strict_borda,
            enforce_exact_allocation,
            baseline_strict_valid,
            baseline_strict_errors,
        )
        baseline_by_candidate = vote_rows_from_parsed(baseline_parsed, labels_for_vote)
        fresh_ranks = fresh_baseline_ranks(baseline_by_candidate)
        fresh_top = next((cid for cid, rank in fresh_ranks.items() if rank == 1), "")
        fresh_bottom = next(
            (cid for cid, rank in fresh_ranks.items() if rank == len(fresh_ranks)),
            "",
        )
        for candidate_id in labels_for_vote:
            values = baseline_by_candidate[candidate_id]
            all_vote_rows.append(
                {
                    "ballot_index": ballot_idx,
                    "ballot_id": ballot_id,
                    "prompt_id": prompt_id,
                    "evaluator_id": evaluator_id,
                    "domain": str(group.iloc[0].get("domain", "")),
                    "condition": "baseline",
                    "intervention": "baseline",
                    "baseline_replicate_index": 0,
                    "direction": "",
                    "target_policy": "",
                    "strength": 0.0,
                    "effective_strength": 0.0,
                    "candidate_id": candidate_id,
                    "display_id": candidate_to_display.get(candidate_id, ""),
                    "is_target_candidate": False,
                    "fresh_baseline_rank": fresh_ranks.get(candidate_id, math.nan),
                    "target_fresh_baseline_rank": math.nan,
                    "fresh_baseline_top_candidate_id": fresh_top,
                    "fresh_baseline_bottom_candidate_id": fresh_bottom,
                    "parse_error": baseline_error,
                    "strict_allocation_valid": baseline_strict_valid,
                    "allocation_retry_count": baseline_retry_count,
                    "raw_absolute_total": baseline_raw_total,
                    "strict_allocation_errors_json": json.dumps(baseline_strict_errors),
                    "allocation_recovered_from_invalid": baseline_recovered,
                    **values,
                }
            )
        raw_rows.append(
            {
                "ballot_index": ballot_idx,
                "ballot_id": ballot_id,
                "condition": "baseline",
                "intervention": "baseline",
                "raw_output": baseline_output,
                "parse_error": baseline_error,
                "strict_allocation_valid": baseline_strict_valid,
                "allocation_retry_count": baseline_retry_count,
                "raw_absolute_total": baseline_raw_total,
                "strict_allocation_errors_json": json.dumps(baseline_strict_errors),
                "allocation_recovered_from_invalid": baseline_recovered,
            }
        )
        if not fresh_ranks:
            print(
                f"Skipping ballot {ballot_idx + 1}/{len(ballot_groups)} after "
                f"unparseable fresh baseline: {ballot_id}"
            )
            continue

        for replicate_index in range(1, max(1, args.baseline_replicates)):
            rep_output, rep_strict_valid, rep_retry_count, rep_strict_errors, rep_raw_total = generate_vote_with_retries(
                model=model,
                tokenizer=tokenizer,
                group=group,
                messages=_messages,
                candidate_to_display=candidate_to_display,
                display_to_candidate=display_to_candidate,
                target_candidate_id=None,
                direction=None,
                layer=args.layer,
                strength=0.0,
                labels_for_vote=labels_for_vote,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enforce_exact_allocation=enforce_exact_allocation,
                allocation_max_retries=allocation_max_retries,
            )
            rep_parsed, rep_error, rep_recovered = parse_after_strict_gate(
                rep_output,
                display_to_candidate,
                labels_for_vote,
                include_reason,
                strict_borda,
                enforce_exact_allocation,
                rep_strict_valid,
                rep_strict_errors,
            )
            rep_by_candidate = vote_rows_from_parsed(rep_parsed, labels_for_vote)
            for candidate_id in labels_for_vote:
                values = rep_by_candidate[candidate_id]
                all_vote_rows.append(
                    {
                        "ballot_index": ballot_idx,
                        "ballot_id": ballot_id,
                        "prompt_id": prompt_id,
                        "evaluator_id": evaluator_id,
                        "domain": str(group.iloc[0].get("domain", "")),
                        "condition": "baseline_replicate",
                        "intervention": "baseline_replicate",
                        "baseline_replicate_index": replicate_index,
                        "direction": "",
                        "target_policy": "",
                        "strength": 0.0,
                        "effective_strength": 0.0,
                        "candidate_id": candidate_id,
                        "display_id": candidate_to_display.get(candidate_id, ""),
                        "is_target_candidate": False,
                        "fresh_baseline_rank": fresh_ranks.get(candidate_id, math.nan),
                        "target_fresh_baseline_rank": math.nan,
                        "fresh_baseline_top_candidate_id": fresh_top,
                        "fresh_baseline_bottom_candidate_id": fresh_bottom,
                        "parse_error": rep_error,
                        "strict_allocation_valid": rep_strict_valid,
                        "allocation_retry_count": rep_retry_count,
                        "raw_absolute_total": rep_raw_total,
                        "strict_allocation_errors_json": json.dumps(rep_strict_errors),
                        "allocation_recovered_from_invalid": rep_recovered,
                        **values,
                    }
                )
            raw_rows.append(
                {
                    "ballot_index": ballot_idx,
                    "ballot_id": ballot_id,
                    "condition": "baseline_replicate",
                    "intervention": "baseline_replicate",
                    "baseline_replicate_index": replicate_index,
                    "raw_output": rep_output,
                    "parse_error": rep_error,
                    "strict_allocation_valid": rep_strict_valid,
                    "allocation_retry_count": rep_retry_count,
                    "raw_absolute_total": rep_raw_total,
                    "strict_allocation_errors_json": json.dumps(rep_strict_errors),
                    "allocation_recovered_from_invalid": rep_recovered,
                }
            )

        for intervention in interventions:
            direction_name, target_policy = intervention_to_direction_and_policy(intervention)
            if args.target_source == "fresh_baseline":
                target_candidate_id = choose_target_from_baseline(
                    baseline_by_candidate,
                    target_policy,
                    shown_order,
                )
            else:
                target_candidate_id = choose_target(group, target_policy)
            target_display_id = candidate_to_display[target_candidate_id]
            direction = directions[direction_name].vector
            for input_strength, effective_strength in zip(strengths, effective_strengths):
                steered_output, steered_strict_valid, steered_retry_count, steered_strict_errors, steered_raw_total = generate_vote_with_retries(
                    model=model,
                    tokenizer=tokenizer,
                    group=group,
                    messages=_messages,
                    candidate_to_display=candidate_to_display,
                    display_to_candidate=display_to_candidate,
                    target_candidate_id=target_candidate_id,
                    direction=direction,
                    layer=args.layer,
                    strength=effective_strength,
                    labels_for_vote=labels_for_vote,
                    max_model_len=args.max_model_len,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enforce_exact_allocation=enforce_exact_allocation,
                    allocation_max_retries=allocation_max_retries,
                )
                parsed, parse_error, steered_recovered = parse_after_strict_gate(
                    steered_output,
                    display_to_candidate,
                    labels_for_vote,
                    include_reason,
                    strict_borda,
                    enforce_exact_allocation,
                    steered_strict_valid,
                    steered_strict_errors,
                )
                by_candidate = vote_rows_from_parsed(parsed, labels_for_vote)
                for candidate_id in labels_for_vote:
                    values = by_candidate[candidate_id]
                    all_vote_rows.append(
                        {
                            "ballot_index": ballot_idx,
                            "ballot_id": ballot_id,
                            "prompt_id": prompt_id,
                            "evaluator_id": evaluator_id,
                            "domain": str(group.iloc[0].get("domain", "")),
                            "condition": "steered",
                            "intervention": intervention,
                            "direction": direction_name,
                            "target_policy": target_policy,
                            "strength": input_strength,
                            "effective_strength": effective_strength,
                            "candidate_id": candidate_id,
                            "display_id": candidate_to_display.get(candidate_id, ""),
                            "target_candidate_id": target_candidate_id,
                            "target_display_id": target_display_id,
                            "is_target_candidate": candidate_id == target_candidate_id,
                            "fresh_baseline_rank": fresh_ranks.get(candidate_id, math.nan),
                            "target_fresh_baseline_rank": fresh_ranks.get(
                                target_candidate_id,
                                math.nan,
                            ),
                            "fresh_baseline_top_candidate_id": fresh_top,
                            "fresh_baseline_bottom_candidate_id": fresh_bottom,
                            "parse_error": parse_error,
                            "strict_allocation_valid": steered_strict_valid,
                            "allocation_retry_count": steered_retry_count,
                            "raw_absolute_total": steered_raw_total,
                            "strict_allocation_errors_json": json.dumps(steered_strict_errors),
                            "allocation_recovered_from_invalid": steered_recovered,
                            **values,
                        }
                    )
                raw_rows.append(
                    {
                        "ballot_index": ballot_idx,
                        "ballot_id": ballot_id,
                        "condition": "steered",
                        "intervention": intervention,
                        "direction": direction_name,
                        "target_policy": target_policy,
                        "strength": input_strength,
                        "effective_strength": effective_strength,
                        "target_candidate_id": target_candidate_id,
                        "target_display_id": target_display_id,
                        "target_fresh_baseline_rank": fresh_ranks.get(
                            target_candidate_id,
                            math.nan,
                        ),
                        "fresh_baseline_top_candidate_id": fresh_top,
                        "fresh_baseline_bottom_candidate_id": fresh_bottom,
                        "raw_output": steered_output,
                        "parse_error": parse_error,
                        "strict_allocation_valid": steered_strict_valid,
                        "allocation_retry_count": steered_retry_count,
                        "raw_absolute_total": steered_raw_total,
                        "strict_allocation_errors_json": json.dumps(steered_strict_errors),
                        "allocation_recovered_from_invalid": steered_recovered,
                    }
                )
        print(f"Completed ballot {ballot_idx + 1}/{len(ballot_groups)}: {ballot_id}")

    vote_frame = pd.DataFrame(all_vote_rows)
    raw_frame = pd.DataFrame(raw_rows)
    summary = summarize_results(vote_frame)
    direction_frame = pd.DataFrame(direction_summary_rows(directions, args.layer, median_norm))
    config = pd.DataFrame(
        [
            {
                "level25_output_dir": str(level25_dir),
                "mech_output_dir": str(mech_dir),
                "model": args.model,
                "model_revision": args.model_revision,
                "layer": args.layer,
                "layers_available_json": json.dumps(layers),
                "visible_reference": visible_reference,
                "interventions": args.interventions,
                "strengths": args.strengths,
                "strength_mode": args.strength_mode,
                "target_source": args.target_source,
                "training_repair_policy": args.training_repair_policy,
                "effective_strengths_json": json.dumps(effective_strengths),
                "max_prompts": args.max_prompts,
                "max_ballots": args.max_ballots,
                "baseline_replicates": args.baseline_replicates,
                "n_eval_prompt_ids": len(eval_prompt_ids),
                "n_ballots_run": len(ballot_groups),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "dtype": args.dtype,
                "device": args.device,
                "attn_implementation": args.attn_implementation,
                "strict_borda": strict_borda,
                "show_candidate_labels": show_candidate_labels,
                "enforce_exact_allocation": enforce_exact_allocation,
                "allocation_max_retries": allocation_max_retries,
                "max_model_len": args.max_model_len,
            }
        ]
    )

    save_table(config, out_dir, "causal_steering_run_config")
    save_table(training_summary, out_dir, "causal_steering_training_labels")
    save_table(direction_frame, out_dir, "causal_steering_directions")
    save_table(vote_frame, out_dir, "causal_steering_vote_rows")
    save_table(raw_frame, out_dir, "causal_steering_raw_outputs")
    save_table(summary, out_dir, "causal_steering_summary")
    noise_summary = baseline_noise_summary(vote_frame)
    save_table(noise_summary, out_dir, "causal_steering_baseline_noise")
    if not noise_summary.empty:
        print("\nBaseline generation noise (replicates vs primary baseline)")
        print(noise_summary.to_string(index=False))
    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nCausal steering summary")
    if summary.empty:
        print("(empty)")
    else:
        print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
