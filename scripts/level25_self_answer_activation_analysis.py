#!/usr/bin/env python3
"""
Level 2.5 activation analysis for self-answer ideal-point voting.

Consumes a Level 2.5 output directory. For each evaluator, it compares candidate
representations against that evaluator's own private self-answer representation
and tests whether self-answer geometry predicts signed allocation behavior.
"""

from __future__ import annotations

import argparse
import hashlib
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

from level25_ballot_prompt import (
    DEFAULT_NORMAL_CRITERION,
    VOTE_FIELDS,
    self_answer_evaluator_prompt,
    self_answer_prompt,
)
from level2_relational_activation_analysis import (
    char_span_to_token_indices,
    cosine_similarity,
    load_model_and_tokenizer,
    parse_layers,
    prompt_group_folds,
    safe_auc,
    save_table,
    standardize_train_test,
)


CANDIDATE_LABELS = ["A", "B", "C", "D"]
PIPELINE_PROTOCOL_VERSION = 2


@dataclass
class TextSpan:
    object_type: str
    candidate_id: str
    char_start: int
    char_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level25-output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="")
    parser.add_argument(
        "--ballot-variant",
        choices=["absolute_allocation", "free_range"],
        default="absolute_allocation",
        help=(
            "Which ballot builder to replay for exact-context activation "
            "extraction. Use 'free_range' when analyzing a "
            "level25_self_answer_vote_freerange_vLLM.py run so the replayed "
            "context matches the signed_fullscale ballot the votes came from."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--layers", default="16,20,24")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument(
        "--context-mode",
        choices=["exact_ballot", "reconstructed"],
        default="exact_ballot",
        help=(
            "Replay the exact chat-formatted ballot input or use the legacy "
            "short reconstructed context used by the original Qwen analysis."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="",
        choices=["", "eager", "sdpa", "flash_attention_2"],
        help="Optional Hugging Face attention implementation override.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--logistic-cs",
        "--logistic-c",
        dest="logistic_cs",
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated C grid swept with group-aware inner CV.",
    )
    parser.add_argument(
        "--ridge-alphas",
        "--ridge-alpha",
        dest="ridge_alphas",
        default="0.1,1,10,100,1000,10000,100000",
        help="Comma-separated alpha grid swept with group-aware inner CV.",
    )
    parser.add_argument("--save-activation-matrix", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str, flag: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError(f"{flag} must contain at least one value")
    return values


def inner_group_splits(
    prompt_ids: pd.Series,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Group-aware inner CV splits (by prompt) for hyperparameter selection.

    Returns None when there are too few prompt groups to split, in which case
    callers should fall back to a non-grouped selection strategy.
    """
    from sklearn.model_selection import GroupKFold

    groups = prompt_ids.astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return None
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def read_csv_required(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def read_bool_config(base: Path, key: str, default: bool) -> bool:
    path = base / "run_config.csv"
    if not path.exists():
        return default
    config = pd.read_csv(path)
    if config.empty or key not in config.columns:
        return default
    value = str(config.iloc[0][key]).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return default


def read_run_config(base: Path) -> dict[str, Any]:
    path = base / "run_config.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if not frame.empty else {}


def config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    if key not in config or pd.isna(config[key]):
        return default
    value = config[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def parse_tie_selection(selection: Any) -> set[str]:
    text = str(selection)
    if text.startswith("TIE:"):
        return {part.strip() for part in text[4:].split(",") if part.strip()}
    return {text.strip()}


def build_self_vote_dataset(base: Path, max_prompts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = read_csv_required(base, "prompts.csv")
    candidates = read_csv_required(base, "candidates.csv")
    votes = read_csv_required(base, "direct_votes.csv")
    self_answers = read_csv_required(base, "self_answers.csv")

    if max_prompts > 0:
        prompt_ids = prompts["prompt_id"].drop_duplicates().head(max_prompts).tolist()
        prompts = prompts[prompts["prompt_id"].isin(prompt_ids)].copy()
        candidates = candidates[candidates["prompt_id"].isin(prompt_ids)].copy()
        votes = votes[votes["prompt_id"].isin(prompt_ids)].copy()
        self_answers = self_answers[self_answers["prompt_id"].isin(prompt_ids)].copy()

    candidate_col = "candidate" if "candidate" in candidates.columns else "candidate_id"
    candidates = candidates.rename(columns={candidate_col: "candidate_id"})
    candidates["candidate_id"] = candidates["candidate_id"].astype(str).str.strip().str.upper()
    votes["candidate_id"] = votes["candidate_id"].astype(str).str.strip().str.upper()

    dataset = votes.merge(
        candidates[["prompt_id", "candidate_id", "candidate_answer"]],
        on=["prompt_id", "candidate_id"],
        how="left",
        validate="many_to_one",
    )
    dataset = dataset.merge(
        self_answers[
            [
                "prompt_id",
                "evaluator_id",
                "self_answer",
                "evaluator_model",
                "evaluator",
            ]
        ],
        on=["prompt_id", "evaluator_id"],
        how="inner",
        suffixes=("", "_self"),
    )
    dataset = dataset.merge(
        prompts[["prompt_id", "user_prompt"]],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )

    judge_path = base / "external_judge_results.csv"
    if judge_path.exists():
        judge = pd.read_csv(judge_path)
        if not judge.empty and "judge_winner" in judge.columns:
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
            dataset = dataset.merge(pd.DataFrame(judge_flags), on=["prompt_id", "candidate_id"], how="left")
            # Prompts absent from the judge table must not leak NaN into the
            # boolean target: bool(NaN) evaluates True.
            dataset["judge_winner"] = dataset["judge_winner"].fillna(False).astype(bool)
            dataset["judge_winner_text"] = dataset["judge_winner_text"].fillna("")
        else:
            dataset["judge_winner"] = False
            dataset["judge_winner_text"] = ""
            dataset["judge_consensus_share"] = np.nan
    else:
        dataset["judge_winner"] = False
        dataset["judge_winner_text"] = ""
        dataset["judge_consensus_share"] = np.nan

    dataset["voter_allocation"] = dataset["allocation"].astype(float)
    dataset["voter_positive_mass"] = dataset["voter_allocation"].clip(lower=0.0)
    dataset["voter_negative_mass"] = (-dataset["voter_allocation"].clip(upper=0.0))
    dataset["voter_signed_abs_share"] = dataset["voter_allocation"].abs()
    dataset["voter_help_label"] = dataset["voter_allocation"] > 0
    dataset["voter_hurt_label"] = dataset["voter_allocation"] < 0
    allocation_group = dataset.groupby(["prompt_id", "evaluator_id"])["voter_allocation"]
    allocation_mean = allocation_group.transform("mean")
    allocation_std = allocation_group.transform("std").replace(0.0, np.nan)
    dataset["voter_allocation_centered"] = dataset["voter_allocation"] - allocation_mean
    dataset["voter_allocation_z"] = (
        (dataset["voter_allocation"] - allocation_mean) / allocation_std
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dataset["voter_allocation_rank"] = dataset.groupby(
        ["prompt_id", "evaluator_id"]
    )["voter_allocation"].rank(method="average", ascending=False)
    if "best_pick_vote" in dataset.columns:
        dataset["voter_best_pick_vote"] = dataset["best_pick_vote"].astype(float) > 0
    else:
        dataset["voter_best_pick_vote"] = False
    if "borda_points" in dataset.columns:
        dataset["voter_borda_points"] = dataset["borda_points"].astype(float)
        borda_group = dataset.groupby(["prompt_id", "evaluator_id"])["voter_borda_points"]
        borda_mean = borda_group.transform("mean")
        borda_std = borda_group.transform("std").replace(0.0, np.nan)
        dataset["voter_borda_points_centered"] = dataset["voter_borda_points"] - borda_mean
        dataset["voter_borda_points_z"] = (
            (dataset["voter_borda_points"] - borda_mean) / borda_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        dataset["voter_borda_points"] = 0.0
        dataset["voter_borda_points_centered"] = 0.0
        dataset["voter_borda_points_z"] = 0.0

    # Backward-compatible aliases. In Level 2.5 these are voter-level outcomes,
    # not aggregate prompt-candidate means.
    dataset["mean_allocation"] = dataset["voter_allocation"]
    dataset["positive_mass"] = dataset["voter_positive_mass"]
    dataset["negative_mass"] = dataset["voter_negative_mass"]
    dataset["signed_abs_share"] = dataset["voter_signed_abs_share"]
    dataset["help_label"] = dataset["voter_help_label"]
    dataset["hurt_label"] = dataset["voter_hurt_label"]
    dataset["allocation_rank_within_prompt"] = dataset["voter_allocation_rank"]
    dataset["allocation_centered_within_ballot"] = dataset["voter_allocation_centered"]
    dataset["allocation_z_within_ballot"] = dataset["voter_allocation_z"]

    signed_rows = []
    for (prompt_id, evaluator_id), group in dataset.groupby(["prompt_id", "evaluator_id"]):
        max_alloc = group["voter_allocation"].max()
        for row in group.itertuples(index=False):
            signed_rows.append(
                {
                    "prompt_id": prompt_id,
                    "evaluator_id": evaluator_id,
                    "candidate_id": row.candidate_id,
                    "voter_signed_top_choice": abs(row.voter_allocation - max_alloc) < 1e-12,
                }
            )
    dataset = dataset.merge(
        pd.DataFrame(signed_rows),
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
    )
    dataset["signed_allocation_winner"] = dataset["voter_signed_top_choice"]
    dataset = dataset.sort_values(["prompt_id", "evaluator_id", "candidate_id"]).reset_index(drop=True)
    return prompts, dataset


def label_quality_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subset",
        "n_prompts",
        "n_ballots",
        "n_candidate_rows",
        "n_help",
        "n_neutral",
        "n_hurt",
        "help_rate",
        "neutral_rate",
        "hurt_rate",
        "n_one_hot_ballots",
        "one_hot_ballot_rate",
        "mean_allocation",
        "mean_absolute_allocation",
    ]
    subsets: list[tuple[str, pd.DataFrame]] = [("all_parsed", dataset)]
    if "vote_repair_count" in dataset.columns:
        repair_count = pd.to_numeric(dataset["vote_repair_count"], errors="coerce")
        subsets.append(("no_repair", dataset[repair_count.eq(0)].copy()))

    rows = []
    for subset_name, frame in subsets:
        allocation = pd.to_numeric(frame.get("voter_allocation"), errors="coerce")
        ballot_keys = ["prompt_id", "evaluator_id"]
        ballot_groups = frame.groupby(ballot_keys, sort=False) if not frame.empty else []
        one_hot = 0
        for _, group in ballot_groups:
            values = pd.to_numeric(group["voter_allocation"], errors="coerce").fillna(0.0)
            if int(values.abs().gt(1e-12).sum()) == 1 and abs(float(values.abs().sum()) - 1.0) < 1e-9:
                one_hot += 1
        n_rows = int(len(frame))
        n_ballots = (
            int(frame.drop_duplicates(ballot_keys).shape[0]) if not frame.empty else 0
        )
        n_help = int(allocation.gt(0).sum())
        n_hurt = int(allocation.lt(0).sum())
        n_neutral = int(allocation.eq(0).sum())
        rows.append(
            {
                "subset": subset_name,
                "n_prompts": int(frame["prompt_id"].nunique()) if not frame.empty else 0,
                "n_ballots": n_ballots,
                "n_candidate_rows": n_rows,
                "n_help": n_help,
                "n_neutral": n_neutral,
                "n_hurt": n_hurt,
                "help_rate": n_help / n_rows if n_rows else np.nan,
                "neutral_rate": n_neutral / n_rows if n_rows else np.nan,
                "hurt_rate": n_hurt / n_rows if n_rows else np.nan,
                "n_one_hot_ballots": one_hot,
                "one_hot_ballot_rate": one_hot / n_ballots if n_ballots else np.nan,
                "mean_allocation": float(allocation.mean()) if n_rows else np.nan,
                "mean_absolute_allocation": float(allocation.abs().mean())
                if n_rows
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def recorded_display_order(group: pd.DataFrame) -> list[str]:
    """Validated presentation order recorded at vote time, or [] if unusable."""
    candidate_ids = sorted(str(x) for x in group["candidate_id"].astype(str))
    order_text = clean_optional_text(getattr(group.iloc[0], "candidate_display_order", ""))
    wanted = [part.strip() for part in order_text.split(",") if part.strip()]
    if wanted and sorted(wanted) == candidate_ids:
        return wanted
    return []


def recorded_ballot_field_order(group: pd.DataFrame) -> list[str] | None:
    value = clean_optional_text(getattr(group.iloc[0], "ballot_field_order", ""))
    fields = [part.strip() for part in value.split(",") if part.strip()]
    if len(fields) == len(VOTE_FIELDS) and set(fields) == set(VOTE_FIELDS):
        return fields
    return None


def context_candidate_rows(group: pd.DataFrame) -> list[Any]:
    """Candidate rows in the order the evaluator saw them when voting.

    Uses candidate_display_order recorded by level25_self_answer_vote_vLLM.py.
    Falls back to sorted candidate_id for output directories produced before
    presentation order was recorded.
    """
    rows_by_id = {str(row.candidate_id): row for row in group.itertuples(index=False)}
    wanted = recorded_display_order(group)
    if wanted:
        return [rows_by_id[candidate_id] for candidate_id in wanted]
    return [rows_by_id[candidate_id] for candidate_id in sorted(rows_by_id)]


def build_reference_self_context(group: pd.DataFrame) -> tuple[str, list[TextSpan]]:
    first = group.iloc[0]
    parts = [
        "Original user prompt:",
        str(first.user_prompt),
        "",
        "Reference answer written by this evaluator:",
    ]
    text = "\n".join(parts)
    text += "\n"
    self_start = len(text)
    text += str(first.self_answer)
    self_end = len(text)
    return text, [TextSpan("self_answer", "SELF", self_start, self_end)]


def build_self_answer_context(
    group: pd.DataFrame,
    self_answer_visible_reference: bool,
) -> tuple[str, list[TextSpan]]:
    first = group.iloc[0]
    parts = [
        (
            "You are evaluating candidate answers using your own private reference answer."
            if self_answer_visible_reference
            else "You are evaluating candidate answers."
        ),
        "",
        "Original user prompt:",
        str(first.user_prompt),
        "",
    ]
    spans: list[TextSpan] = []
    if self_answer_visible_reference:
        parts.append("Private reference answer written by this evaluator:")
    text = "\n".join(parts)
    if self_answer_visible_reference:
        text += "\n"
        self_start = len(text)
        text += str(first.self_answer)
        self_end = len(text)
        spans.append(TextSpan("self_answer", "SELF", self_start, self_end))
    text += "\n\nCandidate answers:"
    for row in context_candidate_rows(group):
        # Label candidates with the display ID the evaluator saw when voting,
        # falling back to the original candidate ID for older outputs. Spans
        # always store the original candidate ID for merging with labels.
        display_label = clean_optional_text(getattr(row, "display_id", "")) or str(
            row.candidate_id
        )
        header = f"\n\nCandidate {display_label}:\n"
        text += header
        start = len(text)
        text += str(row.candidate_answer)
        end = len(text)
        spans.append(TextSpan("candidate", str(row.candidate_id), start, end))
    if self_answer_visible_reference:
        text += (
            "\n\nThink about which candidates are closest to the private reference "
            "answer and which candidates deserve positive support, negative "
            "opposition, or neutrality."
        )
    else:
        text += (
            "\n\nThink about which candidates deserve positive support, negative "
            "opposition, or neutrality."
        )
    return text, spans


def evaluator_from_saved_group(
    group: pd.DataFrame,
    run_config: dict[str, Any],
) -> dict[str, str]:
    first = group.iloc[0]
    name = str(first.get("evaluator", "normal_01"))
    mode = str(run_config.get("evaluator_mode", "normal")).strip().lower()
    if mode == "normal":
        return {
            "name": name,
            "criterion": DEFAULT_NORMAL_CRITERION,
            "mode": "normal",
        }

    from level1_direct_vote_eval import EVALUATORS

    for evaluator in EVALUATORS:
        if str(evaluator.get("name")) == name:
            return {
                "name": name,
                "criterion": str(evaluator["criterion"]),
                "mode": "role",
            }
    raise ValueError(f"Could not recover role evaluator configuration for {name!r}.")


def locate_text_span(
    rendered: str,
    text: str,
    search_from: int,
    description: str,
) -> tuple[int, int]:
    start = rendered.find(text, search_from)
    if start < 0:
        raise ValueError(f"Could not locate {description} in the rendered chat context.")
    return start, start + len(text)


def build_exact_ballot_context(
    group: pd.DataFrame,
    tokenizer: Any,
    self_answer_visible_reference: bool,
    run_config: dict[str, Any],
) -> tuple[str, list[TextSpan]]:
    first = group.iloc[0]
    shown_order = recorded_display_order(group)
    if not shown_order:
        raise ValueError(
            "Exact ballot replay requires a valid candidate_display_order for every ballot."
        )
    evaluator = evaluator_from_saved_group(group, run_config)
    ballot_field_order = recorded_ballot_field_order(group)
    if config_bool(run_config, "randomize_ballot_field_order", False) and ballot_field_order is None:
        raise ValueError(
            "Exact ballot replay requires ballot_field_order when the source run "
            "randomized ballot field order."
        )
    candidate_frame = group[["candidate_id", "candidate_answer"]].drop_duplicates(
        "candidate_id"
    )
    messages, display_to_candidate, replayed_order = self_answer_evaluator_prompt(
        user_prompt=str(first.user_prompt),
        self_answer=str(first.self_answer),
        candidates_for_prompt=candidate_frame,
        evaluator=evaluator,
        shuffle_candidates=False,
        include_reason=not config_bool(run_config, "no_vote_reasons", False),
        show_candidate_labels=config_bool(run_config, "show_candidate_labels", True),
        self_answer_visible_reference=self_answer_visible_reference,
        strict_borda=config_bool(run_config, "strict_borda", False),
        fixed_candidate_order=shown_order,
        strict_absolute_allocation=config_bool(
            run_config,
            "enforce_exact_allocation",
            False,
        ),
        vote_field_order=ballot_field_order,
    )
    if replayed_order != shown_order:
        raise ValueError(
            f"Replayed candidate order {replayed_order} did not match saved order {shown_order}."
        )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    candidate_to_display = {
        candidate_id: display_id
        for display_id, candidate_id in display_to_candidate.items()
    }
    rows_by_id = {
        str(row.candidate_id): row
        for row in group.drop_duplicates("candidate_id").itertuples(index=False)
    }
    spans: list[TextSpan] = []
    search_from = 0
    if self_answer_visible_reference:
        reference_marker = "these candidates. It is NOT a candidate and must NOT receive votes:\n"
        marker_start = rendered.find(reference_marker)
        if marker_start < 0:
            raise ValueError("Could not locate the private-reference marker in exact ballot replay.")
        self_start, self_end = locate_text_span(
            rendered,
            str(first.self_answer).strip(),
            marker_start + len(reference_marker),
            "visible self-answer",
        )
        spans.append(TextSpan("self_answer", "SELF", self_start, self_end))

    for candidate_id in shown_order:
        row = rows_by_id[candidate_id]
        display_id = candidate_to_display[candidate_id]
        header = f"Candidate {display_id}:\n"
        header_start = rendered.find(header, search_from)
        if header_start < 0:
            raise ValueError(f"Could not locate candidate header {header!r} in exact ballot replay.")
        answer_start, answer_end = locate_text_span(
            rendered,
            str(row.candidate_answer),
            header_start + len(header),
            f"answer for candidate {candidate_id}",
        )
        spans.append(TextSpan("candidate", candidate_id, answer_start, answer_end))
        search_from = answer_end
    return rendered, spans


def build_exact_self_answer_context(
    group: pd.DataFrame,
    tokenizer: Any,
    run_config: dict[str, Any],
) -> tuple[str, list[TextSpan]]:
    first = group.iloc[0]
    evaluator = evaluator_from_saved_group(group, run_config)
    answer = str(first.self_answer).strip()
    messages = self_answer_prompt(str(first.user_prompt), evaluator)
    rendered = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    answer_start = rendered.rfind(answer)
    if answer_start < 0:
        raise ValueError("Could not locate self-answer in exact generation-context replay.")
    return rendered, [
        TextSpan("self_answer", "SELF", answer_start, answer_start + len(answer))
    ]


def extract_self_answer_activations(
    dataset: pd.DataFrame,
    model_name: str,
    dtype: str,
    device: str,
    layer_spec: str,
    pooling: str,
    max_model_len: int,
    self_answer_visible_reference: bool,
    context_mode: str,
    run_config: dict[str, Any],
    attn_implementation: str = "",
    model_revision: str = "",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[int], pd.DataFrame, pd.DataFrame]:
    import torch

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        dtype,
        device,
        attn_implementation,
        model_revision,
    )
    selected_layers: list[int] | None = None
    candidate_rows = []
    candidate_activations = []
    self_activations = []
    object_path_rows = []
    diagnostic_rows = []

    for (prompt_id, evaluator_id), group in tqdm(
        dataset.groupby(["prompt_id", "evaluator_id"]),
        desc="Extracting self-answer activations",
    ):
        if context_mode == "exact_ballot":
            prompt_text, spans = build_exact_ballot_context(
                group,
                tokenizer,
                self_answer_visible_reference,
                run_config,
            )
        else:
            prompt_text, spans = build_self_answer_context(
                group,
                self_answer_visible_reference,
            )
        context_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        ballot_field_order = recorded_ballot_field_order(group)
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
            out_of_range = [layer for layer in selected_layers if layer < 0 or layer >= len(hidden_states)]
            if out_of_range:
                raise ValueError(
                    f"Requested layer(s) out of range for {model_name}: {out_of_range}. "
                    f"Model returned {len(hidden_states)} hidden-state tensors."
                )
        assert selected_layers is not None

        span_vectors: dict[tuple[str, str], np.ndarray] = {}
        extracted_candidate_ids = set()
        missing_span_ids = []
        truncated_span_ids = []
        for span in spans:
            token_indices = char_span_to_token_indices(offsets, span.char_start, span.char_end)
            if not token_indices:
                if span.object_type == "candidate":
                    missing_span_ids.append(span.candidate_id)
                else:
                    missing_span_ids.append("SELF")
                continue
            if token_indices[-1] == len(offsets) - 1 and span.char_end > offsets[token_indices[-1]][1]:
                truncated_span_ids.append(span.candidate_id if span.object_type == "candidate" else "SELF")
            pooled_indices = [token_indices[-1]] if pooling == "last" else token_indices
            layer_vectors = []
            for layer_idx in selected_layers:
                vec = hidden_states[layer_idx][0, pooled_indices, :].mean(dim=0)
                layer_vectors.append(vec.detach().float().cpu().numpy())
            vectors = np.stack(layer_vectors, axis=0)
            span_vectors[(span.object_type, span.candidate_id)] = vectors
            if span.object_type == "candidate":
                extracted_candidate_ids.add(span.candidate_id)
            for layer_pos, layer_idx in enumerate(selected_layers):
                object_path_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "evaluator_id": evaluator_id,
                        "object_type": span.object_type,
                        "candidate_id": span.candidate_id,
                        "layer_index": layer_idx,
                        "layer_position": layer_pos,
                        "activation_norm": float(np.linalg.norm(vectors[layer_pos])),
                        "n_tokens": int(len(token_indices)),
                        "span_char_start": int(span.char_start),
                        "span_char_end": int(span.char_end),
                    }
                )

        self_vec = span_vectors.get(("self_answer", "SELF"))
        self_input_token_count = None
        self_context_sha256 = ""
        if self_vec is None and not self_answer_visible_reference:
            if context_mode == "exact_ballot":
                self_text, self_spans = build_exact_self_answer_context(
                    group,
                    tokenizer,
                    run_config,
                )
            else:
                self_text, self_spans = build_reference_self_context(group)
            self_context_sha256 = hashlib.sha256(self_text.encode("utf-8")).hexdigest()
            self_encoded = tokenizer(
                self_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_model_len,
            )
            self_offsets = self_encoded.pop("offset_mapping")[0].tolist()
            self_encoded = {key: value.to(model.device) for key, value in self_encoded.items()}
            self_input_token_count = int(self_encoded["input_ids"].shape[1])
            with torch.no_grad():
                self_output = model(**self_encoded, output_hidden_states=True, use_cache=False)
            self_hidden_states = self_output.hidden_states
            for span in self_spans:
                token_indices = char_span_to_token_indices(
                    self_offsets,
                    span.char_start,
                    span.char_end,
                )
                if not token_indices:
                    missing_span_ids.append("SELF")
                    continue
                if token_indices[-1] == len(self_offsets) - 1 and span.char_end > self_offsets[token_indices[-1]][1]:
                    truncated_span_ids.append("SELF")
                pooled_indices = [token_indices[-1]] if pooling == "last" else token_indices
                layer_vectors = []
                for layer_idx in selected_layers:
                    vec = self_hidden_states[layer_idx][0, pooled_indices, :].mean(dim=0)
                    layer_vectors.append(vec.detach().float().cpu().numpy())
                self_vec = np.stack(layer_vectors, axis=0)
                for layer_pos, layer_idx in enumerate(selected_layers):
                    object_path_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "evaluator_id": evaluator_id,
                            "object_type": span.object_type,
                            "candidate_id": span.candidate_id,
                            "layer_index": layer_idx,
                            "layer_position": layer_pos,
                            "activation_norm": float(np.linalg.norm(self_vec[layer_pos])),
                            "n_tokens": int(len(token_indices)),
                            "span_char_start": int(span.char_start),
                            "span_char_end": int(span.char_end),
                        }
                    )
            del self_output, self_hidden_states
        if self_vec is not None:
            for row in group.sort_values("candidate_id").itertuples(index=False):
                cand_vec = span_vectors.get(("candidate", row.candidate_id))
                if cand_vec is None:
                    continue
                candidate_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "evaluator_id": evaluator_id,
                        "candidate_id": row.candidate_id,
                    }
                )
                candidate_activations.append(cand_vec)
                self_activations.append(self_vec)

        expected_candidate_ids = sorted(group["candidate_id"].astype(str).unique())
        extracted_candidate_ids_sorted = sorted(extracted_candidate_ids)
        diagnostic_rows.append(
            {
                "prompt_id": prompt_id,
                "evaluator_id": evaluator_id,
                "expected_candidate_rows": int(len(expected_candidate_ids)),
                "extracted_candidate_rows": int(len(extracted_candidate_ids_sorted)),
                "expected_candidate_ids": ",".join(expected_candidate_ids),
                "extracted_candidate_ids": ",".join(extracted_candidate_ids_sorted),
                "context_candidate_order": ",".join(
                    span.candidate_id for span in spans if span.object_type == "candidate"
                ),
                "context_order_source": (
                    "exact_voting_display_order"
                    if context_mode == "exact_ballot"
                    else (
                        "voting_display_order"
                        if recorded_display_order(group)
                        else "sorted_fallback"
                    )
                ),
                "context_mode": context_mode,
                "ballot_field_order": ",".join(ballot_field_order or VOTE_FIELDS),
                "context_sha256": context_sha256,
                "self_context_sha256": self_context_sha256,
                "missing_span_ids": ",".join(sorted(set(missing_span_ids))),
                "possibly_truncated_span_ids": ",".join(sorted(set(truncated_span_ids))),
                "self_answer_extracted": bool(self_vec is not None),
                "input_token_count": int(encoded["input_ids"].shape[1]),
                "self_input_token_count": self_input_token_count,
                "self_answer_visible_reference": bool(self_answer_visible_reference),
                "max_model_len": int(max_model_len),
            }
        )

        del output, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not candidate_activations:
        raise RuntimeError("No candidate/self activation rows were extracted.")
    diagnostics = pd.DataFrame(diagnostic_rows)
    incomplete = diagnostics[
        (diagnostics["expected_candidate_rows"] != diagnostics["extracted_candidate_rows"])
        | (~diagnostics["self_answer_extracted"])
        | (diagnostics["possibly_truncated_span_ids"].astype(str) != "")
    ]
    if not incomplete.empty:
        preview = incomplete[
            [
                "prompt_id",
                "evaluator_id",
                "expected_candidate_rows",
                "extracted_candidate_rows",
                "missing_span_ids",
                "possibly_truncated_span_ids",
                "input_token_count",
                "max_model_len",
            ]
        ].head(10).to_dict(orient="records")
        raise RuntimeError(
            "Incomplete or possibly truncated activation extraction. "
            "Increase --max-model-len or shorten generated text. "
            f"First affected rows: {preview}"
        )
    return (
        np.stack(candidate_activations, axis=0),
        np.stack(self_activations, axis=0),
        pd.DataFrame(candidate_rows),
        selected_layers or [],
        pd.DataFrame(object_path_rows),
        diagnostics,
    )


def build_candidate_set_geometry(
    candidate_activations: np.ndarray,
    self_activations: np.ndarray,
    dataset: pd.DataFrame,
    layers: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_rows = []
    pair_rows = []
    summary_rows = []
    for layer_pos, layer_idx in enumerate(layers):
        X = candidate_activations[:, layer_pos, :]
        S = self_activations[:, layer_pos, :]
        for (prompt_id, evaluator_id), group in dataset.groupby(["prompt_id", "evaluator_id"], sort=True):
            idx = group.index.to_numpy()
            values = X[idx]
            self_values = S[idx]
            if len(values) == 0:
                continue
            self_vec = self_values[0]
            centroid = values.mean(axis=0)
            distances_to_self = np.linalg.norm(values - self_vec, axis=1)
            cosines_to_self = np.array(
                [cosine_similarity(values[local_i], self_vec) for local_i in range(len(values))],
                dtype=float,
            )
            distances_to_centroid = np.linalg.norm(values - centroid, axis=1)
            mean_pairwise = []
            min_pairwise = []
            max_pairwise = []
            mean_pairwise_cosine = []
            min_pairwise_cosine = []
            max_pairwise_cosine = []
            mean_self_distance_advantage = []
            min_self_distance_advantage = []
            max_self_distance_advantage = []
            mean_self_cosine_advantage = []
            min_self_cosine_advantage = []
            max_self_cosine_advantage = []
            for local_i, row_i in enumerate(group.itertuples(index=False)):
                other_distances = []
                other_cosines = []
                self_distance_advantages = []
                self_cosine_advantages = []
                for local_j, row_j in enumerate(group.itertuples(index=False)):
                    if local_i == local_j:
                        continue
                    dist = float(np.linalg.norm(values[local_i] - values[local_j]))
                    cand_cos = cosine_similarity(values[local_i], values[local_j])
                    self_dist_advantage = float(distances_to_self[local_j] - distances_to_self[local_i])
                    self_cos_advantage = float(cosines_to_self[local_i] - cosines_to_self[local_j])
                    other_distances.append(dist)
                    other_cosines.append(cand_cos)
                    self_distance_advantages.append(self_dist_advantage)
                    self_cosine_advantages.append(self_cos_advantage)
                    if local_i < local_j:
                        pair_rows.append(
                            {
                                "prompt_id": prompt_id,
                                "evaluator_id": evaluator_id,
                                "layer_index": layer_idx,
                                "layer_position": layer_pos,
                                "candidate_i": row_i.candidate_id,
                                "candidate_j": row_j.candidate_id,
                                "euclidean_distance": dist,
                                "cosine_similarity": cand_cos,
                                "distance_to_self_i": float(distances_to_self[local_i]),
                                "distance_to_self_j": float(distances_to_self[local_j]),
                                "distance_to_self_difference_i_minus_j": float(
                                    distances_to_self[local_i] - distances_to_self[local_j]
                                ),
                                "distance_to_self_advantage_i_over_j": float(
                                    distances_to_self[local_j] - distances_to_self[local_i]
                                ),
                                "cosine_to_self_i": float(cosines_to_self[local_i]),
                                "cosine_to_self_j": float(cosines_to_self[local_j]),
                                "cosine_to_self_difference_i_minus_j": float(
                                    cosines_to_self[local_i] - cosines_to_self[local_j]
                                ),
                                "cosine_to_self_advantage_i_over_j": float(
                                    cosines_to_self[local_i] - cosines_to_self[local_j]
                                ),
                                "allocation_i": float(row_i.voter_allocation),
                                "allocation_j": float(row_j.voter_allocation),
                                "allocation_difference": float(row_i.voter_allocation - row_j.voter_allocation),
                                "abs_allocation_difference": float(abs(row_i.voter_allocation - row_j.voter_allocation)),
                            }
                        )
                mean_pairwise.append(float(np.mean(other_distances)))
                min_pairwise.append(float(np.min(other_distances)))
                max_pairwise.append(float(np.max(other_distances)))
                mean_pairwise_cosine.append(float(np.mean(other_cosines)))
                min_pairwise_cosine.append(float(np.min(other_cosines)))
                max_pairwise_cosine.append(float(np.max(other_cosines)))
                mean_self_distance_advantage.append(float(np.mean(self_distance_advantages)))
                min_self_distance_advantage.append(float(np.min(self_distance_advantages)))
                max_self_distance_advantage.append(float(np.max(self_distance_advantages)))
                mean_self_cosine_advantage.append(float(np.mean(self_cosine_advantages)))
                min_self_cosine_advantage.append(float(np.min(self_cosine_advantages)))
                max_self_cosine_advantage.append(float(np.max(self_cosine_advantages)))

            dist_self_rank = pd.Series(distances_to_self).rank(method="average", ascending=True).to_numpy()
            dist_centroid_rank = pd.Series(distances_to_centroid).rank(method="average", ascending=False).to_numpy()
            for local_i, row in enumerate(group.itertuples(index=False)):
                prev_self_distance = np.nan
                if layer_pos > 0:
                    prev_self_distance = float(
                        np.linalg.norm(
                            candidate_activations[idx[local_i], layer_pos - 1, :]
                            - self_activations[idx[local_i], layer_pos - 1, :]
                        )
                    )
                current_self_distance = float(distances_to_self[local_i])
                path_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "domain": row.domain,
                        "evaluator_id": evaluator_id,
                        "evaluator": row.evaluator,
                        "candidate_id": row.candidate_id,
                        "layer_index": layer_idx,
                        "layer_position": layer_pos,
                        "distance_to_self_answer": current_self_distance,
                        "cosine_to_self_answer": float(cosines_to_self[local_i]),
                        "candidate_minus_self_norm": current_self_distance,
                        "distance_to_self_rank_closest": float(dist_self_rank[local_i]),
                        "distance_to_prompt_candidate_centroid": float(distances_to_centroid[local_i]),
                        "distance_to_centroid_outlier_rank": float(dist_centroid_rank[local_i]),
                        "mean_distance_to_other_candidates": mean_pairwise[local_i],
                        "min_distance_to_other_candidate": min_pairwise[local_i],
                        "max_distance_to_other_candidate": max_pairwise[local_i],
                        "mean_cosine_to_other_candidates": mean_pairwise_cosine[local_i],
                        "min_cosine_to_other_candidate": min_pairwise_cosine[local_i],
                        "max_cosine_to_other_candidate": max_pairwise_cosine[local_i],
                        "mean_self_distance_advantage_vs_others": mean_self_distance_advantage[local_i],
                        "min_self_distance_advantage_vs_others": min_self_distance_advantage[local_i],
                        "max_self_distance_advantage_vs_others": max_self_distance_advantage[local_i],
                        "mean_self_cosine_advantage_vs_others": mean_self_cosine_advantage[local_i],
                        "min_self_cosine_advantage_vs_others": min_self_cosine_advantage[local_i],
                        "max_self_cosine_advantage_vs_others": max_self_cosine_advantage[local_i],
                        "self_distance_change_from_previous_layer": (
                            current_self_distance - prev_self_distance
                            if np.isfinite(prev_self_distance)
                            else np.nan
                        ),
                        "voter_allocation": float(row.voter_allocation),
                        "voter_allocation_centered": float(row.voter_allocation_centered),
                        "voter_allocation_z": float(row.voter_allocation_z),
                        "voter_positive_mass": float(row.voter_positive_mass),
                        "voter_negative_mass": float(row.voter_negative_mass),
                        "voter_signed_abs_share": float(row.voter_signed_abs_share),
                        "voter_allocation_rank": float(row.voter_allocation_rank),
                        "voter_help_label": bool(row.voter_help_label),
                        "voter_hurt_label": bool(row.voter_hurt_label),
                        "voter_signed_top_choice": bool(row.voter_signed_top_choice),
                        "voter_best_pick_vote": bool(getattr(row, "voter_best_pick_vote", False)),
                        "voter_borda_points": float(getattr(row, "voter_borda_points", np.nan)),
                        "voter_borda_points_centered": float(getattr(row, "voter_borda_points_centered", np.nan)),
                        "voter_borda_points_z": float(getattr(row, "voter_borda_points_z", np.nan)),
                        "mean_allocation": float(row.mean_allocation),
                        "positive_mass": float(row.positive_mass),
                        "negative_mass": float(row.negative_mass),
                        "signed_abs_share": float(row.signed_abs_share),
                        "allocation_rank_within_prompt": float(row.allocation_rank_within_prompt),
                        "help_label": bool(row.help_label),
                        "hurt_label": bool(row.hurt_label),
                        "signed_allocation_winner": bool(row.signed_allocation_winner),
                        "judge_winner": bool(row.judge_winner),
                    }
                )

            summary_rows.append(
                {
                    "prompt_id": prompt_id,
                    "evaluator_id": evaluator_id,
                    "layer_index": layer_idx,
                    "mean_distance_to_self_answer": float(np.mean(distances_to_self)),
                    "min_distance_to_self_answer": float(np.min(distances_to_self)),
                    "max_distance_to_self_answer": float(np.max(distances_to_self)),
                    "distance_to_self_spread": float(np.max(distances_to_self) - np.min(distances_to_self)),
                    "mean_distance_to_centroid": float(np.mean(distances_to_centroid)),
                    "allocation_spread": float(group["voter_allocation"].max() - group["voter_allocation"].min()),
                    "negative_mass_sum": float(group["voter_negative_mass"].sum()),
                    "positive_mass_sum": float(group["voter_positive_mass"].sum()),
                }
            )
    return pd.DataFrame(path_rows), pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


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


def layer_self_geometry_summary(candidate_to_self: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_idx, group in candidate_to_self.groupby("layer_index", sort=True):
        helped = group[group["voter_help_label"]]
        hurt = group[group["voter_hurt_label"]]
        signed_winners = group[group["voter_signed_top_choice"]]
        signed_losers = group[~group["voter_signed_top_choice"]]
        rows.append(
            {
                "layer_index": layer_idx,
                "n_candidate_rows": int(len(group)),
                "mean_distance_to_self_answer": float(group["distance_to_self_answer"].mean()),
                "mean_distance_to_self_helped": float(helped["distance_to_self_answer"].mean()),
                "mean_distance_to_self_hurt": float(hurt["distance_to_self_answer"].mean()),
                "mean_distance_to_self_signed_winners": float(signed_winners["distance_to_self_answer"].mean()),
                "mean_distance_to_self_signed_losers": float(signed_losers["distance_to_self_answer"].mean()),
                "corr_distance_to_self_voter_allocation": safe_corr(
                    group["distance_to_self_answer"].to_numpy(),
                    group["voter_allocation"].to_numpy(),
                ),
                "corr_distance_to_self_voter_positive_mass": safe_corr(
                    group["distance_to_self_answer"].to_numpy(),
                    group["voter_positive_mass"].to_numpy(),
                ),
                "corr_distance_to_self_voter_negative_mass": safe_corr(
                    group["distance_to_self_answer"].to_numpy(),
                    group["voter_negative_mass"].to_numpy(),
                ),
                "corr_distance_to_self_voter_signed_abs_share": safe_corr(
                    group["distance_to_self_answer"].to_numpy(),
                    group["voter_signed_abs_share"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_feature_matrix(
    candidate_activations: np.ndarray,
    self_activations: np.ndarray,
    geometry: pd.DataFrame,
    dataset: pd.DataFrame,
    layer_pos: int,
    layer_idx: int,
    feature_family: str,
) -> np.ndarray:
    X = candidate_activations[:, layer_pos, :]
    S = self_activations[:, layer_pos, :]
    if feature_family == "candidate_activation":
        return X
    if feature_family == "candidate_minus_self_activation":
        return X - S
    if feature_family == "candidate_plus_self_delta_activation":
        return np.concatenate([X, X - S], axis=1)

    geo = geometry[geometry["layer_index"] == layer_idx].copy()
    if geo.empty:
        raise ValueError(f"No geometry rows found for layer_index={layer_idx}")
    geo = dataset[["prompt_id", "evaluator_id", "candidate_id"]].merge(
        geo,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    candidate_set_cols = [
        "distance_to_prompt_candidate_centroid",
        "distance_to_centroid_outlier_rank",
        "mean_distance_to_other_candidates",
        "min_distance_to_other_candidate",
        "max_distance_to_other_candidate",
        "mean_cosine_to_other_candidates",
        "min_cosine_to_other_candidate",
        "max_cosine_to_other_candidate",
    ]
    self_cols = [
        "distance_to_self_answer",
        "cosine_to_self_answer",
        "candidate_minus_self_norm",
        "distance_to_self_rank_closest",
        "self_distance_change_from_previous_layer",
    ]
    pairwise_self_contrast_cols = [
        "mean_self_distance_advantage_vs_others",
        "min_self_distance_advantage_vs_others",
        "max_self_distance_advantage_vs_others",
        "mean_self_cosine_advantage_vs_others",
        "min_self_cosine_advantage_vs_others",
        "max_self_cosine_advantage_vs_others",
    ]
    interaction_pairs = {
        "self_dist_x_centroid_dist": (
            "distance_to_self_answer",
            "distance_to_prompt_candidate_centroid",
        ),
        "self_dist_x_outlier_rank": (
            "distance_to_self_answer",
            "distance_to_centroid_outlier_rank",
        ),
        "self_rank_x_centroid_dist": (
            "distance_to_self_rank_closest",
            "distance_to_prompt_candidate_centroid",
        ),
        "cosine_self_x_centroid_dist": (
            "cosine_to_self_answer",
            "distance_to_prompt_candidate_centroid",
        ),
        "self_change_x_self_dist": (
            "self_distance_change_from_previous_layer",
            "distance_to_self_answer",
        ),
        "self_distance_advantage_x_self_cosine_advantage": (
            "mean_self_distance_advantage_vs_others",
            "mean_self_cosine_advantage_vs_others",
        ),
        "self_distance_advantage_x_mean_pairwise_distance": (
            "mean_self_distance_advantage_vs_others",
            "mean_distance_to_other_candidates",
        ),
        "self_cosine_advantage_x_mean_pairwise_cosine": (
            "mean_self_cosine_advantage_vs_others",
            "mean_cosine_to_other_candidates",
        ),
    }
    for col, (left, right) in interaction_pairs.items():
        geo[col] = geo[left].astype(float) * geo[right].astype(float)
    interaction_cols = list(interaction_pairs)
    if feature_family == "candidate_set_geometry":
        cols = candidate_set_cols
    elif feature_family == "self_answer_geometry":
        cols = self_cols
    elif feature_family == "pairwise_self_contrast_geometry":
        cols = pairwise_self_contrast_cols
    elif feature_family == "combined_scalar_geometry":
        cols = candidate_set_cols + self_cols + pairwise_self_contrast_cols
    elif feature_family == "combined_scalar_geometry_with_interactions":
        cols = candidate_set_cols + self_cols + pairwise_self_contrast_cols + interaction_cols
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")
    return geo[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def run_probes(
    candidate_activations: np.ndarray,
    self_activations: np.ndarray,
    geometry: pd.DataFrame,
    dataset: pd.DataFrame,
    layers: list[int],
    seed: int,
    logistic_cs: list[float],
    ridge_alphas: list[float],
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

    targets = [
        ("voter_help_label", "classification"),
        ("voter_hurt_label", "classification"),
        ("voter_signed_top_choice", "classification"),
        ("voter_best_pick_vote", "classification"),
        ("judge_winner", "classification"),
        ("voter_allocation", "regression"),
        ("voter_allocation_centered", "regression"),
        ("voter_allocation_z", "regression"),
        ("voter_positive_mass", "regression"),
        ("voter_negative_mass", "regression"),
        ("voter_signed_abs_share", "regression"),
        ("voter_allocation_rank", "regression"),
        ("voter_borda_points", "regression"),
        ("voter_borda_points_centered", "regression"),
        ("voter_borda_points_z", "regression"),
    ]
    feature_families = [
        "candidate_activation",
        "candidate_minus_self_activation",
        "candidate_plus_self_delta_activation",
        "candidate_set_geometry",
        "self_answer_geometry",
        "pairwise_self_contrast_geometry",
        "combined_scalar_geometry",
        "combined_scalar_geometry_with_interactions",
    ]
    folds = prompt_group_folds(dataset["prompt_id"], n_folds=5, seed=seed)
    fold_inner_splits = [
        inner_group_splits(dataset["prompt_id"].iloc[train_idx])
        for train_idx, _ in folds
    ]
    rows = []
    for layer_pos, layer_idx in enumerate(layers):
        for family in feature_families:
            X = build_feature_matrix(
                candidate_activations,
                self_activations,
                geometry,
                dataset,
                layer_pos,
                layer_idx,
                family,
            )
            for target, kind in targets:
                y_raw = dataset[target]
                fold_metrics = []
                for (train_idx, test_idx), inner_cv in zip(folds, fold_inner_splits):
                    if kind == "classification":
                        y_train = y_raw.iloc[train_idx].astype(bool).astype(int).to_numpy()
                        y_test = y_raw.iloc[test_idx].astype(bool).astype(int).to_numpy()
                        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                            continue
                        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
                        chosen_c = float(np.median(logistic_cs))
                        clf = None
                        if len(logistic_cs) > 1 and inner_cv is not None:
                            try:
                                clf = LogisticRegressionCV(
                                    Cs=logistic_cs,
                                    cv=inner_cv,
                                    scoring="roc_auc",
                                    max_iter=2000,
                                    class_weight="balanced",
                                    random_state=seed,
                                )
                                clf.fit(X_train, y_train)
                                chosen_c = float(clf.C_[0])
                            except Exception:
                                clf = None
                        if clf is None:
                            clf = LogisticRegression(
                                C=chosen_c,
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
                                "chosen_c": chosen_c,
                            }
                        )
                    else:
                        y_train = y_raw.iloc[train_idx].astype(float).to_numpy()
                        y_test = y_raw.iloc[test_idx].astype(float).to_numpy()
                        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
                        # Group-aware inner CV when possible; otherwise RidgeCV
                        # falls back to efficient leave-one-out (GCV).
                        reg = RidgeCV(alphas=ridge_alphas, cv=inner_cv)
                        reg.fit(X_train, y_train)
                        preds = reg.predict(X_test)
                        fold_metrics.append(
                            {
                                "rmse": math.sqrt(mean_squared_error(y_test, preds)),
                                "r2": r2_score(y_test, preds),
                                "chosen_alpha": float(reg.alpha_),
                            }
                        )
                if fold_metrics:
                    row: dict[str, Any] = {
                        "layer_index": layer_idx,
                        "layer_position": layer_pos,
                        "feature_family": family,
                        "feature_dim": int(X.shape[1]),
                        "target": target,
                        "kind": kind,
                        "n_folds": len(fold_metrics),
                    }
                    metric_names = sorted({key for metric in fold_metrics for key in metric})
                    for metric_name in metric_names:
                        values = [metric[metric_name] for metric in fold_metrics if metric_name in metric]
                        row[f"mean_{metric_name}"] = float(np.nanmean(values))
                        row[f"std_{metric_name}"] = float(np.nanstd(values))
                    rows.append(row)
    return pd.DataFrame(rows)


def best_probe_layers(probe_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in probe_summary.groupby("target"):
        if "mean_auc" in group.columns and group["mean_auc"].notna().any():
            best = group.sort_values("mean_auc", ascending=False).iloc[0]
            metric = "mean_auc"
        elif "mean_r2" in group.columns and group["mean_r2"].notna().any():
            best = group.sort_values("mean_r2", ascending=False).iloc[0]
            metric = "mean_r2"
        else:
            continue
        rows.append(
            {
                "target": target,
                "best_layer_index": int(best["layer_index"]),
                "best_feature_family": str(best["feature_family"]),
                "selection_metric": metric,
                "selection_value": float(best[metric]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.ballot_variant == "free_range":
        # Replay the free-range (signed_fullscale) ballot so extracted
        # candidate-span activations match the context those votes came from.
        global VOTE_FIELDS, self_answer_evaluator_prompt
        from Ballot_Prompt_FreeRangeAblation import (
            VOTE_FIELDS as _FR_VOTE_FIELDS,
            self_answer_evaluator_prompt as _fr_evaluator_prompt,
        )
        VOTE_FIELDS = _FR_VOTE_FIELDS
        self_answer_evaluator_prompt = _fr_evaluator_prompt
        print("Using free-range (signed_fullscale) ballot builder for exact replay.")
    np.random.seed(args.seed)
    logistic_cs = parse_float_list(args.logistic_cs, "--logistic-cs")
    ridge_alphas = parse_float_list(args.ridge_alphas, "--ridge-alphas")
    base = Path(args.level25_output_dir)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"level25_self_answer_mech_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, dataset = build_self_vote_dataset(base, args.max_prompts)
    source_run_config = read_run_config(base)
    self_answer_visible_reference = read_bool_config(
        base,
        "self_answer_visible_reference",
        True,
    )
    n_label_rows_before_extraction = int(len(dataset))
    save_table(prompts, out_dir, "prompts_used")
    save_table(dataset, out_dir, "self_answer_vote_rows_with_text")
    save_table(dataset.drop(columns=["candidate_answer", "self_answer"], errors="ignore"), out_dir, "self_answer_vote_labels")
    save_table(label_quality_summary(dataset), out_dir, "label_quality_summary")

    (
        candidate_activations,
        self_activations,
        row_index,
        layers,
        object_paths,
        extraction_diagnostics,
    ) = extract_self_answer_activations(
        dataset,
        args.model,
        args.dtype,
        args.device,
        args.layers,
        args.pooling,
        args.max_model_len,
        self_answer_visible_reference,
        args.context_mode,
        source_run_config,
        args.attn_implementation,
        args.model_revision,
    )
    save_table(extraction_diagnostics, out_dir, "activation_extraction_diagnostics")
    dataset = row_index.merge(
        dataset,
        on=["prompt_id", "evaluator_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    save_table(object_paths, out_dir, "self_answer_activation_paths_long")

    candidate_to_self, pairwise, geometry_prompt_summary = build_candidate_set_geometry(
        candidate_activations,
        self_activations,
        dataset,
        layers,
    )
    geometry_by_layer = layer_self_geometry_summary(candidate_to_self)
    save_table(candidate_to_self, out_dir, "candidate_to_self_geometry")
    save_table(pairwise, out_dir, "candidate_pairwise_geometry")
    save_table(geometry_prompt_summary, out_dir, "self_geometry_by_prompt_evaluator_layer")
    save_table(geometry_by_layer, out_dir, "self_geometry_by_layer")

    probe_summary = run_probes(
        candidate_activations,
        self_activations,
        candidate_to_self,
        dataset,
        layers,
        args.seed,
        logistic_cs,
        ridge_alphas,
    )
    best_layers = best_probe_layers(probe_summary)
    save_table(probe_summary, out_dir, "self_geometry_probe_summary")
    save_table(best_layers, out_dir, "self_geometry_best_probe_layers")

    if args.save_activation_matrix:
        save_table(row_index, out_dir, "activation_row_index")
        np.savez_compressed(
            out_dir / "self_answer_activations.npz",
            candidate_activations=candidate_activations,
            self_activations=self_activations,
            layers=np.array(layers, dtype=int),
        )

    run_config = pd.DataFrame(
        [
            {
                "level25_output_dir": str(base),
                "model": args.model,
                "model_revision": args.model_revision,
                "layers": args.layers,
                "selected_layers_json": json.dumps(layers),
                "pooling": args.pooling,
                "context_mode": args.context_mode,
                "max_model_len": args.max_model_len,
                "dtype": args.dtype,
                "device": args.device,
                "attn_implementation": args.attn_implementation,
                "max_prompts": args.max_prompts,
                "self_answer_visible_reference": self_answer_visible_reference,
                "n_label_rows_before_extraction": n_label_rows_before_extraction,
                "n_prompt_evaluator_groups": int(
                    extraction_diagnostics[["prompt_id", "evaluator_id"]].drop_duplicates().shape[0]
                ),
                "n_activation_rows": int(candidate_activations.shape[0]),
                "n_layers": int(candidate_activations.shape[1]),
                "hidden_size": int(candidate_activations.shape[2]),
                "logistic_cs_json": json.dumps(logistic_cs),
                "ridge_alphas_json": json.dumps(ridge_alphas),
                "save_activation_matrix": args.save_activation_matrix,
            }
        ]
    )
    save_table(run_config, out_dir, "run_config")
    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nSelf-answer geometry best probe layers")
    print(best_layers.to_string(index=False) if not best_layers.empty else "No probes fit.")
    print(f"\nSaved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
