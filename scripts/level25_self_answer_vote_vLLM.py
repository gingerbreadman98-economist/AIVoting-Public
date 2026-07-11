#!/usr/bin/env python3
"""
Level 2.5 self-answer ideal-point voting with vLLM.

Each evaluator first answers the prompt privately. The evaluator then sees that
answer as a private reference, but votes only over the original candidate
answers A/B/C/D. This creates data for testing whether signed allocation behaves
like distance from an evaluator-specific ideal answer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from level1_direct_vote_eval import (
    CANDIDATE_LABELS,
    aggregate_all,
    ballot_quality_diagnostics,
    build_direct_evaluators,
    build_summaries,
    concat_frames,
    diagnostics,
    extract_json,
    load_candidates_csv,
    load_prompts,
    save_table,
    translate_display_ids,
    validate_direct_votes,
    vote_repair_diagnostics,
    is_placeholder_reason,
)
from level1_direct_vote_eval_vLLM import (
    batched_generate_texts,
    generate_candidates_vllm,
    load_vllm_model,
    parse_model_list,
    release_vllm_model,
    run_external_judge_vllm,
    run_weak_selector_vllm,
    set_vllm_safe_seed,
)
from level1_direct_vote_eval import displayed_candidates
from level25_ballot_prompt import (
    VOTE_FIELDS,
    self_answer_evaluator_prompt as canonical_self_answer_evaluator_prompt,
    self_answer_prompt as canonical_self_answer_prompt,
)

PIPELINE_PROTOCOL_VERSION = 2


DISPLAY_LABELS = ["1", "2", "3", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--candidate-model-revision", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--evaluator-models",
        default="",
        help="Comma-separated evaluator models. Empty means use --candidate-model.",
    )
    parser.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompts-csv", default="")
    parser.add_argument("--candidates-csv", default="")
    parser.add_argument(
        "--paired-run-dir",
        default="",
        help=(
            "Reuse self_answers.csv and candidate_display_order values from a "
            "completed paired condition. Ballots without a saved display order "
            "in that run are omitted from the paired condition."
        ),
    )
    parser.add_argument(
        "--reuse-paired-inputs",
        action="store_true",
        help=(
            "Reuse self_answers.csv and candidate_display_order from --paired-run-dir "
            "without requiring the opposite visibility condition or matching ballot "
            "policy. Use this only for controlled prompt/protocol ablations."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--self-answer-max-new-tokens", type=int, default=384)
    parser.add_argument("--evaluator-max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--evaluator-mode", choices=["role", "normal"], default="normal")
    parser.add_argument("--normal-evaluator-repeats", type=int, default=10)
    parser.add_argument("--evaluator-temperature", type=float, default=0.05)
    parser.add_argument("--evaluator-top-p", type=float, default=0.95)
    parser.add_argument(
        "--self-answer-temperature",
        type=float,
        default=0.8,
        help=(
            "Sampling temperature for private self answers. Kept deliberately "
            "higher than --evaluator-temperature so repeated evaluators produce "
            "diverse ideal points instead of near-identical reference answers."
        ),
    )
    parser.add_argument("--self-answer-top-p", type=float, default=0.95)
    parser.add_argument(
        "--hide-self-answer-reference",
        dest="self_answer_visible_reference",
        action="store_false",
        help=(
            "Generate each evaluator's self-answer, but do not show it in the "
            "later voting prompt. This supports a hidden ideal-point ablation."
        ),
    )
    parser.set_defaults(self_answer_visible_reference=True)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--weak-selector-temperature", type=float, default=0.0)
    parser.add_argument("--weak-selector-top-p", type=float, default=1.0)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--weak-selector-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-vote-reasons", action="store_true")
    parser.add_argument(
        "--randomize-ballot-field-order",
        action="store_true",
        help=(
            "Randomize the requested order of best pick, Borda ranking, and signed "
            "allocation once per ballot. The order is seed-derived and recorded."
        ),
    )
    parser.add_argument("--debate-max-new-tokens", type=int, default=192)
    parser.add_argument("--shuffle-evaluator-candidates", action="store_true")
    parser.add_argument("--shuffle-judge-candidates", action="store_true")
    parser.add_argument("--shuffle-weak-selector-candidates", action="store_true")
    parser.add_argument("--show-candidate-labels", action="store_true")
    parser.add_argument("--hide-candidate-labels", dest="show_candidate_labels", action="store_false")
    parser.set_defaults(show_candidate_labels=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--skip-candidate-generation",
        action="store_true",
        help="Require --candidates-csv and skip candidate generation.",
    )
    parser.add_argument(
        "--strict-borda",
        action="store_true",
        help="Require Borda rankings to be strict singleton ranks with no ties.",
    )
    parser.add_argument(
        "--enforce-exact-allocation",
        action="store_true",
        help=(
            "Require raw signed allocations to use four integer cents values with "
            "sum(abs(cents)) exactly 100. Invalid ballots receive correction retries "
            "and are excluded if they remain invalid."
        ),
    )
    parser.add_argument(
        "--allocation-max-retries",
        type=int,
        default=2,
        help="Maximum correction generations after an invalid raw allocation ballot.",
    )
    args = parser.parse_args()
    args.debate_stage = False
    args.private_critique_stage = False
    return args


def legacy_self_answer_prompt(user_prompt: str, evaluator: dict[str, str]) -> list[dict[str, str]]:
    criterion = evaluator.get(
        "criterion",
        (
            "Answer the prompt as well as you can. Prioritize correctness, "
            "helpfulness, completeness, safety, clarity, and fit to the user's request."
        ),
    )
    return [
        {
            "role": "system",
            "content": (
                "You are creating your own private reference answer. This answer "
                "will be used only as your internal standard for later evaluation. "
                "Return only the answer text, not JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Reference-answer instruction:\n{criterion}\n\n"
                "Write the best answer you can to the original prompt. Do not "
                "mention candidates, voting, or evaluation."
            ),
        },
    ]


def legacy_self_answer_evaluator_prompt(
    user_prompt: str,
    self_answer: str,
    candidates_for_prompt: pd.DataFrame,
    evaluator: dict[str, str],
    shuffle_candidates: bool,
    include_reason: bool,
    show_candidate_labels: bool,
    self_answer_visible_reference: bool,
    strict_borda: bool = False,
    fixed_candidate_order: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    if fixed_candidate_order is None:
        shown_candidates, display_to_candidate = displayed_candidates(
            candidates_for_prompt,
            shuffle_candidates,
            show_candidate_labels,
        )
    else:
        candidate_col = (
            "candidate" if "candidate" in candidates_for_prompt.columns else "candidate_id"
        )
        available = candidates_for_prompt[candidate_col].astype(str).tolist()
        fixed_candidate_order = [str(candidate_id) for candidate_id in fixed_candidate_order]
        if len(fixed_candidate_order) != len(available) or set(fixed_candidate_order) != set(available):
            raise ValueError(
                "Paired candidate order must contain every candidate exactly once; "
                f"got {fixed_candidate_order}, expected {sorted(available)}"
            )
        by_candidate = candidates_for_prompt.set_index(
            candidates_for_prompt[candidate_col].astype(str),
            drop=False,
        )
        shown_candidates = by_candidate.loc[fixed_candidate_order].copy().reset_index(drop=True)
        if show_candidate_labels:
            shown_candidates["display_id"] = shown_candidates[candidate_col].astype(str)
        else:
            shown_candidates["display_id"] = CANDIDATE_LABELS[: len(shown_candidates)]
        display_to_candidate = {
            str(row.display_id): str(getattr(row, candidate_col))
            for row in shown_candidates.itertuples(index=False)
        }
    candidate_text = "\n\n".join(
        f"Candidate {row.display_id}:\n{row.candidate_answer}"
        for row in shown_candidates.itertuples(index=False)
    )
    # Original candidate IDs in the order the evaluator actually saw them.
    # Persisted with each vote so downstream activation extraction can rebuild
    # the same presentation order instead of a fixed sorted order.
    shown_order = [
        str(display_to_candidate[row.display_id])
        for row in shown_candidates.itertuples(index=False)
    ]
    labels = sorted(display_to_candidate.keys())
    example_ids = labels + DISPLAY_LABELS[len(labels) :]
    ex1, ex2, ex3, ex4 = example_ids[:4]
    criterion = evaluator.get(
        "criterion",
        (
            "Judge overall answer quality for the original user prompt. Balance "
            "correctness, helpfulness, completeness, safety, clarity, and fit "
            "to the user's request."
        ),
    )
    if evaluator.get("mode") == "normal":
        system_content = (
            "You are an independent evaluator casting valid ballots as a normal "
            "AI judge of answer quality. Return only JSON."
        )
        criterion_header = "Evaluation instruction"
        vote_scope = "overall answer quality"
    else:
        system_content = (
            "You are an independent evaluator casting valid ballots under your "
            "assigned criterion. Return only JSON."
        )
        criterion_header = "Evaluation criterion"
        vote_scope = "your criterion"
    comparison_anchor = (
        "your private reference answer and each other"
        if self_answer_visible_reference
        else "each other"
    )
    signed_reason_schema = (
        f'"signed_allocation_reason":"briefly explain how candidates compare to {comparison_anchor}",'
        if include_reason
        else ""
    )
    reason_instruction = (
        "Include signed_allocation_reason in the votes object, but do not "
        "include any other reason or explanation fields."
        if include_reason
        else "Do not include any reason or explanation fields."
    )
    signed_reason_focus = (
        "which candidate(s) are closest to your private reference answer, "
        "which are farthest or riskiest, and why"
        if self_answer_visible_reference
        else "which candidate(s) are strongest, which are farthest or riskiest, and why"
    )
    signed_reason_instruction = (
        "signed_allocation_reason: before giving the signed allocation vote, "
        f"briefly explain {signed_reason_focus}. Keep this to one short sentence.\n"
        if include_reason
        else ""
    )
    reference_block = (
        "Your private reference answer, written by you before seeing "
        "these candidates. It is NOT a candidate and must NOT receive "
        f"votes:\n{self_answer.strip()}\n\n"
        if self_answer_visible_reference
        else ""
    )
    reference_vote_guard = (
        "Do not vote for your private reference answer. "
        if self_answer_visible_reference
        else ""
    )
    reference_validity_note = (
        "The private reference answer has no candidate ID and is not "
        "valid to vote for. "
        if self_answer_visible_reference
        else ""
    )
    borda_instruction = (
        "borda_ranking: a strict ranking from best to worst with no ties. "
        "It must contain exactly four singleton arrays, one candidate ID per rank. "
        "Every candidate ID must appear exactly once.\n"
        if strict_borda
        else
        "borda_ranking: an array of ranked groups from best to worst. "
        "Each group is an array of candidate IDs tied at that rank. "
        "Every candidate ID must appear exactly once.\n"
    )
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"{reference_block}"
                f"Candidate answers you may vote for:\n{candidate_text}\n\n"
                f"{criterion_header}:\n{criterion}\n\n"
                "Vote only over the candidate answers above. "
                f"{reference_vote_guard}Create separate valid votes for "
                f"each aggregation system under {vote_scope}. Return a JSON "
                "object with exactly one field: votes. The votes object must "
                "contain best_pick, borda_ranking, signed_allocation_cents, "
                "and absolute_cents_total"
                f"{', plus signed_allocation_reason' if include_reason else ''}. "
                "Do not include raw scores.\n\n"
                "Use exactly this JSON shape:\n"
                f'{{"votes":{{"best_pick":"{ex1}",'
                f'"borda_ranking":[["{ex1}"],["{ex2}"],["{ex3}"],["{ex4}"]],'
                f'{signed_reason_schema}'
                f'"signed_allocation_cents":[{{"candidate_id":"{ex1}","cents":100}},'
                f'{{"candidate_id":"{ex2}","cents":0}},'
                f'{{"candidate_id":"{ex3}","cents":0}},'
                f'{{"candidate_id":"{ex4}","cents":0}}],'
                f'"absolute_cents_total":100}}}}\n'
                "Do not make votes an array. Put the allocation array inside "
                "votes.signed_allocation_cents.\n\n"
                "best_pick: one candidate ID, the single best candidate under "
                f"{vote_scope}.\n"
                f"{borda_instruction}"
                f"{signed_reason_instruction}"
                "signed_allocation_cents: this election gives you exactly 100 "
                "total influence cents. Positive cents help a candidate win. "
                "Negative cents hurt a candidate's chance to win. Neutral "
                "candidates get 0. Helping and hurting are not separate budgets: "
                "sum(abs(cents)) across all candidates must equal exactly 100. "
                f"Valid example: {ex1}=+60, {ex2}=-40, {ex3}=0, {ex4}=0. "
                f"Invalid example: {ex1}=+100 and {ex2}=-100 because the "
                "absolute total is 200. Use integer cents only. Each cents "
                "value must be between -100 and 100. Return one object for each "
                "candidate ID with candidate_id and cents. Include "
                "absolute_cents_total equal to the sum of absolute cents.\n\n"
                f"Valid candidate IDs are: {', '.join(labels)}. "
                f"{reference_validity_note}{reason_instruction}"
            ),
        },
    ]
    return messages, display_to_candidate, shown_order


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
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in {"candidate_id", "best_pick", "best_candidate"}:
                if str(item).strip().upper() in forbidden:
                    return True
            elif key_text in {"borda_ranking", "signed_allocation", "signed_allocation_cents"}:
                if contains_forbidden_reference_vote(item):
                    return True
            elif isinstance(item, (dict, list)):
                if contains_forbidden_reference_vote(item):
                    return True
        return False
    if isinstance(value, list):
        return any(contains_forbidden_reference_vote(item) for item in value)
    if isinstance(value, str):
        return value.strip().upper() in forbidden
    return False


def deterministic_ballot_field_order(
    prompt_id: str,
    evaluator_id: str,
    seed: int,
) -> list[str]:
    permutations = list(itertools.permutations(VOTE_FIELDS))
    key = f"{seed}|{prompt_id}|{evaluator_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(permutations)
    return list(permutations[index])


def parse_ballot_field_order(value: Any) -> list[str] | None:
    fields = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(fields) == len(VOTE_FIELDS) and set(fields) == set(VOTE_FIELDS):
        return fields
    return None


def validate_exact_allocation_raw(
    output: str,
    display_to_candidate: dict[str, str],
    expected_candidates: list[str],
) -> tuple[bool, list[str], int | None]:
    """Validate the model's raw allocation before parser repairs are applied."""
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


def run_self_answers_vllm(
    prompts: pd.DataFrame,
    bundle: Any,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_evaluators = build_direct_evaluators(args)
    jobs = [
        {"prompt_id": prompt.prompt_id, "domain": prompt.domain, "user_prompt": prompt.user_prompt, "evaluator": evaluator}
        for prompt in prompts.itertuples(index=False)
        for evaluator in direct_evaluators
    ]
    random.shuffle(jobs)
    messages_batch = [
        canonical_self_answer_prompt(job["user_prompt"], job["evaluator"])
        for job in jobs
    ]
    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.self_answer_max_new_tokens,
        temperature=args.self_answer_temperature,
        top_p=args.self_answer_top_p,
        batch_size=args.batch_size,
        desc="Self answers",
    )
    rows = []
    failures = []
    for job, output in zip(jobs, outputs):
        evaluator = job["evaluator"]
        evaluator_id = f"{bundle.name}::{evaluator['name']}"
        try:
            answer = output.strip()
            if len(answer) < 20 or is_placeholder_reason(answer):
                raise ValueError("empty or placeholder self-answer")
            rows.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "evaluator_model": bundle.name,
                    "evaluator": evaluator["name"],
                    "evaluator_id": evaluator_id,
                    "self_answer": answer,
                    "raw_output": output,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "evaluator_model": bundle.name,
                    "evaluator": evaluator["name"],
                    "evaluator_id": evaluator_id,
                    "error": repr(exc),
                    "raw_output": output,
                }
            )
    columns = [
        "prompt_id",
        "domain",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "self_answer",
        "raw_output",
    ]
    failure_columns = [
        "prompt_id",
        "domain",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "error",
        "raw_output",
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(failures, columns=failure_columns)


def load_paired_run_inputs(
    paired_run_dir: Path,
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    evaluator_model: str,
    args: argparse.Namespace,
    reuse_inputs: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
    pd.DataFrame,
]:
    required = [
        paired_run_dir / "run_config.csv",
        paired_run_dir / "prompts.csv",
        paired_run_dir / "candidates.csv",
        paired_run_dir / "self_answers.csv",
        paired_run_dir / "direct_votes.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Paired run is missing required files: " + ", ".join(missing))

    paired_config = pd.read_csv(paired_run_dir / "run_config.csv")
    if paired_config.empty:
        raise ValueError("Paired run_config.csv is empty.")
    cfg = paired_config.iloc[0]

    if "evaluator_models_json" in paired_config.columns:
        paired_models = [str(model) for model in json.loads(str(cfg["evaluator_models_json"]))]
        if str(evaluator_model) not in paired_models:
            raise ValueError(
                f"Evaluator model {evaluator_model!r} is not present in paired run models "
                f"{paired_models}."
            )

    def config_bool(key: str, default: bool) -> bool:
        if key not in paired_config.columns:
            return default
        return str(cfg[key]).strip().lower() in {"1", "true", "yes", "y"}

    checks = {
        "batch_size": args.batch_size,
        "normal_evaluator_repeats": args.normal_evaluator_repeats,
        "max_model_len": args.max_model_len,
        "self_answer_max_new_tokens": args.self_answer_max_new_tokens,
        "self_answer_temperature": args.self_answer_temperature,
        "self_answer_top_p": args.self_answer_top_p,
        "evaluator_max_new_tokens": args.evaluator_max_new_tokens,
        "evaluator_temperature": args.evaluator_temperature,
        "allocation_max_retries": args.allocation_max_retries,
    }
    # A paired visibility run intentionally uses a new generation seed. The
    # archived self-answers and display/field orders, not the sampling seed,
    # define the paired experimental inputs.
    mismatches = []
    for key, current_value in checks.items():
        if key in paired_config.columns and str(cfg[key]) != str(current_value):
            try:
                matches = float(cfg[key]) == float(current_value)
            except Exception:
                matches = False
            if not matches:
                mismatches.append(f"{key}: paired={cfg[key]!r}, current={current_value!r}")
    bool_checks = {
        "shuffle_evaluator_candidates": args.shuffle_evaluator_candidates,
        "show_candidate_labels": args.show_candidate_labels,
        "strict_borda": args.strict_borda,
        "randomize_ballot_field_order": args.randomize_ballot_field_order,
        "enforce_exact_allocation": args.enforce_exact_allocation,
        "enforce_eager": args.enforce_eager,
    }
    for key, current_value in bool_checks.items():
        if key in paired_config.columns and config_bool(key, current_value) != bool(current_value):
            mismatches.append(
                f"{key}: paired={config_bool(key, current_value)!r}, current={current_value!r}"
            )
    if mismatches and not reuse_inputs:
        raise ValueError("Paired run configuration mismatch: " + "; ".join(mismatches))

    paired_visible = config_bool("self_answer_visible_reference", False)
    if not reuse_inputs and paired_visible == bool(args.self_answer_visible_reference):
        raise ValueError(
            "Paired run must use the opposite self-answer visibility condition; "
            f"both are visible={paired_visible}."
        )

    current_prompt_ids = set(prompts["prompt_id"].astype(str))
    paired_prompts = pd.read_csv(paired_run_dir / "prompts.csv")
    paired_prompts = paired_prompts[
        paired_prompts["prompt_id"].astype(str).isin(current_prompt_ids)
    ].copy()
    prompt_columns = ["prompt_id", "domain", "user_prompt"]
    current_prompt_view = prompts[prompt_columns].sort_values("prompt_id").reset_index(drop=True)
    paired_prompt_view = paired_prompts[prompt_columns].sort_values("prompt_id").reset_index(drop=True)
    if not current_prompt_view.equals(paired_prompt_view):
        raise ValueError("Current prompts do not exactly match the paired run prompts.")

    paired_candidates = pd.read_csv(paired_run_dir / "candidates.csv")
    paired_candidates = paired_candidates[
        paired_candidates["prompt_id"].astype(str).isin(current_prompt_ids)
    ].copy()
    candidate_col = "candidate" if "candidate" in candidates.columns else "candidate_id"
    paired_candidate_col = (
        "candidate" if "candidate" in paired_candidates.columns else "candidate_id"
    )
    candidate_columns = ["prompt_id", candidate_col, "candidate_answer"]
    paired_candidate_columns = ["prompt_id", paired_candidate_col, "candidate_answer"]
    current_candidate_view = candidates[candidate_columns].rename(
        columns={candidate_col: "candidate_id"}
    ).sort_values(["prompt_id", "candidate_id"]).reset_index(drop=True)
    paired_candidate_view = paired_candidates[paired_candidate_columns].rename(
        columns={paired_candidate_col: "candidate_id"}
    ).sort_values(["prompt_id", "candidate_id"]).reset_index(drop=True)
    if not current_candidate_view.equals(paired_candidate_view):
        raise ValueError("Current candidate answers do not exactly match the paired run.")

    self_answers = pd.read_csv(paired_run_dir / "self_answers.csv")
    self_answers = self_answers[
        self_answers["evaluator_model"].astype(str) == str(evaluator_model)
    ].copy()
    self_answers = self_answers[self_answers["prompt_id"].isin(prompts["prompt_id"])].copy()
    if self_answers.empty:
        raise ValueError(f"No paired self-answers found for evaluator model {evaluator_model!r}.")
    key_columns = ["prompt_id", "evaluator_id"]
    if self_answers.duplicated(key_columns).any():
        raise ValueError("Paired self_answers.csv contains duplicate ballot keys.")

    design_path = paired_run_dir / "allocation_validation_diagnostics.csv"
    paired_votes = pd.read_csv(design_path if design_path.exists() else paired_run_dir / "direct_votes.csv")
    if "candidate_display_order" not in paired_votes.columns:
        paired_votes = pd.read_csv(paired_run_dir / "direct_votes.csv")
    paired_votes = paired_votes[
        paired_votes["evaluator_model"].astype(str) == str(evaluator_model)
    ].copy()
    paired_votes = paired_votes[paired_votes["prompt_id"].isin(prompts["prompt_id"])].copy()
    order_counts = paired_votes.groupby(key_columns)["candidate_display_order"].nunique()
    if order_counts.gt(1).any():
        raise ValueError("Paired direct_votes.csv has inconsistent display orders within a ballot.")
    order_columns = key_columns + ["candidate_display_order"]
    if "ballot_field_order" in paired_votes.columns:
        field_counts = paired_votes.groupby(key_columns)["ballot_field_order"].nunique()
        if field_counts.gt(1).any():
            raise ValueError("Paired direct_votes.csv has inconsistent ballot field orders.")
        order_columns.append("ballot_field_order")
    order_rows = paired_votes.drop_duplicates(key_columns)[order_columns]
    paired_orders = {
        (str(row.prompt_id), str(row.evaluator_id)): [
            part.strip()
            for part in str(row.candidate_display_order).split(",")
            if part.strip()
        ]
        for row in order_rows.itertuples(index=False)
    }
    paired_field_orders = {
        (str(row.prompt_id), str(row.evaluator_id)): parsed
        for row in order_rows.itertuples(index=False)
        if (parsed := parse_ballot_field_order(getattr(row, "ballot_field_order", "")))
        is not None
    }
    paired_randomized_fields = config_bool("randomize_ballot_field_order", False)
    if paired_randomized_fields and not reuse_inputs:
        missing_field_orders = set(paired_orders) - set(paired_field_orders)
        if missing_field_orders:
            raise ValueError(
                "Paired run enabled randomized ballot field order but some ballots lack "
                "a saved ballot_field_order."
            )

    expected_keys = {
        (str(row.prompt_id), str(row.evaluator_id))
        for row in self_answers.itertuples(index=False)
    }
    paired_keys = set(paired_orders)
    eligible_keys = expected_keys & paired_keys
    diagnostics = pd.DataFrame(
        [
            {
                "paired_run_dir": str(paired_run_dir),
                "paired_visibility": paired_visible,
                "current_visibility": bool(args.self_answer_visible_reference),
                "reuse_paired_inputs": reuse_inputs,
                "ignored_paired_config_mismatches": "; ".join(mismatches) if reuse_inputs else "",
                "self_answer_rows_loaded": int(len(self_answers)),
                "paired_orders_loaded": int(len(paired_orders)),
                "paired_field_orders_loaded": int(len(paired_field_orders)),
                "eligible_paired_ballots": int(len(eligible_keys)),
                "omitted_without_paired_order": int(len(expected_keys - paired_keys)),
            }
        ]
    )
    failure_columns = [
        "prompt_id",
        "domain",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "error",
        "raw_output",
    ]
    return (
        self_answers,
        pd.DataFrame(columns=failure_columns),
        paired_orders,
        paired_field_orders,
        diagnostics,
    )


def run_self_answer_direct_votes_vllm(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    self_answers: pd.DataFrame,
    bundle: Any,
    args: argparse.Namespace,
    paired_orders: dict[tuple[str, str], list[str]] | None = None,
    paired_field_orders: dict[tuple[str, str], list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompt_lookup = prompts.set_index("prompt_id")["user_prompt"].to_dict()
    domain_lookup = candidates.drop_duplicates("prompt_id").set_index("prompt_id")["domain"].to_dict()
    labels = sorted(candidates["candidate"].unique())
    evaluator_lookup = {
        f"{bundle.name}::{evaluator['name']}": evaluator
        for evaluator in build_direct_evaluators(args)
    }

    jobs = []
    messages_batch = []
    for row in self_answers.itertuples(index=False):
        pair_key = (str(row.prompt_id), str(row.evaluator_id))
        if paired_orders is not None and pair_key not in paired_orders:
            continue
        evaluator = evaluator_lookup.get(row.evaluator_id)
        if evaluator is None:
            continue
        group = candidates[candidates["prompt_id"] == row.prompt_id]
        ballot_field_order = (
            paired_field_orders.get(pair_key)
            if paired_field_orders is not None
            else None
        )
        if ballot_field_order is None and args.randomize_ballot_field_order:
            ballot_field_order = deterministic_ballot_field_order(
                str(row.prompt_id),
                str(row.evaluator_id),
                args.seed,
            )
        messages, display_to_candidate, shown_order = canonical_self_answer_evaluator_prompt(
            prompt_lookup[row.prompt_id],
            row.self_answer,
            group,
            evaluator,
            args.shuffle_evaluator_candidates,
            not args.no_vote_reasons,
            args.show_candidate_labels,
            args.self_answer_visible_reference,
            args.strict_borda,
            paired_orders[pair_key] if paired_orders is not None else None,
            strict_absolute_allocation=args.enforce_exact_allocation,
            vote_field_order=ballot_field_order,
        )
        jobs.append(
            {
                "prompt_id": row.prompt_id,
                "domain": domain_lookup[row.prompt_id],
                "evaluator": evaluator,
                "evaluator_id": row.evaluator_id,
                "display_to_candidate": display_to_candidate,
                "candidate_to_display": {
                    str(candidate): str(display)
                    for display, candidate in display_to_candidate.items()
                },
                "candidate_display_order": ",".join(shown_order),
                "ballot_field_order": ",".join(ballot_field_order or VOTE_FIELDS),
                "messages": messages,
            }
        )
        messages_batch.append(messages)

    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.evaluator_max_new_tokens,
        temperature=args.evaluator_temperature,
        top_p=args.evaluator_top_p,
        batch_size=args.batch_size,
        desc="Self-answer direct vote evaluations",
    )

    states = []
    for job, output in zip(jobs, outputs):
        if args.enforce_exact_allocation:
            first_valid, errors, raw_total = validate_exact_allocation_raw(
                output,
                job["display_to_candidate"],
                labels,
            )
        else:
            first_valid, errors, raw_total = None, [], None
        states.append(
            {
                "job": job,
                "first_output": output,
                "final_output": output,
                "first_valid": first_valid,
                "final_valid": first_valid,
                "first_raw_total": raw_total,
                "final_raw_total": raw_total,
                "errors": errors,
                "retry_count": 0,
                "attempt_history": [
                    {
                        "attempt": 0,
                        "raw_output": output,
                        "valid": first_valid,
                        "raw_absolute_total": raw_total,
                        "validation_errors": errors,
                    }
                ],
            }
        )

    if args.enforce_exact_allocation:
        pending = [state for state in states if not state["final_valid"]]
        for _attempt in range(1, args.allocation_max_retries + 1):
            if not pending:
                break
            correction_batch = [
                allocation_correction_messages(
                    state["job"]["messages"],
                    state["final_output"],
                    state["errors"],
                    state["final_raw_total"],
                )
                for state in pending
            ]
            retry_outputs = batched_generate_texts(
                bundle,
                correction_batch,
                max_new_tokens=args.evaluator_max_new_tokens,
                temperature=args.evaluator_temperature,
                top_p=args.evaluator_top_p,
                batch_size=args.batch_size,
                desc="Correcting invalid absolute-allocation ballots",
            )
            next_pending = []
            for state, output in zip(pending, retry_outputs):
                valid, errors, raw_total = validate_exact_allocation_raw(
                    output,
                    state["job"]["display_to_candidate"],
                    labels,
                )
                state["final_output"] = output
                state["final_valid"] = valid
                state["final_raw_total"] = raw_total
                state["errors"] = errors
                state["retry_count"] += 1
                state["attempt_history"].append(
                    {
                        "attempt": state["retry_count"],
                        "raw_output": output,
                        "valid": valid,
                        "raw_absolute_total": raw_total,
                        "validation_errors": errors,
                    }
                )
                if not valid:
                    next_pending.append(state)
            pending = next_pending

    rows = []
    failures = []
    allocation_diagnostics = []
    for state in states:
        job = state["job"]
        output = state["final_output"]
        evaluator = job["evaluator"]
        allocation_diagnostics.append(
            {
                "prompt_id": job["prompt_id"],
                "domain": job["domain"],
                "evaluator_model": bundle.name,
                "evaluator": evaluator["name"],
                "evaluator_id": job["evaluator_id"],
                "candidate_display_order": job["candidate_display_order"],
                "ballot_field_order": job["ballot_field_order"],
                "display_to_candidate_json": json.dumps(job["display_to_candidate"], sort_keys=True),
                "enforce_exact_allocation": args.enforce_exact_allocation,
                "first_pass_valid": state["first_valid"],
                "final_valid": state["final_valid"],
                "retry_count": state["retry_count"],
                "attempt_count": state["retry_count"] + 1,
                "first_raw_absolute_total": state["first_raw_total"],
                "final_raw_absolute_total": state["final_raw_total"],
                "final_validation_errors_json": json.dumps(state["errors"]),
                "allocation_attempt_history_json": json.dumps(state["attempt_history"]),
                "first_raw_output": state["first_output"],
                "final_raw_output": output,
            }
        )
        if args.enforce_exact_allocation and not state["final_valid"]:
            failures.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "candidate_id": ",".join(labels),
                    "evaluator_model": bundle.name,
                    "evaluator": evaluator["name"],
                    "evaluator_id": job["evaluator_id"],
                    "error": "strict_allocation_invalid_after_retries:"
                    + ";".join(state["errors"]),
                    "raw_output": output,
                }
            )
            continue
        try:
            parsed = translate_display_ids(extract_json(output), job["display_to_candidate"])
            if contains_forbidden_reference_vote(parsed):
                raise ValueError("vote attempted to use private reference answer as a candidate")
            for item in validate_direct_votes(
                parsed,
                labels,
                not args.no_vote_reasons,
                strict_borda=args.strict_borda,
            ):
                rows.append(
                    {
                        "prompt_id": job["prompt_id"],
                        "domain": job["domain"],
                        "candidate_id": item["candidate_id"],
                        "display_id": job["candidate_to_display"].get(
                            str(item["candidate_id"]), ""
                        ),
                        "candidate_display_order": job["candidate_display_order"],
                        "ballot_field_order": job["ballot_field_order"],
                        "evaluator_model": bundle.name,
                        "evaluator": evaluator["name"],
                        "evaluator_id": job["evaluator_id"],
                        "best_pick_vote": item["best_pick_vote"],
                        "borda_points": item["borda_points"],
                        "allocation": item["allocation"],
                        "reason": item["reason"],
                        "vote_repairs_json": item["vote_repairs_json"],
                        "vote_repair_count": item["vote_repair_count"],
                        "allocation_first_pass_valid": state["first_valid"],
                        "allocation_retry_count": state["retry_count"],
                        "raw_output": output,
                        "placeholder_reason": False,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "candidate_id": ",".join(labels),
                    "evaluator_model": bundle.name,
                    "evaluator": evaluator["name"],
                    "evaluator_id": job["evaluator_id"],
                    "error": repr(exc),
                    "raw_output": output,
                }
            )
    columns = [
        "prompt_id",
        "domain",
        "candidate_id",
        "display_id",
        "candidate_display_order",
        "ballot_field_order",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "best_pick_vote",
        "borda_points",
        "allocation",
        "reason",
        "vote_repairs_json",
        "vote_repair_count",
        "allocation_first_pass_valid",
        "allocation_retry_count",
        "raw_output",
        "placeholder_reason",
    ]
    failure_columns = [
        "prompt_id",
        "domain",
        "candidate_id",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "error",
        "raw_output",
    ]
    diagnostic_columns = [
        "prompt_id",
        "domain",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "candidate_display_order",
        "ballot_field_order",
        "display_to_candidate_json",
        "enforce_exact_allocation",
        "first_pass_valid",
        "final_valid",
        "retry_count",
        "attempt_count",
        "first_raw_absolute_total",
        "final_raw_absolute_total",
        "final_validation_errors_json",
        "allocation_attempt_history_json",
        "first_raw_output",
        "final_raw_output",
    ]
    return (
        pd.DataFrame(rows, columns=columns),
        pd.DataFrame(failures, columns=failure_columns),
        pd.DataFrame(allocation_diagnostics, columns=diagnostic_columns),
    )


def main() -> None:
    args = parse_args()
    if args.num_candidates != 4:
        raise ValueError("Level 2.5 uses exactly four candidate IDs: A, B, C, D")
    if args.skip_candidate_generation and not args.candidates_csv:
        raise ValueError("--skip-candidate-generation requires --candidates-csv")
    if args.reuse_paired_inputs and not args.paired_run_dir:
        raise ValueError("--reuse-paired-inputs requires --paired-run-dir")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.normal_evaluator_repeats < 1:
        raise ValueError("--normal-evaluator-repeats must be at least 1")
    if args.judge_repeats < 0:
        raise ValueError("--judge-repeats must be nonnegative")
    if args.weak_selector_repeats < 0:
        raise ValueError("--weak-selector-repeats must be nonnegative")
    if args.allocation_max_retries < 0:
        raise ValueError("--allocation-max-retries must be nonnegative")

    set_vllm_safe_seed(args.seed)
    evaluator_models = parse_model_list(args.evaluator_models, args.candidate_model)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"level25_self_answer_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_csv, args.max_prompts)
    save_table(prompts, out_dir, "prompts")
    save_table(
        pd.DataFrame(
            [
                {
                    "backend": "vllm",
                    "experiment": "level25_self_answer_ideal_point",
                    "candidate_model": args.candidate_model,
                    "candidate_model_revision": args.candidate_model_revision,
                    "candidate_source": (
                        "imported_csv" if args.candidates_csv else "generated"
                    ),
                    "candidate_model_used": (
                        "" if args.candidates_csv else args.candidate_model
                    ),
                    "candidates_csv": args.candidates_csv,
                    "paired_run_dir": args.paired_run_dir,
                    "reuse_paired_inputs": args.reuse_paired_inputs,
                    "self_answer_source": (
                        "reused_paired_inputs"
                        if args.reuse_paired_inputs
                        else ("paired_run" if args.paired_run_dir else "generated")
                    ),
                    "evaluator_models_json": json.dumps(evaluator_models),
                    "judge_model": args.judge_model,
                    "fallback_model": args.fallback_model,
                    "model_revision": args.model_revision,
                    "evaluator_mode": args.evaluator_mode,
                    "normal_evaluator_repeats": args.normal_evaluator_repeats,
                    "max_prompts": args.max_prompts,
                    "self_answer_max_new_tokens": args.self_answer_max_new_tokens,
                    "self_answer_temperature": args.self_answer_temperature,
                    "self_answer_top_p": args.self_answer_top_p,
                    "evaluator_max_new_tokens": args.evaluator_max_new_tokens,
                    "no_vote_reasons": args.no_vote_reasons,
                    "randomize_ballot_field_order": args.randomize_ballot_field_order,
                    "self_answer_visible_reference": args.self_answer_visible_reference,
                    "private_critique_stage": False,
                    "judge_temperature": args.judge_temperature,
                    "evaluator_temperature": args.evaluator_temperature,
                    "evaluator_top_p": args.evaluator_top_p,
                    "weak_selector_temperature": args.weak_selector_temperature,
                    "judge_repeats": args.judge_repeats,
                    "weak_selector_repeats": args.weak_selector_repeats,
                    "shuffle_evaluator_candidates": args.shuffle_evaluator_candidates,
                    "shuffle_judge_candidates": args.shuffle_judge_candidates,
                    "shuffle_weak_selector_candidates": args.shuffle_weak_selector_candidates,
                    "show_candidate_labels": args.show_candidate_labels,
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "max_model_len": args.max_model_len,
                    "enforce_eager": args.enforce_eager,
                    "strict_borda": args.strict_borda,
                    "enforce_exact_allocation": args.enforce_exact_allocation,
                    "allocation_max_retries": args.allocation_max_retries,
                    "seed": args.seed,
                }
            ]
        ),
        out_dir,
        "run_config",
    )

    if args.candidates_csv:
        candidates = load_candidates_csv(args.candidates_csv, prompts, args.num_candidates)
    else:
        candidate_bundle = load_vllm_model(args.candidate_model, args)
        try:
            candidates = generate_candidates_vllm(prompts, candidate_bundle, args)
        finally:
            release_vllm_model(candidate_bundle)
    save_table(candidates, out_dir, "candidates")

    evaluation_frames = []
    failed_evaluation_frames = []
    allocation_diagnostic_frames = []
    self_answer_frames = []
    failed_self_answer_frames = []
    weak_selection_frames = []
    weak_vote_frames = []
    weak_failure_frames = []
    paired_diagnostic_frames = []

    for evaluator_model_name in evaluator_models:
        paired_orders = None
        paired_field_orders = None
        if args.paired_run_dir:
            (
                self_answers_part,
                failed_self_answers_part,
                paired_orders,
                paired_field_orders,
                paired_diagnostics_part,
            ) = load_paired_run_inputs(
                Path(args.paired_run_dir),
                prompts,
                candidates,
                evaluator_model_name,
                args,
                reuse_inputs=args.reuse_paired_inputs,
            )
            paired_diagnostic_frames.append(paired_diagnostics_part)
        evaluator_bundle = load_vllm_model(evaluator_model_name, args)
        try:
            if not args.paired_run_dir:
                self_answers_part, failed_self_answers_part = run_self_answers_vllm(
                    prompts,
                    evaluator_bundle,
                    args,
                )
            (
                evaluations_part,
                failed_evaluations_part,
                allocation_diagnostics_part,
            ) = run_self_answer_direct_votes_vllm(
                prompts,
                candidates,
                self_answers_part,
                evaluator_bundle,
                args,
                paired_orders=paired_orders,
                paired_field_orders=paired_field_orders,
            )
            weak_selections_part, weak_votes_part, weak_failures_part = run_weak_selector_vllm(
                prompts,
                candidates,
                evaluator_bundle,
                args,
            )
        finally:
            release_vllm_model(evaluator_bundle)

        self_answer_frames.append(self_answers_part)
        failed_self_answer_frames.append(failed_self_answers_part)
        evaluation_frames.append(evaluations_part)
        failed_evaluation_frames.append(failed_evaluations_part)
        allocation_diagnostic_frames.append(allocation_diagnostics_part)
        weak_selection_frames.append(weak_selections_part)
        weak_vote_frames.append(weak_votes_part)
        weak_failure_frames.append(weak_failures_part)

    self_answers = concat_frames(self_answer_frames)
    failed_self_answers = concat_frames(failed_self_answer_frames)
    evaluations = concat_frames(evaluation_frames)
    failed_evaluations = concat_frames(failed_evaluation_frames)
    allocation_diagnostics = concat_frames(allocation_diagnostic_frames)
    weak_selector_aggregations = concat_frames(weak_selection_frames)
    weak_selector_votes = concat_frames(weak_vote_frames)
    failed_weak_selector = concat_frames(weak_failure_frames)
    paired_diagnostics = concat_frames(paired_diagnostic_frames)

    save_table(self_answers, out_dir, "self_answers")
    save_table(failed_self_answers, out_dir, "failed_self_answers")
    save_table(paired_diagnostics, out_dir, "paired_input_diagnostics")
    save_table(evaluations, out_dir, "direct_votes")
    save_table(evaluations, out_dir, "prompted_evaluations")
    save_table(allocation_diagnostics, out_dir, "allocation_validation_diagnostics")
    save_table(vote_repair_diagnostics(evaluations), out_dir, "vote_repair_diagnostics")
    save_table(failed_evaluations, out_dir, "failed_evaluations")
    save_table(
        ballot_quality_diagnostics(evaluations, failed_evaluations),
        out_dir,
        "ballot_quality_diagnostics",
    )
    save_table(weak_selector_aggregations, out_dir, "weak_selector_results")
    save_table(weak_selector_votes, out_dir, "weak_selector_votes")
    save_table(failed_weak_selector, out_dir, "failed_weak_selector_results")

    aggregations, score_matrices = aggregate_all(evaluations)
    comparison_selections = pd.concat(
        [aggregations, weak_selector_aggregations],
        ignore_index=True,
    )
    save_table(aggregations, out_dir, "aggregations")
    save_table(comparison_selections, out_dir, "comparison_selections")
    save_table(score_matrices, out_dir, "vote_matrices_long")
    save_table(score_matrices, out_dir, "score_matrices_long")

    if args.judge_repeats > 0:
        judge_bundle = load_vllm_model(args.judge_model, args)
        try:
            judge_results, judge_votes, failed_judge_results = run_external_judge_vllm(
                prompts,
                candidates,
                judge_bundle,
                args,
            )
        finally:
            release_vllm_model(judge_bundle)
        selection_table, method_summary, domain_summary = build_summaries(
            comparison_selections,
            judge_results,
        )
    else:
        judge_results = pd.DataFrame(
            columns=[
                "prompt_id",
                "domain",
                "judge_model",
                "judge_repeats",
                "judge_winner",
                "judge_consensus_share",
                "judge_vote_counts_json",
            ]
        )
        judge_votes = pd.DataFrame(
            columns=[
                "prompt_id",
                "domain",
                "judge_model",
                "judge_repeat",
                "judge_winner",
                "raw_output",
            ]
        )
        failed_judge_results = pd.DataFrame(
            columns=["prompt_id", "domain", "judge_model", "error", "raw_output"]
        )
        selection_table = pd.DataFrame()
        method_summary = pd.DataFrame()
        domain_summary = pd.DataFrame()
    save_table(judge_results, out_dir, "external_judge_results")
    save_table(judge_votes, out_dir, "external_judge_votes")
    save_table(failed_judge_results, out_dir, "failed_external_judge_results")

    save_table(selection_table, out_dir, "selection_table")
    save_table(method_summary, out_dir, "method_summary")
    save_table(domain_summary, out_dir, "domain_summary")

    (
        eval_diag,
        candidate_score_diag,
        completeness_diag,
        flat_score_diag,
        duplicate_score_diag,
        evaluator_correlation_diag,
        position_diag,
    ) = diagnostics(evaluations, candidates, aggregations)
    save_table(eval_diag, out_dir, "evaluation_diagnostics")
    save_table(candidate_score_diag, out_dir, "candidate_score_diagnostics")
    save_table(completeness_diag, out_dir, "evaluator_completeness_diagnostics")
    save_table(flat_score_diag, out_dir, "flat_score_diagnostics")
    save_table(duplicate_score_diag, out_dir, "duplicate_score_diagnostics")
    save_table(evaluator_correlation_diag, out_dir, "evaluator_correlation_diagnostics")
    save_table(position_diag, out_dir, "candidate_position_diagnostics")

    archive_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("\nMethod summary")
    print(method_summary.to_string(index=False) if not method_summary.empty else "No methods summarized.")
    print("\nDomain summary")
    print(domain_summary.to_string(index=False) if not domain_summary.empty else "No domains summarized.")
    print("\nSelection table")
    print(selection_table.to_string(index=False) if not selection_table.empty else "No selections.")
    print(f"Saved outputs to {out_dir}")
    print(f"Created archive {archive_path}")


if __name__ == "__main__":
    main()
