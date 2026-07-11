#!/usr/bin/env python3
"""
Level 1 direct-vote evaluator using vLLM batched inference.

This script preserves the independence structure of level1_direct_vote_eval.py:
each evaluator, weak-selector, judge, and candidate-generation request is still a
separate prompt. vLLM only batches those separate prompts at the inference engine
level, which improves throughput without putting multiple elections into one
shared chat context.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

# vLLM launches worker processes. If Python/torch has touched CUDA before those
# workers are created, Linux's default fork method can fail with:
# "Cannot re-initialize CUDA in forked subprocess." Force spawn before vLLM is
# imported anywhere in this process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from level1_direct_vote_eval import (
    CANDIDATE_LABELS,
    aggregate_all,
    ballot_quality_diagnostics,
    build_direct_evaluators,
    build_summaries,
    candidate_messages,
    concat_frames,
    diagnostics,
    evaluator_prompt,
    evaluator_reaction_prompt,
    extract_json,
    failure_diagnostics,
    judge_prompt,
    load_candidates_csv,
    load_prompts,
    save_table,
    tie_aware_winner,
    translate_display_ids,
    validate_direct_votes,
    vote_repair_diagnostics,
    weak_selector_prompt,
    is_placeholder_reason,
)


def set_vllm_safe_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


@dataclass
class VLLMBundle:
    llm: Any
    tokenizer: Any
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--evaluator-models",
        default="",
        help="Comma-separated evaluator models. Empty means use --candidate-model.",
    )
    parser.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompts-csv", default="")
    parser.add_argument("--candidates-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--evaluator-max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--evaluator-mode",
        choices=["role", "normal"],
        default="normal",
    )
    parser.add_argument("--normal-evaluator-repeats", type=int, default=5)
    parser.add_argument("--evaluator-temperature", type=float, default=0.7)
    parser.add_argument("--evaluator-top-p", type=float, default=0.95)
    parser.add_argument("--judge-temperature", type=float, default=0.05)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--weak-selector-temperature", type=float, default=0.05)
    parser.add_argument("--weak-selector-top-p", type=float, default=0.95)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--weak-selector-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-vote-reasons", action="store_true")
    parser.add_argument(
        "--debate-stage",
        action="store_true",
        help=(
            "Run a pre-vote reaction round. Each evaluator first comments on "
            "the candidate space; final voters then see all reactions before "
            "casting ballots."
        ),
    )
    parser.add_argument(
        "--private-critique-stage",
        action="store_true",
        help=(
            "Run a pre-vote critique round where each evaluator first critiques "
            "the candidate space privately, then sees only its own critique "
            "before casting final ballots."
        ),
    )
    parser.add_argument(
        "--debate-max-new-tokens",
        type=int,
        default=192,
        help="Max new tokens for each pre-vote debate reaction.",
    )
    parser.add_argument("--shuffle-evaluator-candidates", action="store_true")
    parser.add_argument("--shuffle-judge-candidates", action="store_true")
    parser.add_argument("--shuffle-weak-selector-candidates", action="store_true")
    parser.add_argument(
        "--show-candidate-labels",
        action="store_true",
        help=(
            "Show true candidate IDs A/B/C/D in prompts. This is the default; "
            "use --hide-candidate-labels to anonymize labels."
        ),
    )
    parser.add_argument(
        "--hide-candidate-labels",
        dest="show_candidate_labels",
        action="store_false",
        help=(
            "Hide true candidate IDs behind anonymous display IDs 1/2/3/4 "
            "and map back internally."
        ),
    )
    parser.set_defaults(show_candidate_labels=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of independent prompts to send to vLLM per generate() call.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--skip-candidate-generation",
        action="store_true",
        help="Require --candidates-csv and skip candidate generation.",
    )
    return parser.parse_args()


def parse_model_list(raw: str, default_model: str) -> list[str]:
    if not raw.strip():
        return [default_model]
    models = [part.strip() for part in raw.split(",") if part.strip()]
    return list(dict.fromkeys(models))


import re
import time


_KV_CACHE_LEN_RE = re.compile(
    r"estimated maximum model length is (\d+)", re.IGNORECASE
)


def _kv_cache_safe_len(exc: Exception) -> int | None:
    """If vLLM failed only because the KV cache couldn't fit the requested
    context length, vLLM's own error message names the largest context
    length that WOULD fit in the memory actually available right now (e.g.
    "...the estimated maximum model length is 4432"). Parse that number and
    apply a small safety margin. Returns None if the message doesn't match
    (i.e. this wasn't a fixable context-length sizing error -- some other
    failure, like a bad model name or true OOM on the weights themselves,
    where retrying with a smaller context wouldn't help).
    """
    match = _KV_CACHE_LEN_RE.search(str(exc))
    if not match:
        return None
    estimated = int(match.group(1))
    # Round down to a multiple of 128 and leave ~10% headroom so the retry
    # isn't shaving things so close it can fail again on minor fluctuations.
    safe = max(512, int(estimated * 0.9))
    return (safe // 128) * 128


def _free_gpu_memory(pause_seconds: float = 2.0) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    # A crashed vLLM engine subprocess prints "destroy_process_group() was
    # not called before program exit" -- its CUDA context can take a moment
    # to be fully reclaimed by the driver. Give it a beat before the next
    # LLM() call in this same process reads how much GPU memory is free;
    # without this pause, back-to-back attempts can undercount free memory
    # and fail even at context lengths that would otherwise fit.
    time.sleep(pause_seconds)


def _try_load(model_name: str, kwargs: dict[str, Any]) -> tuple[Any | None, Exception | None]:
    from vllm import LLM

    try:
        return LLM(**kwargs), None
    except Exception as exc:
        return None, exc


def load_vllm_model(model_name: str, args: argparse.Namespace) -> VLLMBundle:
    base_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
    }
    model_revision = str(getattr(args, "model_revision", "") or "").strip()
    evaluator_models = {
        item.strip()
        for item in str(getattr(args, "evaluator_models", "") or "").split(",")
        if item.strip()
    }
    revision_models = evaluator_models | {str(getattr(args, "fallback_model", "") or "")}
    if model_revision and model_name in revision_models:
        base_kwargs["revision"] = model_revision
    candidate_revision = str(getattr(args, "candidate_model_revision", "") or "").strip()
    if candidate_revision and model_name == str(getattr(args, "candidate_model", "") or ""):
        base_kwargs["revision"] = candidate_revision
    if args.max_model_len > 0:
        base_kwargs["max_model_len"] = args.max_model_len

    # Attempt 1: requested settings as-is (native context length unless the
    # user passed --max-model-len explicitly).
    llm, exc = _try_load(model_name, base_kwargs)
    if llm is not None:
        return VLLMBundle(llm=llm, tokenizer=llm.get_tokenizer(), name=model_name)

    print(f"Failed to load {model_name}: {exc}")

    # Attempt 2: ONLY if this was a fixable KV-cache sizing error (not a true
    # OOM on the weights, missing model, etc.) AND the user didn't already
    # pin --max-model-len, retry the SAME model exactly once more using the
    # safe context length vLLM itself reported. Capped at one retry -- on a
    # tight-memory GPU, repeatedly hammering LLM() with the same multi-GB
    # model in one process has been observed to degrade available GPU
    # memory further (each failed engine init logs "destroy_process_group()
    # was not called", suggesting incomplete teardown), so more retries risk
    # making the eventual fallback attempt fail too.
    safe_len = None if args.max_model_len > 0 else _kv_cache_safe_len(exc)
    if safe_len is not None:
        _free_gpu_memory()
        retry_kwargs = dict(base_kwargs)
        retry_kwargs["max_model_len"] = safe_len
        llm, retry_exc = _try_load(model_name, retry_kwargs)
        if llm is not None:
            print(
                f"Loaded {model_name} with max_model_len={safe_len} after "
                "the native context length didn't fit in available GPU "
                "memory for the KV cache."
            )
            return VLLMBundle(llm=llm, tokenizer=llm.get_tokenizer(), name=model_name)
        print(f"Retry of {model_name} at max_model_len={safe_len} also failed: {retry_exc}")
        exc = retry_exc

    if model_name == args.fallback_model:
        raise exc

    _free_gpu_memory()
    print(f"Falling back to {args.fallback_model}")
    fallback_kwargs: dict[str, Any] = {
        "model": args.fallback_model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len > 0:
        fallback_kwargs["max_model_len"] = args.max_model_len
    fallback_llm, fallback_exc = _try_load(args.fallback_model, fallback_kwargs)
    if fallback_llm is None:
        # Same KV-cache-sizing rescue applies to the fallback model too --
        # don't let a context-length issue on the fallback take down the
        # whole run when we already know how to fix it.
        safe_len = (
            None if args.max_model_len > 0 else _kv_cache_safe_len(fallback_exc)
        )
        if safe_len is not None:
            _free_gpu_memory()
            retry_kwargs = dict(fallback_kwargs)
            retry_kwargs["max_model_len"] = safe_len
            fallback_llm, fallback_exc = _try_load(args.fallback_model, retry_kwargs)
        if fallback_llm is None:
            raise fallback_exc
    return VLLMBundle(
        llm=fallback_llm,
        tokenizer=fallback_llm.get_tokenizer(),
        name=args.fallback_model,
    )


def release_vllm_model(bundle: VLLMBundle | None) -> None:
    if bundle is None:
        return
    try:
        del bundle.llm
        del bundle.tokenizer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def render_messages(bundle: VLLMBundle, messages: list[dict[str, str]]) -> str:
    return bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batched_generate_texts(
    bundle: VLLMBundle,
    messages_batch: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    desc: str,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs: list[str] = []
    for start in tqdm(range(0, len(messages_batch), batch_size), desc=desc):
        chunk_messages = messages_batch[start : start + batch_size]
        prompts = [render_messages(bundle, messages) for messages in chunk_messages]
        chunk_outputs = bundle.llm.generate(prompts, sampling_params)
        for output in chunk_outputs:
            outputs.append(output.outputs[0].text.strip())
    return outputs


def generate_candidates_vllm(
    prompts: pd.DataFrame,
    bundle: VLLMBundle,
    args: argparse.Namespace,
) -> pd.DataFrame:
    jobs: list[dict[str, Any]] = []
    messages_batch = []
    for prompt in prompts.itertuples(index=False):
        for idx, label in enumerate(CANDIDATE_LABELS[: args.num_candidates]):
            jobs.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "candidate": label,
                }
            )
            messages_batch.append(candidate_messages(prompt.user_prompt, idx))

    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        desc="Generating candidates",
    )

    rows = []
    for job, answer in zip(jobs, outputs):
        rows.append({**job, "candidate_answer": answer})
    return pd.DataFrame(rows)


def run_direct_votes_vllm(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    bundle: VLLMBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompt_lookup = prompts.set_index("prompt_id")["user_prompt"].to_dict()
    domain_lookup = candidates.drop_duplicates("prompt_id").set_index("prompt_id")[
        "domain"
    ].to_dict()
    labels = sorted(candidates["candidate"].unique())
    direct_evaluators = build_direct_evaluators(args)

    jobs = [
        {"prompt_id": prompt_id, "evaluator": evaluator}
        for prompt_id in candidates["prompt_id"].drop_duplicates().tolist()
        for evaluator in direct_evaluators
    ]
    random.shuffle(jobs)

    debate_reactions = []
    failed_debate_reactions = []
    debate_context_by_prompt: dict[str, str] = {}
    private_context_by_job: dict[tuple[str, str], str] = {}
    if args.debate_stage or args.private_critique_stage:
        reaction_jobs = []
        reaction_messages_batch = []
        for job in jobs:
            prompt_id = job["prompt_id"]
            group = candidates[candidates["prompt_id"] == prompt_id]
            messages, display_to_candidate = evaluator_reaction_prompt(
                prompt_lookup[prompt_id],
                group,
                job["evaluator"],
                args.shuffle_evaluator_candidates,
                args.show_candidate_labels,
            )
            reaction_job = dict(job)
            reaction_job["display_to_candidate"] = display_to_candidate
            reaction_jobs.append(reaction_job)
            reaction_messages_batch.append(messages)

        reaction_outputs = batched_generate_texts(
            bundle,
            reaction_messages_batch,
            max_new_tokens=args.debate_max_new_tokens,
            temperature=args.evaluator_temperature,
            top_p=args.evaluator_top_p,
            batch_size=args.batch_size,
            desc="Debate reactions",
        )

        for job, output in zip(reaction_jobs, reaction_outputs):
            prompt_id = job["prompt_id"]
            evaluator = job["evaluator"]
            try:
                parsed = translate_display_ids(
                    extract_json(output),
                    job["display_to_candidate"],
                )
                reaction = str(parsed.get("reaction", "")).strip()
                if is_placeholder_reason(reaction):
                    raise ValueError("placeholder debate reaction")
                debate_reactions.append(
                    {
                        "prompt_id": prompt_id,
                        "domain": domain_lookup[prompt_id],
                        "evaluator_model": bundle.name,
                        "evaluator": evaluator["name"],
                        "evaluator_id": f"{bundle.name}::{evaluator['name']}",
                        "reaction": reaction,
                        "raw_output": output,
                    }
                )
                if args.private_critique_stage:
                    private_context_by_job[
                        (prompt_id, f"{bundle.name}::{evaluator['name']}")
                    ] = f"{evaluator['name']} private critique: {reaction}"
            except Exception as exc:
                failed_debate_reactions.append(
                    {
                        "prompt_id": prompt_id,
                        "evaluator_model": bundle.name,
                        "evaluator": evaluator["name"],
                        "evaluator_id": f"{bundle.name}::{evaluator['name']}",
                        "error": repr(exc),
                        "raw_output": output,
                    }
                )

        if args.debate_stage:
            reactions_by_prompt: dict[str, list[str]] = {}
            for row in debate_reactions:
                reactions_by_prompt.setdefault(row["prompt_id"], []).append(
                    f"{row['evaluator']}: {row['reaction']}"
                )
            debate_context_by_prompt = {
                prompt_id: "\n".join(reactions)
                for prompt_id, reactions in reactions_by_prompt.items()
            }

    messages_batch = []
    for job in jobs:
        prompt_id = job["prompt_id"]
        group = candidates[candidates["prompt_id"] == prompt_id]
        messages, display_to_candidate = evaluator_prompt(
            prompt_lookup[prompt_id],
            group,
            job["evaluator"],
            args.shuffle_evaluator_candidates,
            not args.no_vote_reasons,
            args.show_candidate_labels,
            (
                private_context_by_job.get(
                    (prompt_id, f"{bundle.name}::{job['evaluator']['name']}"),
                    "",
                )
                if args.private_critique_stage
                else debate_context_by_prompt.get(prompt_id, "")
            ),
        )
        job["display_to_candidate"] = display_to_candidate
        messages_batch.append(messages)

    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.evaluator_max_new_tokens,
        temperature=args.evaluator_temperature,
        top_p=args.evaluator_top_p,
        batch_size=args.batch_size,
        desc="Direct vote evaluations",
    )

    rows = []
    failures = []
    for job, output in zip(jobs, outputs):
        prompt_id = job["prompt_id"]
        evaluator = job["evaluator"]
        try:
            parsed = translate_display_ids(
                extract_json(output),
                job["display_to_candidate"],
            )
            for item in validate_direct_votes(
                parsed,
                labels,
                not args.no_vote_reasons,
            ):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "domain": domain_lookup[prompt_id],
                        "candidate_id": item["candidate_id"],
                        "evaluator_model": bundle.name,
                        "evaluator": evaluator["name"],
                        "evaluator_id": f"{bundle.name}::{evaluator['name']}",
                        "best_pick_vote": item["best_pick_vote"],
                        "borda_points": item["borda_points"],
                        "allocation": item["allocation"],
                        "reason": item["reason"],
                        "vote_repairs_json": item["vote_repairs_json"],
                        "vote_repair_count": item["vote_repair_count"],
                        "raw_output": output,
                        "placeholder_reason": False,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "prompt_id": prompt_id,
                    "candidate_id": ",".join(labels),
                    "evaluator_model": bundle.name,
                    "evaluator": evaluator["name"],
                    "evaluator_id": f"{bundle.name}::{evaluator['name']}",
                    "error": repr(exc),
                    "raw_output": output,
                }
            )

    evaluation_columns = [
        "prompt_id",
        "domain",
        "candidate_id",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "best_pick_vote",
        "borda_points",
        "allocation",
        "reason",
        "vote_repairs_json",
        "vote_repair_count",
        "raw_output",
        "placeholder_reason",
    ]
    failure_columns = [
        "prompt_id",
        "candidate_id",
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "error",
        "raw_output",
    ]
    return (
        pd.DataFrame(rows, columns=evaluation_columns),
        pd.DataFrame(failures, columns=failure_columns),
        pd.DataFrame(
            debate_reactions,
            columns=[
                "prompt_id",
                "domain",
                "evaluator_model",
                "evaluator",
                "evaluator_id",
                "reaction",
                "raw_output",
            ],
        ),
        pd.DataFrame(
            failed_debate_reactions,
            columns=[
                "prompt_id",
                "evaluator_model",
                "evaluator",
                "evaluator_id",
                "error",
                "raw_output",
            ],
        ),
    )


def run_weak_selector_vllm(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    bundle: VLLMBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_labels = set(candidates["candidate"].unique())
    jobs = []
    messages_batch = []
    for prompt in prompts.itertuples(index=False):
        group = candidates[candidates["prompt_id"] == prompt.prompt_id]
        for repeat_idx in range(args.weak_selector_repeats):
            messages, display_to_candidate = weak_selector_prompt(
                prompt.user_prompt,
                group,
                args.shuffle_weak_selector_candidates,
                args.show_candidate_labels,
            )
            jobs.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "user_prompt": prompt.user_prompt,
                    "repeat_idx": repeat_idx,
                    "display_to_candidate": display_to_candidate,
                }
            )
            messages_batch.append(messages)

    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.weak_selector_temperature,
        top_p=args.weak_selector_top_p,
        batch_size=args.batch_size,
        desc="Weak selector",
    )

    failures = []
    vote_rows = []
    prompt_votes: dict[tuple[str, str], list[str]] = {}
    for job, output in zip(jobs, outputs):
        try:
            parsed = translate_display_ids(
                extract_json(output),
                job["display_to_candidate"],
            )
            best = str(parsed["best_candidate"]).strip().upper()
            if best not in candidate_labels:
                raise ValueError(f"weak selector returned invalid candidate {best!r}")
            reason = str(parsed.get("reason", "")).strip()
            if is_placeholder_reason(reason):
                raise ValueError("placeholder weak selector reason")
            key = (job["prompt_id"], job["domain"])
            prompt_votes.setdefault(key, []).append(best)
            vote_rows.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "selector_model": bundle.name,
                    "repeat_idx": job["repeat_idx"],
                    "selection": best,
                    "reason": reason,
                    "raw_output": output,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "prompt_id": job["prompt_id"],
                    "repeat_idx": job["repeat_idx"],
                    "selector_model": bundle.name,
                    "error": repr(exc),
                    "raw_output": output,
                }
            )

    rows = []
    for (prompt_id, domain), votes in prompt_votes.items():
        vote_counts = {label: votes.count(label) for label in sorted(set(votes))}
        rows.append(
            {
                "prompt_id": prompt_id,
                "domain": domain,
                "method": f"single_weak_selector:{bundle.name}",
                "selection": tie_aware_winner(vote_counts),
                "scores_json": json.dumps(vote_counts, sort_keys=True),
                "selector_repeats": len(votes),
                "selector_consensus_share": max(vote_counts.values()) / len(votes),
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "prompt_id",
                "domain",
                "method",
                "selection",
                "scores_json",
                "selector_repeats",
                "selector_consensus_share",
            ],
        ),
        pd.DataFrame(
            vote_rows,
            columns=[
                "prompt_id",
                "domain",
                "selector_model",
                "repeat_idx",
                "selection",
                "reason",
                "raw_output",
            ],
        ),
        pd.DataFrame(
            failures,
            columns=["prompt_id", "repeat_idx", "selector_model", "error", "raw_output"],
        ),
    )


def run_external_judge_vllm(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    bundle: VLLMBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_labels = set(candidates["candidate"].unique())
    jobs = []
    messages_batch = []
    for prompt in prompts.itertuples(index=False):
        group = candidates[candidates["prompt_id"] == prompt.prompt_id]
        for repeat_idx in range(args.judge_repeats):
            messages, display_to_candidate = judge_prompt(
                prompt.user_prompt,
                group,
                args.shuffle_judge_candidates,
                args.show_candidate_labels,
            )
            jobs.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "repeat_idx": repeat_idx,
                    "display_to_candidate": display_to_candidate,
                }
            )
            messages_batch.append(messages)

    outputs = batched_generate_texts(
        bundle,
        messages_batch,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        batch_size=args.batch_size,
        desc="External judge",
    )

    failures = []
    vote_rows = []
    prompt_votes: dict[tuple[str, str], list[str]] = {}
    for job, output in zip(jobs, outputs):
        try:
            parsed = translate_display_ids(
                extract_json(output),
                job["display_to_candidate"],
            )
            best = str(parsed["best_candidate"]).strip().upper()
            if best not in candidate_labels:
                raise ValueError(f"judge returned invalid candidate {best!r}")
            reason = str(parsed.get("reason", "")).strip()
            if is_placeholder_reason(reason):
                raise ValueError("placeholder judge reason")
            key = (job["prompt_id"], job["domain"])
            prompt_votes.setdefault(key, []).append(best)
            vote_rows.append(
                {
                    "prompt_id": job["prompt_id"],
                    "domain": job["domain"],
                    "judge_model": bundle.name,
                    "repeat_idx": job["repeat_idx"],
                    "judge_winner": best,
                    "reason": reason,
                    "raw_output": output,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "prompt_id": job["prompt_id"],
                    "repeat_idx": job["repeat_idx"],
                    "error": repr(exc),
                    "raw_output": output,
                }
            )

    rows = []
    for (prompt_id, domain), votes in prompt_votes.items():
        vote_counts = {label: votes.count(label) for label in sorted(set(votes))}
        rows.append(
            {
                "prompt_id": prompt_id,
                "domain": domain,
                "judge_model": bundle.name,
                "judge_winner": tie_aware_winner(vote_counts),
                "judge_repeats": len(votes),
                "judge_vote_counts_json": json.dumps(vote_counts, sort_keys=True),
                "judge_consensus_share": max(vote_counts.values()) / len(votes),
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "prompt_id",
                "domain",
                "judge_model",
                "judge_winner",
                "judge_repeats",
                "judge_vote_counts_json",
                "judge_consensus_share",
            ],
        ),
        pd.DataFrame(
            vote_rows,
            columns=[
                "prompt_id",
                "domain",
                "judge_model",
                "repeat_idx",
                "judge_winner",
                "reason",
                "raw_output",
            ],
        ),
        pd.DataFrame(failures, columns=["prompt_id", "repeat_idx", "error", "raw_output"]),
    )


def main() -> None:
    args = parse_args()
    if args.num_candidates != 4:
        raise ValueError("Level 1 uses exactly four candidate IDs: A, B, C, D")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.judge_repeats < 1:
        raise ValueError("--judge-repeats must be at least 1")
    if args.weak_selector_repeats < 1:
        raise ValueError("--weak-selector-repeats must be at least 1")
    if args.skip_candidate_generation and not args.candidates_csv:
        raise ValueError("--skip-candidate-generation requires --candidates-csv")
    if args.debate_stage and args.private_critique_stage:
        raise ValueError("Use either --debate-stage or --private-critique-stage, not both")
    if args.debate_stage and not args.show_candidate_labels:
        raise ValueError(
            "--debate-stage requires visible candidate labels. Do not combine "
            "it with --hide-candidate-labels, because shuffled anonymous "
            "display IDs would make shared reactions ambiguous."
        )

    set_vllm_safe_seed(args.seed)
    evaluator_models = parse_model_list(args.evaluator_models, args.candidate_model)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"level1_direct_vote_vllm_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_csv, args.max_prompts)
    save_table(prompts, out_dir, "prompts")
    save_table(
        pd.DataFrame(
            [
                {
                    "backend": "vllm",
                    "candidate_model": args.candidate_model,
                    "candidates_csv": args.candidates_csv,
                    "evaluator_models_json": json.dumps(evaluator_models),
                    "judge_model": args.judge_model,
                    "fallback_model": args.fallback_model,
                    "evaluator_mode": args.evaluator_mode,
                    "normal_evaluator_repeats": args.normal_evaluator_repeats,
                    "evaluator_temperature": args.evaluator_temperature,
                    "evaluator_top_p": args.evaluator_top_p,
                    "evaluator_max_new_tokens": args.evaluator_max_new_tokens,
                    "no_vote_reasons": args.no_vote_reasons,
                    "debate_stage": args.debate_stage,
                    "private_critique_stage": args.private_critique_stage,
                    "debate_max_new_tokens": args.debate_max_new_tokens,
                    "judge_temperature": args.judge_temperature,
                    "judge_top_p": args.judge_top_p,
                    "judge_repeats": args.judge_repeats,
                    "weak_selector_temperature": args.weak_selector_temperature,
                    "weak_selector_top_p": args.weak_selector_top_p,
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
    debate_reaction_frames = []
    failed_debate_reaction_frames = []
    weak_selection_frames = []
    weak_vote_frames = []
    weak_failure_frames = []
    for evaluator_model_name in evaluator_models:
        evaluator_bundle = load_vllm_model(evaluator_model_name, args)
        try:
            (
                evaluations_part,
                failed_evaluations_part,
                debate_reactions_part,
                failed_debate_reactions_part,
            ) = run_direct_votes_vllm(prompts, candidates, evaluator_bundle, args)
            weak_selections_part, weak_votes_part, weak_failures_part = (
                run_weak_selector_vllm(prompts, candidates, evaluator_bundle, args)
            )
        finally:
            release_vllm_model(evaluator_bundle)
        evaluation_frames.append(evaluations_part)
        failed_evaluation_frames.append(failed_evaluations_part)
        debate_reaction_frames.append(debate_reactions_part)
        failed_debate_reaction_frames.append(failed_debate_reactions_part)
        weak_selection_frames.append(weak_selections_part)
        weak_vote_frames.append(weak_votes_part)
        weak_failure_frames.append(weak_failures_part)

    evaluations = concat_frames(evaluation_frames)
    failed_evaluations = concat_frames(failed_evaluation_frames)
    debate_reactions = concat_frames(debate_reaction_frames)
    failed_debate_reactions = concat_frames(failed_debate_reaction_frames)
    weak_selector_aggregations = concat_frames(weak_selection_frames)
    weak_selector_votes = concat_frames(weak_vote_frames)
    failed_weak_selector = concat_frames(weak_failure_frames)

    save_table(evaluations, out_dir, "direct_votes")
    save_table(evaluations, out_dir, "prompted_evaluations")
    save_table(debate_reactions, out_dir, "debate_reactions")
    save_table(failed_debate_reactions, out_dir, "failed_debate_reactions")
    save_table(
        failure_diagnostics(failed_debate_reactions),
        out_dir,
        "failed_debate_reaction_diagnostics",
    )
    save_table(vote_repair_diagnostics(evaluations), out_dir, "vote_repair_diagnostics")
    save_table(failed_evaluations, out_dir, "failed_evaluations")
    save_table(
        ballot_quality_diagnostics(evaluations, failed_evaluations),
        out_dir,
        "ballot_quality_diagnostics",
    )
    save_table(
        failure_diagnostics(failed_evaluations),
        out_dir,
        "failed_evaluation_diagnostics",
    )
    save_table(weak_selector_aggregations, out_dir, "weak_selector_results")
    save_table(weak_selector_votes, out_dir, "weak_selector_votes")
    save_table(failed_weak_selector, out_dir, "failed_weak_selector_results")
    save_table(
        failure_diagnostics(failed_weak_selector),
        out_dir,
        "failed_weak_selector_diagnostics",
    )

    aggregations, score_matrices = aggregate_all(evaluations)
    comparison_selections = concat_frames([aggregations, weak_selector_aggregations])
    save_table(aggregations, out_dir, "aggregations")
    save_table(comparison_selections, out_dir, "comparison_selections")
    save_table(score_matrices, out_dir, "vote_matrices_long")
    save_table(score_matrices, out_dir, "score_matrices_long")

    judge_bundle = load_vllm_model(args.judge_model, args)
    try:
        judge_results, judge_votes, failed_judge_results = run_external_judge_vllm(
            prompts, candidates, judge_bundle, args
        )
    finally:
        release_vllm_model(judge_bundle)
    save_table(judge_results, out_dir, "external_judge_results")
    save_table(judge_votes, out_dir, "external_judge_votes")
    save_table(failed_judge_results, out_dir, "failed_external_judge_results")
    save_table(
        failure_diagnostics(failed_judge_results),
        out_dir,
        "failed_external_judge_diagnostics",
    )

    selection_table, method_summary, domain_summary = build_summaries(
        comparison_selections,
        judge_results,
    )
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

    print("\nMethod summary")
    print(method_summary.to_string(index=False))
    print("\nDomain summary")
    print(domain_summary.to_string(index=False))
    print("\nSelection table")
    print(selection_table.to_string(index=False))

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved outputs to {out_dir}")
    print(f"Created archive {archive}")


if __name__ == "__main__":
    main()
