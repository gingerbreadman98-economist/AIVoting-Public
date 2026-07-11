#!/usr/bin/env python3
"""Pure prompt builders shared by Level 2.5 voting and activation replay."""

from __future__ import annotations

from typing import Any

import pandas as pd

from level1_direct_vote_eval import CANDIDATE_LABELS, displayed_candidates


DISPLAY_LABELS = ["1", "2", "3", "4"]
VOTE_FIELDS = ("best_pick", "borda_ranking", "signed_allocation")
DEFAULT_NORMAL_CRITERION = (
    "Judge overall answer quality for the original user prompt. Balance "
    "correctness, helpfulness, completeness, safety, clarity, and fit "
    "to the user's request."
)


def self_answer_prompt(
    user_prompt: str,
    evaluator: dict[str, str],
) -> list[dict[str, str]]:
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


def self_answer_evaluator_prompt(
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
    strict_absolute_allocation: bool = False,
    vote_field_order: list[str] | tuple[str, ...] | None = None,
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
    shown_order = [
        str(display_to_candidate[row.display_id])
        for row in shown_candidates.itertuples(index=False)
    ]
    labels = sorted(display_to_candidate.keys())
    example_ids = labels + DISPLAY_LABELS[len(labels) :]
    ex1, ex2, ex3, ex4 = example_ids[:4]
    criterion = evaluator.get("criterion", DEFAULT_NORMAL_CRITERION)

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
    if strict_absolute_allocation:
        allocation_example = (
            f"{ex1}=+45, {ex2}=+15, {ex3}=-25, {ex4}=-15"
        )
        allocation_schema = (
            f'"signed_allocation_cents":[{{"candidate_id":"{ex1}","cents":45}},'
            f'{{"candidate_id":"{ex2}","cents":15}},'
            f'{{"candidate_id":"{ex3}","cents":-25}},'
            f'{{"candidate_id":"{ex4}","cents":-15}}],'
        )
        strict_budget_instruction = (
            "This is one shared budget, not separate positive and negative budgets. "
            "Every positive or negative cent consumes the same 100-cent total. "
            "Before responding, calculate sum(abs(cents)) across all four candidates; "
            "it must equal exactly 100. A ballot with +100 support and -100 opposition "
            "has total 200 and is invalid. "
        )
    else:
        allocation_example = f"{ex1}=+60, {ex2}=-40, {ex3}=0, {ex4}=0"
        allocation_schema = (
            f'"signed_allocation_cents":[{{"candidate_id":"{ex1}","cents":100}},'
            f'{{"candidate_id":"{ex2}","cents":0}},'
            f'{{"candidate_id":"{ex3}","cents":0}},'
            f'{{"candidate_id":"{ex4}","cents":0}}],'
        )
        strict_budget_instruction = ""
    if vote_field_order is None:
        field_order = VOTE_FIELDS
        randomization_instruction = ""
    else:
        field_order = tuple(str(field).strip() for field in vote_field_order)
        if len(field_order) != len(VOTE_FIELDS) or set(field_order) != set(VOTE_FIELDS):
            raise ValueError(
                "vote_field_order must contain best_pick, borda_ranking, and "
                "signed_allocation exactly once."
            )
        randomization_instruction = (
            "Consider and report the ballot fields in this order: "
            + ", then ".join(field_order)
            + ". Do not let an earlier field mechanically determine a later field.\n\n"
        )

    allocation_shape = f"{signed_reason_schema}{allocation_schema}\"absolute_cents_total\":100"
    shape_by_field = {
        "best_pick": f'"best_pick":"{ex1}"',
        "borda_ranking": f'"borda_ranking":[["{ex1}"],["{ex2}"],["{ex3}"],["{ex4}"]]',
        "signed_allocation": allocation_shape,
    }
    json_shape = '{"votes":{' + ",".join(shape_by_field[field] for field in field_order) + "}}"
    instruction_by_field = {
        "best_pick": (
            "best_pick: one candidate ID, the single best candidate under "
            f"{vote_scope}.\n"
        ),
        "borda_ranking": borda_instruction,
        "signed_allocation": (
            f"{signed_reason_instruction}"
            "signed_allocation_cents: this election gives you exactly 100 "
            "total influence cents. Positive cents help a candidate win. "
            "Negative cents hurt a candidate's chance to win. Neutral "
            "candidates get 0. Helping and hurting are not separate budgets: "
            "sum(abs(cents)) across all candidates must equal exactly 100. "
            f"{strict_budget_instruction}"
            f"Valid example: {allocation_example}. "
            f"Invalid example: {ex1}=+100 and {ex2}=-100 because the "
            "absolute total is 200. Use integer cents only. Each cents "
            "value must be between -100 and 100. Return one object for each "
            "candidate ID with candidate_id and cents. Include "
            "absolute_cents_total equal to the sum of absolute cents.\n"
        ),
    }
    field_instructions = "".join(instruction_by_field[field] for field in field_order)

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
                f"{randomization_instruction}"
                "Use exactly this JSON shape:\n"
                f"{json_shape}\n"
                "Do not make votes an array. Put the allocation array inside "
                "votes.signed_allocation_cents.\n\n"
                f"{field_instructions}\n"
                f"Valid candidate IDs are: {', '.join(labels)}. "
                f"{reference_validity_note}{reason_instruction}"
            ),
        },
    ]
    return messages, display_to_candidate, shown_order
