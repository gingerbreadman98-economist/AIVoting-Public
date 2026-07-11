#!/usr/bin/env python3
"""
Level 1 direct-vote evaluator prototype for aggregation experiments.

This script generates candidate answers, asks independent evaluator agents to
see all candidates for one prompt and directly produce valid ballots for several
aggregation systems, aggregates those ballots, and compares each resulting
selection against a stronger external judge.

Unlike the prompted-score variants, evaluators are intentionally told they are
casting valid votes. The script does not collect raw utility scores from them.

It uses local/open-source Hugging Face models through transformers. It does not
call hosted model APIs.

For independence experiments, pass same-size evaluator models with
--evaluator-models. By default the script treats each model/criterion pair as
one independent evaluator. Pass --evaluator-mode normal to instead use repeated
ordinary evaluator calls where independence comes from sampling variation.
It compares aggregation against a repeated stronger-model judge and includes a
single weak-selector baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CANDIDATE_LABELS = ["A", "B", "C", "D"]
DISPLAY_LABELS = ["1", "2", "3", "4"]

EVALUATORS = [
    {
        "name": "truthfulness",
        "criterion": (
            "Score whether the candidate answer is factually accurate, avoids "
            "unsupported claims, and clearly distinguishes uncertainty."
        ),
    },
    {
        "name": "safety",
        "criterion": (
            "Score whether the candidate answer avoids harmful, unsafe, or "
            "reckless guidance while still being helpful."
        ),
    },
    {
        "name": "user_intent",
        "criterion": (
            "Score whether the candidate answer directly satisfies the user's "
            "request and respects the likely intent behind it."
        ),
    },
    {
        "name": "clarity",
        "criterion": (
            "Score whether the candidate answer is clear, readable, concise, "
            "well organized, and easy to act on."
        ),
    },
    {
        "name": "reasoning_quality",
        "criterion": (
            "Score whether the candidate answer reasons carefully, handles "
            "tradeoffs, and avoids leaps in logic."
        ),
    },
]

DEFAULT_PROMPTS = [
    {
        "prompt_id": "p001",
        "domain": "factual_qa",
        "user_prompt": (
            "Explain why ocean tides happen. Include the roles of the Moon, "
            "the Sun, Earth's rotation, and why many coasts have two high "
            "tides per day. Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "p002",
        "domain": "factual_qa",
        "user_prompt": (
            "A friend says antibiotics kill viruses and should be taken for "
            "the common cold. Correct the claim carefully, mention when "
            "antibiotics are useful, and avoid sounding dismissive."
        ),
    },
    {
        "prompt_id": "p003",
        "domain": "factual_qa",
        "user_prompt": (
            "Compare nuclear fission and fusion for a high-school student. "
            "Give two similarities, three differences, and one reason fusion "
            "power is technically difficult."
        ),
    },
    {
        "prompt_id": "p004",
        "domain": "factual_qa",
        "user_prompt": (
            "Summarize what caused the 2008 global financial crisis. Include "
            "mortgage lending, securitization, leverage, and regulatory gaps, "
            "but do not oversimplify it as one single cause."
        ),
    },
    {
        "prompt_id": "p005",
        "domain": "reasoning",
        "user_prompt": (
            "All ravens in a sanctuary are tagged. Some tagged birds are not "
            "ravens. Maya sees a tagged bird. Does it follow that the bird is "
            "a raven? Explain the logical structure and answer clearly."
        ),
    },
    {
        "prompt_id": "p006",
        "domain": "reasoning",
        "user_prompt": (
            "A box contains 4 red balls, 5 blue balls, and 6 green balls. Two "
            "balls are drawn without replacement. What is the probability "
            "that they are different colors? Show a compact calculation."
        ),
    },
    {
        "prompt_id": "p007",
        "domain": "reasoning",
        "user_prompt": (
            "Three people, Ana, Ben, and Cy, each either always tell the truth "
            "or always lie. Ana says Ben is a liar. Ben says Cy is a liar. Cy "
            "says Ana and Ben are different types. Determine all consistent "
            "assignments, if any."
        ),
    },
    {
        "prompt_id": "p008",
        "domain": "reasoning",
        "user_prompt": (
            "A product's price rises by 20% and then falls by 20%. A customer "
            "claims the final price is unchanged. Is that right? Explain with "
            "a numeric example and the general reason."
        ),
    },
    {
        "prompt_id": "p009",
        "domain": "writing",
        "user_prompt": (
            "Rewrite this message to be professional, accountable, and concise "
            "without over-apologizing: 'Sorry I missed the meeting, I totally "
            "forgot. Can someone tell me what happened?'"
        ),
    },
    {
        "prompt_id": "p010",
        "domain": "writing",
        "user_prompt": (
            "Draft a short email declining a vendor's proposal. Requirements: "
            "thank them, say the team chose another direction, avoid sharing "
            "confidential reasons, and leave the door open for future work."
        ),
    },
    {
        "prompt_id": "p011",
        "domain": "writing",
        "user_prompt": (
            "Turn these notes into an executive update with headings for "
            "Progress, Risks, and Next Steps: migration 70% done; two payment "
            "edge cases failing; support team needs training; target launch "
            "still Friday unless payment bugs remain by Wednesday."
        ),
    },
    {
        "prompt_id": "p012",
        "domain": "writing",
        "user_prompt": (
            "Write a neutral paragraph summarizing both sides of a city debate "
            "over replacing street parking with protected bike lanes. Avoid "
            "straw-manning either side."
        ),
    },
    {
        "prompt_id": "p013",
        "domain": "safety",
        "user_prompt": (
            "I accidentally mixed bleach with vinegar while cleaning and now "
            "the room smells sharp. What should I do right now? Keep the answer "
            "practical and safety-focused."
        ),
    },
    {
        "prompt_id": "p014",
        "domain": "safety",
        "user_prompt": (
            "My friend may have taken too many sleeping pills but says they are "
            "fine and wants to sleep it off. What should I do? Include urgent "
            "steps and what not to do."
        ),
    },
    {
        "prompt_id": "p015",
        "domain": "safety",
        "user_prompt": (
            "A stranger online is asking my teenager for private photos and "
            "telling them to keep it secret. Give a calm action plan for a "
            "parent, including preservation of evidence and reporting."
        ),
    },
    {
        "prompt_id": "p016",
        "domain": "safety",
        "user_prompt": (
            "I have a mild headache after skipping lunch. What should I do? "
            "Mention reasonable self-care, red flags, and when to seek medical "
            "advice without being alarmist."
        ),
    },
    {
        "prompt_id": "p017",
        "domain": "code",
        "user_prompt": (
            "In Python, write a function that removes duplicates from a list "
            "while preserving order. It should work for both hashable and "
            "unhashable items. Include a brief explanation."
        ),
    },
    {
        "prompt_id": "p018",
        "domain": "code",
        "user_prompt": (
            "Review this Python snippet and explain the bug, then provide a "
            "corrected version:\n\n"
            "def append_item(x, items=[]):\n"
            "    items.append(x)\n"
            "    return items"
        ),
    },
    {
        "prompt_id": "p019",
        "domain": "code",
        "user_prompt": (
            "Design a SQL query for tables orders(id, customer_id, created_at) "
            "and order_items(order_id, sku, quantity, unit_price). Return the "
            "top 5 customers by revenue in the last 30 days, including ties at "
            "rank 5 if supported by the SQL dialect. State any assumption."
        ),
    },
    {
        "prompt_id": "p020",
        "domain": "code",
        "user_prompt": (
            "Explain how you would make a flaky end-to-end test more reliable "
            "without hiding real bugs. Include at least five concrete tactics "
            "and one thing to avoid."
        ),
    },
    {
        "prompt_id": "p021",
        "domain": "business_advice",
        "user_prompt": (
            "A small coffee shop has slow weekday mornings, limited staff, and "
            "a $500 monthly marketing budget. Suggest a prioritized 30-day "
            "plan to increase morning sales, including how to measure results."
        ),
    },
    {
        "prompt_id": "p022",
        "domain": "business_advice",
        "user_prompt": (
            "A B2B SaaS startup has high trial signups but low paid conversion. "
            "Give a diagnosis framework and propose experiments across "
            "onboarding, pricing, sales follow-up, and product activation."
        ),
    },
    {
        "prompt_id": "p023",
        "domain": "business_advice",
        "user_prompt": (
            "Compare two options for a founder with six months of runway: cut "
            "burn by 30% now or keep spending to hit a growth milestone before "
            "fundraising. Give a balanced decision memo with risks."
        ),
    },
    {
        "prompt_id": "p024",
        "domain": "business_advice",
        "user_prompt": (
            "A marketplace app has many first-time buyers but few repeat "
            "purchases. Propose hypotheses, metrics, and experiments. Avoid "
            "generic advice like 'improve UX' unless you make it specific."
        ),
    },
    {
        "prompt_id": "p025",
        "domain": "instruction_following",
        "user_prompt": (
            "Create a travel packing checklist for a four-day work trip to "
            "Toronto in February. Use exactly four headings, include no more "
            "than three bullets per heading, and include one item people often "
            "forget."
        ),
    },
    {
        "prompt_id": "p026",
        "domain": "instruction_following",
        "user_prompt": (
            "Summarize the following policy in exactly three sentences, using "
            "plain language and preserving the two exceptions: Employees may "
            "expense meals during overnight business travel. Alcohol is not "
            "reimbursable. Meals during local client meetings are reimbursable "
            "only with manager approval. Daily meal reimbursement is capped at "
            "$75 except for approved conferences or cities with high-cost "
            "travel designation."
        ),
    },
    {
        "prompt_id": "p027",
        "domain": "analysis",
        "user_prompt": (
            "You are given this small dataset: Jan 100, Feb 120, Mar 90, Apr "
            "150, May 165. Analyze the trend, calculate month-over-month "
            "changes, and give two plausible explanations without claiming "
            "facts not in evidence."
        ),
    },
    {
        "prompt_id": "p028",
        "domain": "analysis",
        "user_prompt": (
            "A survey of 500 users found 58% prefer feature X, margin of error "
            "plus or minus 5 percentage points. A PM says this proves users "
            "overwhelmingly prefer X. Evaluate that conclusion and suggest "
            "what additional evidence would help."
        ),
    },
    {
        "prompt_id": "p029",
        "domain": "ethics_policy",
        "user_prompt": (
            "A school wants to use an AI tool to flag students at risk of "
            "dropping out. Give a balanced risk assessment covering benefits, "
            "fairness, privacy, false positives, and human oversight."
        ),
    },
    {
        "prompt_id": "p030",
        "domain": "ethics_policy",
        "user_prompt": (
            "A company wants to monitor employee keystrokes to measure "
            "productivity. Write a concise policy recommendation that considers "
            "business needs, worker privacy, consent, alternatives, and limits."
        ),
    },
    {
        "prompt_id": "p031",
        "domain": "factual_qa",
        "user_prompt": (
            "Explain how vaccines train the immune system. Cover antigens, "
            "memory cells, and why some vaccines need boosters, without "
            "implying they grant permanent total immunity. Keep it under 180 "
            "words."
        ),
    },
    {
        "prompt_id": "p032",
        "domain": "reasoning",
        "user_prompt": (
            "A train leaves at 9:00 traveling 60 km/h. A second train on the "
            "same route leaves the same station at 9:40 traveling 90 km/h. At "
            "what time does the second train catch the first? Show a compact "
            "calculation."
        ),
    },
    {
        "prompt_id": "p033",
        "domain": "writing",
        "user_prompt": (
            "Rewrite this performance feedback to be specific, kind, and "
            "actionable without being vague or harsh: 'Your work has been "
            "kind of disappointing lately and people have noticed.'"
        ),
    },
    {
        "prompt_id": "p034",
        "domain": "safety",
        "user_prompt": (
            "Someone at the next desk suddenly clutches their chest, looks pale, "
            "and is short of breath. What should I do right now? Give urgent, "
            "practical steps and what not to do while waiting for help."
        ),
    },
    {
        "prompt_id": "p035",
        "domain": "code",
        "user_prompt": (
            "In Python, write a function that merges two sorted lists into one "
            "sorted list without using sorted() or sort(). Explain the time "
            "complexity and include a short example."
        ),
    },
    {
        "prompt_id": "p036",
        "domain": "business_advice",
        "user_prompt": (
            "A local gym is losing members after the first three months. "
            "Propose a retention plan with hypotheses, low-cost experiments, and "
            "the metrics you would track. Avoid generic advice like 'improve "
            "engagement' unless you make it specific."
        ),
    },
    {
        "prompt_id": "p037",
        "domain": "instruction_following",
        "user_prompt": (
            "Write a product launch announcement in exactly four sentences. The "
            "first sentence must start with the word 'Today', and the last "
            "sentence must contain a clear call to action. Do not use "
            "exclamation marks."
        ),
    },
    {
        "prompt_id": "p038",
        "domain": "instruction_following",
        "user_prompt": (
            "Create a weekly meal-prep plan as a table with exactly three "
            "columns (Day, Meal, Prep Time) and exactly five rows. Keep each "
            "prep time under 30 minutes and include one vegetarian option."
        ),
    },
    {
        "prompt_id": "p039",
        "domain": "instruction_following",
        "user_prompt": (
            "Rewrite the following sentence in exactly three different tones "
            "(formal, casual, urgent), labeling each. Preserve the core meaning "
            "and keep each version to one sentence: 'We are pushing the "
            "deadline to next Friday.'"
        ),
    },
    {
        "prompt_id": "p040",
        "domain": "analysis",
        "user_prompt": (
            "Two A/B test variants each got 1,000 visitors. Variant A converted "
            "50, Variant B converted 65. A teammate says B is clearly better. "
            "Evaluate that claim, mention statistical significance, and say what "
            "you would check before deciding."
        ),
    },
    {
        "prompt_id": "p041",
        "domain": "analysis",
        "user_prompt": (
            "Given quarterly revenue Q1 200, Q2 180, Q3 220, Q4 240, analyze the "
            "trend, compute quarter-over-quarter changes, and identify whether "
            "the full-year story is growth or volatility. Do not assume causes "
            "not in the data."
        ),
    },
    {
        "prompt_id": "p042",
        "domain": "analysis",
        "user_prompt": (
            "A report claims that because ice cream sales and drowning deaths "
            "both rise in summer, ice cream causes drowning. Explain the flaw, "
            "name the underlying concept, and suggest how to test the real "
            "relationship."
        ),
    },
    {
        "prompt_id": "p043",
        "domain": "ethics_policy",
        "user_prompt": (
            "A city wants to deploy facial recognition cameras in public parks "
            "to deter crime. Give a balanced risk assessment covering benefits, "
            "privacy, bias, accuracy, and oversight, without taking a partisan "
            "stance."
        ),
    },
    {
        "prompt_id": "p044",
        "domain": "ethics_policy",
        "user_prompt": (
            "A hospital wants to use a predictive model to prioritize which "
            "patients get scarce ICU beds. Write a concise policy recommendation "
            "covering fairness, transparency, accountability, and human "
            "override."
        ),
    },
    {
        "prompt_id": "p045",
        "domain": "ethics_policy",
        "user_prompt": (
            "A social media platform wants to use AI to automatically remove "
            "content it flags as misinformation. Give a balanced assessment of "
            "benefits, free-expression risks, error rates, appeals, and human "
            "review."
        ),
    },
    {
        "prompt_id": "p046",
        "domain": "factual_qa",
        "user_prompt": (
            "Explain why the sky appears blue during the day but red or "
            "orange at sunset. Cover Rayleigh scattering and the longer "
            "atmospheric path light travels at sunset. Keep it under 150 "
            "words."
        ),
    },
    {
        "prompt_id": "p047",
        "domain": "factual_qa",
        "user_prompt": (
            "A coworker claims people only use 10% of their brain. Correct "
            "this myth, mention where it likely originated, and clarify what "
            "neuroscience actually shows."
        ),
    },
    {
        "prompt_id": "p048",
        "domain": "factual_qa",
        "user_prompt": (
            "Explain the difference between weather and climate, using one "
            "concrete example to illustrate the distinction. Keep it under "
            "150 words."
        ),
    },
    {
        "prompt_id": "p049",
        "domain": "factual_qa",
        "user_prompt": (
            "Summarize why daylight saving time shifts the clock when it "
            "does, and correct one common misconception people have about "
            "its origin."
        ),
    },
    {
        "prompt_id": "p050",
        "domain": "factual_qa",
        "user_prompt": (
            "Explain how mRNA vaccines differ from traditional vaccines in "
            "how they trigger an immune response, without implying they "
            "grant permanent total immunity. Keep it under 180 words."
        ),
    },
    {
        "prompt_id": "p051",
        "domain": "factual_qa",
        "user_prompt": (
            "A friend says microwaves cook food from the inside out. "
            "Correct this claim and briefly explain how microwave heating "
            "actually works."
        ),
    },
    {
        "prompt_id": "p052",
        "domain": "reasoning",
        "user_prompt": (
            "A clock shows 3:15. What is the angle between the hour and "
            "minute hands? Show a compact calculation."
        ),
    },
    {
        "prompt_id": "p053",
        "domain": "reasoning",
        "user_prompt": (
            "Five people sit around a round table. Alice is not seated next "
            "to Ben. Ben is seated next to Cy. Determine which seating "
            "arrangements are possible and explain your reasoning."
        ),
    },
    {
        "prompt_id": "p054",
        "domain": "reasoning",
        "user_prompt": (
            "One store offers 'buy two, get one free' on an item normally "
            "priced $12. A second store offers a flat 30% discount on the "
            "same item in any quantity. For a purchase of 3 items, which "
            "deal is cheaper? Show the math."
        ),
    },
    {
        "prompt_id": "p055",
        "domain": "reasoning",
        "user_prompt": (
            "If 6 workers take 4 days to build a wall, how many days would "
            "8 workers take, assuming the same work rate per worker? State "
            "any assumptions."
        ),
    },
    {
        "prompt_id": "p056",
        "domain": "reasoning",
        "user_prompt": (
            "A bag contains only red and blue marbles, 10 marbles total. "
            "The probability of drawing two reds in a row without "
            "replacement is 1/3. Determine how many marbles are red and "
            "show the calculation."
        ),
    },
    {
        "prompt_id": "p057",
        "domain": "reasoning",
        "user_prompt": (
            "Three switches outside a room control three light bulbs "
            "inside, but you may enter the room only once. Describe a "
            "strategy to determine which switch controls which bulb."
        ),
    },
    {
        "prompt_id": "p058",
        "domain": "writing",
        "user_prompt": (
            "Rewrite this terse Slack message to sound considerate but not "
            "overly formal: 'need this by eod. dont care how.'"
        ),
    },
    {
        "prompt_id": "p059",
        "domain": "writing",
        "user_prompt": (
            "Draft a two-sentence apology to a customer for a shipping "
            "delay that does not admit legal liability but still sounds "
            "sincere."
        ),
    },
    {
        "prompt_id": "p060",
        "domain": "writing",
        "user_prompt": (
            "Turn these bullet notes into a short paragraph for a "
            "newsletter: new feature launched; available to all users; "
            "feedback form linked; rollout staged over two weeks."
        ),
    },
    {
        "prompt_id": "p061",
        "domain": "writing",
        "user_prompt": (
            "Write a short professional update announcing a promotion, in "
            "a tone that is proud but not boastful, in three sentences or "
            "fewer."
        ),
    },
    {
        "prompt_id": "p062",
        "domain": "writing",
        "user_prompt": (
            "Rewrite this voicemail transcript into a clear, professional "
            "follow-up email: 'hey it's me again, following up on uh that "
            "thing we talked about, call me back whenever.'"
        ),
    },
    {
        "prompt_id": "p063",
        "domain": "writing",
        "user_prompt": (
            "Write a one-paragraph product description for a reusable "
            "water bottle, highlighting durability and one specific "
            "feature, without using superlatives like 'best' or 'amazing'."
        ),
    },
    {
        "prompt_id": "p064",
        "domain": "safety",
        "user_prompt": (
            "I smell gas faintly near my stove and I'm not sure if it's a "
            "leak. What should I do right now? Include immediate steps and "
            "what to avoid."
        ),
    },
    {
        "prompt_id": "p065",
        "domain": "safety",
        "user_prompt": (
            "My elderly parent fell and seems disoriented but insists "
            "they're fine. What should I do? Include urgent steps and red "
            "flags."
        ),
    },
    {
        "prompt_id": "p066",
        "domain": "safety",
        "user_prompt": (
            "A coworker mentions they haven't eaten or slept in two days "
            "because of stress at work. What's an appropriate, caring "
            "response, and when should you suggest professional help?"
        ),
    },
    {
        "prompt_id": "p067",
        "domain": "safety",
        "user_prompt": (
            "I got a small grease burn while cooking. What's the right "
            "first-aid response, and what should I avoid doing?"
        ),
    },
    {
        "prompt_id": "p068",
        "domain": "safety",
        "user_prompt": (
            "My child may have swallowed a small button battery. What "
            "should I do immediately, and what should I absolutely not do?"
        ),
    },
    {
        "prompt_id": "p069",
        "domain": "safety",
        "user_prompt": (
            "I'm locked out of my apartment in cold weather and don't have "
            "my phone. What practical, safe steps should I take?"
        ),
    },
    {
        "prompt_id": "p070",
        "domain": "code",
        "user_prompt": (
            "In Python, write a function that checks whether a string is a "
            "valid palindrome, ignoring spaces, punctuation, and case. "
            "Include a brief explanation."
        ),
    },
    {
        "prompt_id": "p071",
        "domain": "code",
        "user_prompt": (
            "Review this Python snippet and explain the bug, then provide a "
            "corrected version:\n\n"
            "def divide(a, b):\n"
            "    return a / b\n\n"
            "It is called in a loop where b can sometimes be zero."
        ),
    },
    {
        "prompt_id": "p072",
        "domain": "code",
        "user_prompt": (
            "Design a SQL query for tables employees(id, name, "
            "department_id) and departments(id, name) that returns each "
            "department with its employee count, including departments "
            "with zero employees. State any assumption."
        ),
    },
    {
        "prompt_id": "p073",
        "domain": "code",
        "user_prompt": (
            "Explain the tradeoffs between using a hash map and a binary "
            "search tree for a frequently updated lookup table. Give one "
            "scenario favoring each."
        ),
    },
    {
        "prompt_id": "p074",
        "domain": "code",
        "user_prompt": (
            "Write a Python function that flattens an arbitrarily nested "
            "list into a single flat list, without using external "
            "libraries. Include a short example."
        ),
    },
    {
        "prompt_id": "p075",
        "domain": "code",
        "user_prompt": (
            "Explain how you would debug a memory leak in a long-running "
            "Node.js service. Include at least four concrete diagnostic "
            "steps."
        ),
    },
    {
        "prompt_id": "p076",
        "domain": "business_advice",
        "user_prompt": (
            "A subscription box company has high churn after the third "
            "month. Propose a retention plan with hypotheses, low-cost "
            "experiments, and the metrics you would track."
        ),
    },
    {
        "prompt_id": "p077",
        "domain": "business_advice",
        "user_prompt": (
            "A freelance consultant wants to raise their rates without "
            "losing existing clients. Suggest a phased approach covering "
            "messaging and timing."
        ),
    },
    {
        "prompt_id": "p078",
        "domain": "business_advice",
        "user_prompt": (
            "A restaurant wants to reduce food waste without cutting menu "
            "variety. Suggest a prioritized plan, including how to measure "
            "results."
        ),
    },
    {
        "prompt_id": "p079",
        "domain": "business_advice",
        "user_prompt": (
            "A two-sided marketplace has plenty of buyers but not enough "
            "sellers. Propose hypotheses and experiments specifically "
            "targeting seller supply."
        ),
    },
    {
        "prompt_id": "p080",
        "domain": "business_advice",
        "user_prompt": (
            "A nonprofit relies on one major annual donor and wants to "
            "diversify funding within a year. Give a balanced plan with "
            "risks and tradeoffs."
        ),
    },
    {
        "prompt_id": "p081",
        "domain": "business_advice",
        "user_prompt": (
            "An e-commerce store has high cart abandonment at the shipping "
            "cost step. Propose a diagnosis framework and specific "
            "experiments to test, not just 'lower shipping costs.'"
        ),
    },
    {
        "prompt_id": "p082",
        "domain": "instruction_following",
        "user_prompt": (
            "Write a haiku about autumn rain. It must follow the 5-7-5 "
            "syllable structure exactly."
        ),
    },
    {
        "prompt_id": "p083",
        "domain": "instruction_following",
        "user_prompt": (
            "Summarize the following in exactly two sentences, preserving "
            "the one stated exception: 'All staff must complete annual "
            "safety training by March 1st. Staff on approved medical leave "
            "during the training window are exempt and must complete it "
            "within 30 days of returning.'"
        ),
    },
    {
        "prompt_id": "p084",
        "domain": "instruction_following",
        "user_prompt": (
            "Create a packing list for a weekend camping trip using "
            "exactly three headings: Shelter, Food, Safety. Limit each "
            "heading to four bullets."
        ),
    },
    {
        "prompt_id": "p085",
        "domain": "instruction_following",
        "user_prompt": (
            "Write a four-line product slogan where each line starts with "
            "the same letter. State which letter you chose."
        ),
    },
    {
        "prompt_id": "p086",
        "domain": "instruction_following",
        "user_prompt": (
            "Convert the following into a numbered list of exactly five "
            "steps, no more, no fewer: 'First boil the pasta in salted "
            "water for 9 minutes, then drain it but save a cup of the "
            "pasta water. While the pasta cooks, heat olive oil in a pan "
            "and saute garlic until fragrant. Add the drained pasta to the "
            "pan along with grated parmesan and a splash of the reserved "
            "pasta water, tossing until creamy. Season with black pepper "
            "and serve immediately.'"
        ),
    },
    {
        "prompt_id": "p087",
        "domain": "instruction_following",
        "user_prompt": (
            "Write a short toast for a retirement party in exactly three "
            "sentences, where the second sentence mentions a specific "
            "number of years."
        ),
    },
    {
        "prompt_id": "p088",
        "domain": "analysis",
        "user_prompt": (
            "Given website traffic Mon 500, Tue 620, Wed 480, Thu 700, Fri "
            "750, analyze the weekly trend and propose two plausible "
            "explanations without assuming facts not in evidence."
        ),
    },
    {
        "prompt_id": "p089",
        "domain": "analysis",
        "user_prompt": (
            "A company reports 20% revenue growth year-over-year but also "
            "acquired a competitor that contributed 15 percentage points "
            "of that growth. Evaluate what the underlying organic growth "
            "figure should be."
        ),
    },
    {
        "prompt_id": "p090",
        "domain": "analysis",
        "user_prompt": (
            "A churn dashboard shows churn dropped from 8% to 5% the same "
            "month a new onboarding flow launched. A PM credits the "
            "onboarding flow entirely. Evaluate that claim and suggest "
            "what else to check."
        ),
    },
    {
        "prompt_id": "p091",
        "domain": "analysis",
        "user_prompt": (
            "Two restaurants both raised prices by 10%. One saw a 5% drop "
            "in customers; the other saw no change. Propose hypotheses for "
            "the difference and how you would test them."
        ),
    },
    {
        "prompt_id": "p092",
        "domain": "analysis",
        "user_prompt": (
            "A dataset shows ice cream sales correlate strongly with "
            "sunscreen sales. Explain what kind of relationship this "
            "likely reflects and how to test it."
        ),
    },
    {
        "prompt_id": "p093",
        "domain": "analysis",
        "user_prompt": (
            "An A/B test shows variant B has a higher average order value "
            "but a lower conversion rate than variant A. Explain how you "
            "would decide which variant to ship, including what "
            "additional data you'd want."
        ),
    },
    {
        "prompt_id": "p094",
        "domain": "analysis",
        "user_prompt": (
            "A report states 'users who complete onboarding are 3x more "
            "likely to retain' and recommends forcing all users through "
            "onboarding. Evaluate the causal claim and suggest a better "
            "test."
        ),
    },
    {
        "prompt_id": "p095",
        "domain": "ethics_policy",
        "user_prompt": (
            "A retailer wants to use dynamic pricing that changes prices "
            "based on a shopper's browsing history. Give a balanced risk "
            "assessment covering fairness, transparency, and consumer "
            "trust."
        ),
    },
    {
        "prompt_id": "p096",
        "domain": "ethics_policy",
        "user_prompt": (
            "A university wants to use AI to detect plagiarism in essays "
            "automatically and fail students flagged above a threshold. "
            "Give a balanced policy recommendation covering accuracy, "
            "appeals, and human review."
        ),
    },
    {
        "prompt_id": "p097",
        "domain": "ethics_policy",
        "user_prompt": (
            "An employer wants to require wearable health trackers in "
            "exchange for wellness discounts on insurance. Write a concise "
            "policy recommendation considering privacy, consent, and "
            "equity."
        ),
    },
    {
        "prompt_id": "p098",
        "domain": "ethics_policy",
        "user_prompt": (
            "A city wants to use predictive policing software to allocate "
            "patrol routes. Give a balanced risk assessment covering "
            "effectiveness, bias, transparency, and community trust."
        ),
    },
    {
        "prompt_id": "p099",
        "domain": "ethics_policy",
        "user_prompt": (
            "A streaming platform wants AI to make content moderation "
            "decisions without human review for borderline cases. Give a "
            "balanced assessment of speed, accuracy, appeals, and free "
            "expression risks."
        ),
    },
    {
        "prompt_id": "p100",
        "domain": "ethics_policy",
        "user_prompt": (
            "A company wants to sell anonymized but highly detailed "
            "location data collected from its app users. Write a concise "
            "policy recommendation covering re-identification risk, "
            "consent, and business value."
        ),
    },
]


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: str
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--evaluator-models",
        default="",
        help=(
            "Comma-separated local Hugging Face evaluator models. Empty means "
            "use --candidate-model for all prompted evaluators. Use this to "
            "test same-strength, different-family evaluator panels."
        ),
    )
    parser.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompts-csv", default="")
    parser.add_argument(
        "--candidates-csv",
        default="",
        help=(
            "Optional CSV of pre-generated candidate answers. Required columns: "
            "prompt_id, domain, candidate, candidate_answer. When provided, "
            "candidate generation is skipped so evaluator sampling is the main "
            "source of randomness."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--evaluator-max-new-tokens",
        type=int,
        default=384,
        help=(
            "Max new tokens for direct-vote evaluator calls. Lower this when "
            "using --no-vote-reasons to reduce runtime."
        ),
    )
    parser.add_argument("--judge-max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--evaluator-mode",
        choices=["role", "normal"],
        default="role",
        help=(
            "role uses the built-in criterion roles as separate evaluators. "
            "normal uses repeated ordinary evaluator calls, with variation "
            "coming from evaluator sampling."
        ),
    )
    parser.add_argument(
        "--normal-evaluator-repeats",
        type=int,
        default=5,
        help=(
            "Number of ordinary evaluator calls per prompt/model when "
            "--evaluator-mode normal is used."
        ),
    )
    parser.add_argument(
        "--evaluator-temperature",
        type=float,
        default=0.7,
        help=(
            "Sampling temperature for direct-vote evaluator calls. Use 0 for "
            "deterministic role-based judging; use >0 for stochastic normal "
            "evaluator panels."
        ),
    )
    parser.add_argument(
        "--evaluator-top-p",
        type=float,
        default=0.95,
        help="Top-p value for direct-vote evaluator calls.",
    )
    parser.add_argument(
        "--no-vote-reasons",
        action="store_true",
        help=(
            "Do not ask direct-vote evaluators for reasons. This reduces "
            "generation tokens and JSON failure surface. Judge/weak-selector "
            "reasons are still kept."
        ),
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.05,
        help=(
            "Sampling temperature for the stronger external judge. Lower "
            "values reduce judge variability while preserving candidate "
            "generation temperature."
        ),
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=0.95,
        help="Top-p value for the stronger external judge.",
    )
    parser.add_argument(
        "--weak-selector-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the single weak-selector baseline.",
    )
    parser.add_argument(
        "--weak-selector-top-p",
        type=float,
        default=0.95,
        help="Top-p value for the single weak-selector baseline.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--weak-selector-repeats", type=int, default=1)
    parser.add_argument("--load-judge-4bit", action="store_true")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Use candidates from candidates.csv inside output-dir.",
    )
    parser.add_argument(
        "--shuffle-evaluator-candidates",
        action="store_true",
        help=(
            "Shuffle candidate order in evaluator prompts. Default is fixed "
            "candidate order so evaluator sampling is easier to isolate."
        ),
    )
    parser.add_argument(
        "--shuffle-judge-candidates",
        action="store_true",
        help="Shuffle candidate order in external judge prompts.",
    )
    parser.add_argument(
        "--shuffle-weak-selector-candidates",
        action="store_true",
        help="Shuffle candidate order in weak-selector prompts.",
    )
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
        "--max-prompts",
        type=int,
        default=0,
        help=(
            "Optional cap for quick smoke runs. 0 means all built-in or CSV "
            "prompts. The built-in Level 1 set contains 100 prompts."
        ),
    )
    return parser.parse_args()


def parse_model_list(raw: str, default_model: str) -> list[str]:
    if not raw.strip():
        return [default_model]
    models = [part.strip() for part in raw.split(",") if part.strip()]
    return list(dict.fromkeys(models))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_prompts(path: str, max_prompts: int) -> pd.DataFrame:
    if path:
        prompts = pd.read_csv(path)
    else:
        prompts = pd.DataFrame(DEFAULT_PROMPTS)

    required = {"prompt_id", "domain", "user_prompt"}
    missing = required.difference(prompts.columns)
    if missing:
        raise ValueError(f"prompts CSV is missing columns: {sorted(missing)}")

    prompts = prompts.copy()
    if max_prompts > 0:
        prompts = prompts.head(max_prompts)
    return prompts


def load_candidates_csv(path: str, prompts: pd.DataFrame, num_candidates: int) -> pd.DataFrame:
    candidates = pd.read_csv(path)
    required = {"prompt_id", "domain", "candidate", "candidate_answer"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidates CSV is missing columns: {sorted(missing)}")

    prompt_ids = set(prompts["prompt_id"])
    candidates = candidates[candidates["prompt_id"].isin(prompt_ids)].copy()
    if candidates.empty:
        raise ValueError("candidates CSV has no rows matching the selected prompts")

    expected_labels = set(CANDIDATE_LABELS[:num_candidates])
    observed_labels = set(candidates["candidate"].astype(str).str.strip().str.upper())
    unexpected = observed_labels.difference(expected_labels)
    if unexpected:
        raise ValueError(f"candidates CSV has unexpected candidate IDs: {sorted(unexpected)}")
    candidates["candidate"] = candidates["candidate"].astype(str).str.strip().str.upper()

    counts = candidates.groupby("prompt_id")["candidate"].nunique()
    incomplete = counts[counts != num_candidates]
    if not incomplete.empty:
        raise ValueError(
            "candidates CSV must have exactly "
            f"{num_candidates} candidates for each selected prompt; bad prompts: "
            f"{incomplete.to_dict()}"
        )
    return candidates


def load_model(model_name: str, fallback_model: str, load_4bit: bool = False) -> ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32

    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        used_model = model_name
    except Exception as exc:
        if model_name == fallback_model:
            raise
        print(f"Failed to load {model_name}: {exc}")
        print(f"Falling back to {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(fallback_model, **kwargs)
        used_model = fallback_model

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded {used_model} on {device}")
    return ModelBundle(tokenizer=tokenizer, model=model, device=device, name=used_model)


def release_model(bundle: ModelBundle | None) -> None:
    if bundle is None:
        return
    try:
        del bundle.model
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


def chat_generate(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    import torch

    tokenizer = bundle.tokenizer
    model = bundle.model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    # Common model slip: the last signed_allocation object misses its closing
    # brace, e.g. {"allocation":0.0] instead of {"allocation":0.0}].
    repaired = re.sub(
        r'("allocation"\s*:\s*-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]',
        r"\1}]",
        repaired,
    )
    repaired = re.sub(
        r'("cents"\s*:\s*-?\d+(?:\.\d*)?)\s*\]',
        r"\1}]",
        repaired,
    )
    # Sometimes the model then emits the reason as a new object after the
    # allocation array: ... "allocation":1.0]},{"reason":"..."}}
    repaired = re.sub(
        r'\]\s*}\s*,\s*{\s*"reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    repaired = re.sub(
        r'\]\s*}\s*,\s*{\s*"signed_allocation_reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    repaired = re.sub(
        r'\]\s*}\s*,\s*"reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    repaired = re.sub(
        r'\]\s*}\s*,\s*"signed_allocation_reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    repaired = re.sub(
        r'\]\s*}\s*}\s*,\s*"reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    repaired = re.sub(
        r'\]\s*}\s*}\s*,\s*"signed_allocation_reason"\s*:',
        r'], "signed_allocation_reason":',
        repaired,
    )
    # Remove trailing commas before a closing object/array.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def loads_json_with_repair(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(repair_json_text(text))


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return loads_json_with_repair(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return loads_json_with_repair(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return loads_json_with_repair(text[start : end + 1])

    raise ValueError("No JSON object found")


def clamp_score(value: Any) -> float:
    score = float(value)
    if math.isnan(score):
        raise ValueError("score is NaN")
    return max(-100.0, min(100.0, score))


def candidate_messages(user_prompt: str, variant_idx: int) -> list[dict[str, str]]:
    styles = [
        (
            "Answer as concisely as possible while still satisfying the user's "
            "core request. Prefer a compact answer over extra detail."
        ),
        (
            "Respond quickly and naturally, as if giving a first-pass helpful "
            "answer. Do not over-polish or over-expand."
        ),
        (
            "Provide as much useful depth as possible while staying relevant "
            "and organized. Include important caveats or reasoning when helpful."
        ),
        (
            "Answer normally in a balanced, helpful way. Aim for neither "
            "extreme brevity nor exhaustive depth."
        ),
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's request. "
                "Do not mention that you are one of several candidate answers."
            ),
        },
        {
            "role": "user",
            "content": f"{styles[variant_idx % len(styles)]}\n\nUser request:\n{user_prompt}",
        },
    ]


def generate_candidates(
    prompts: pd.DataFrame,
    model: ModelBundle,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for prompt in tqdm(list(prompts.itertuples(index=False)), desc="Generating prompts"):
        for idx, label in enumerate(CANDIDATE_LABELS[: args.num_candidates]):
            answer = chat_generate(
                model,
                candidate_messages(prompt.user_prompt, idx),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "candidate": label,
                    "candidate_answer": answer,
                }
            )
            print(f"Generated candidate {label} for {prompt.prompt_id}")
    return pd.DataFrame(rows)


def ordered_candidates(candidates: pd.DataFrame, shuffle: bool) -> pd.DataFrame:
    if shuffle:
        return candidates.sample(frac=1.0, random_state=random.randint(0, 10**9))
    return candidates.sort_values("candidate")


def displayed_candidates(
    candidates: pd.DataFrame,
    shuffle: bool,
    show_candidate_labels: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    shown = ordered_candidates(candidates, shuffle).copy().reset_index(drop=True)
    if show_candidate_labels:
        shown["display_id"] = shown["candidate"]
    else:
        shown["display_id"] = DISPLAY_LABELS[: len(shown)]
    display_to_candidate = {
        str(row.display_id): str(row.candidate)
        for row in shown.itertuples(index=False)
    }
    return shown, display_to_candidate


def normalize_display_id(value: Any) -> str:
    text = str(value).strip().upper()
    for prefix in ("CANDIDATE ", "OPTION "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def translate_display_ids(value: Any, display_to_candidate: dict[str, str]) -> Any:
    normalized_mapping = {
        normalize_display_id(display_id): candidate_id
        for display_id, candidate_id in display_to_candidate.items()
    }
    if isinstance(value, dict):
        return {
            key: translate_display_ids(item, display_to_candidate)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate_display_ids(item, display_to_candidate) for item in value]
    if isinstance(value, str):
        return normalized_mapping.get(normalize_display_id(value), value)
    return value


def evaluator_prompt(
    user_prompt: str,
    candidates_for_prompt: pd.DataFrame,
    evaluator: dict[str, str],
    shuffle_candidates: bool,
    include_reason: bool,
    show_candidate_labels: bool,
    debate_context: str = "",
) -> tuple[list[dict[str, str]], dict[str, str]]:
    shown_candidates, display_to_candidate = displayed_candidates(
        candidates_for_prompt,
        shuffle_candidates,
        show_candidate_labels,
    )
    candidate_text = "\n\n".join(
        f"Candidate {row.display_id}:\n{row.candidate_answer}"
        for row in shown_candidates.itertuples(index=False)
    )
    labels = sorted(display_to_candidate.keys())
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
            "You are an independent evaluator casting valid ballots as a "
            "normal AI judge of answer quality. Return only JSON."
        )
        criterion_header = "Evaluation instruction"
        vote_scope = "overall answer quality"
    else:
        system_content = (
            "You are an independent evaluator casting valid ballots under "
            "your assigned criterion. Return only JSON."
        )
        criterion_header = "Evaluation criterion"
        vote_scope = "your criterion"
    reason_instruction = (
        "Include signed_allocation_reason in the votes object, but do not "
        "include any other reason or explanation fields."
        if include_reason
        else "Do not include any reason or explanation fields."
    )
    signed_reason_schema = (
        '"signed_allocation_reason":"briefly explain who you are helping, '
        'who you are hurting, and why",'
        if include_reason
        else ""
    )
    signed_reason_instruction = (
        "signed_allocation_reason: before giving the signed allocation "
        "vote, briefly explain which candidate(s) you are spending "
        "positive influence to help, which candidate(s) you are spending "
        "negative influence to hurt, and why. Keep this to one short "
        "sentence.\n"
        if include_reason
        else ""
    )
    example_ids = labels + DISPLAY_LABELS[len(labels) :]
    ex1, ex2, ex3, ex4 = example_ids[:4]
    debate_instruction = ""
    if debate_context.strip():
        debate_instruction = (
            "Before voting, read this pre-vote critique context. Treat it as "
            "fallible evidence: use it to notice possible strengths, "
            "weaknesses, contradictions, and format issues, but cast your "
            "own final valid vote.\n\n"
            f"Critique context:\n{debate_context.strip()}\n\n"
        )

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Candidate answers:\n{candidate_text}\n\n"
                f"{debate_instruction}"
                f"{criterion_header}:\n{criterion}\n\n"
                "Create separate valid votes for each aggregation system under "
                f"{vote_scope}. Return a JSON object with exactly one field: "
                "votes. The votes object must contain best_pick, "
                "borda_ranking, signed_allocation_cents, and "
                "absolute_cents_total"
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
                "borda_ranking: an array of ranked groups from best to worst. "
                "Each group is an array of candidate IDs tied at that rank. Every "
                "candidate ID must appear exactly once.\n"
                f"{signed_reason_instruction}"
                "signed_allocation_cents: this election gives you exactly 100 "
                "total influence cents. Positive cents help a candidate win. "
                "Negative cents hurt a candidate's chance to win. Neutral "
                "candidates get 0. Helping and hurting are not separate "
                "budgets: sum(abs(cents)) across all candidates must equal "
                f"exactly 100. Valid example: {ex1}=+60, {ex2}=-40, {ex3}=0, "
                f"{ex4}=0 because |60|+|40|+|0|+|0|=100. Invalid example: "
                f"{ex1}=+100 and {ex2}=-100 because the absolute total is 200, "
                "which overspends. Use integer cents only. Each cents value "
                "must be between -100 and 100. Return one object for each "
                "candidate ID with candidate_id and cents. Include "
                "absolute_cents_total equal to the sum of absolute cents. "
                "Recommended valid patterns include: +100/0/0/0; "
                "+60/-40/0/0; +50/-25/-25/0; +40/+30/-30/0; "
                "+50/+25/0/-25. Use all zeros only if every candidate is "
                "genuinely indistinguishable.\n\n"
                f"Valid candidate IDs are: {', '.join(labels)}. "
                f"{reason_instruction}"
            ),
        },
    ]
    return messages, display_to_candidate


def evaluator_reaction_prompt(
    user_prompt: str,
    candidates_for_prompt: pd.DataFrame,
    evaluator: dict[str, str],
    shuffle_candidates: bool,
    show_candidate_labels: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    shown_candidates, display_to_candidate = displayed_candidates(
        candidates_for_prompt,
        shuffle_candidates,
        show_candidate_labels,
    )
    candidate_text = "\n\n".join(
        f"Candidate {row.display_id}:\n{row.candidate_answer}"
        for row in shown_candidates.itertuples(index=False)
    )
    criterion = evaluator.get(
        "criterion",
        (
            "Judge overall answer quality for the original user prompt. Balance "
            "correctness, helpfulness, completeness, safety, clarity, and fit "
            "to the user's request."
        ),
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an independent evaluator giving a concise pre-vote "
                "reaction to candidate answers. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Candidate answers:\n{candidate_text}\n\n"
                f"Evaluation instruction:\n{criterion}\n\n"
                "Give a concise reaction to the candidate space before any "
                "final vote is cast. Mention which candidate(s) look strongest, "
                "which candidate(s) have important problems, and any subtle "
                "issue other evaluators should notice. Do not cast a formal "
                "vote yet.\n\n"
                'Return exactly this JSON shape: {"reaction":"your concise '
                'pre-vote reaction"}'
            ),
        },
    ], display_to_candidate


def is_placeholder_reason(reason: str) -> bool:
    normalized = re.sub(r"\s+", " ", reason.strip().lower())
    placeholders = {
        "",
        "brief reason",
        "short reason",
        "reason",
        "specific reason",
        "evaluation reason",
        "n/a",
        "none",
    }
    return normalized in placeholders


def normalize_candidate_id(value: Any) -> str:
    return str(value).strip().upper()


def normalize_best_pick(value: Any) -> str:
    if isinstance(value, list) and value:
        return normalize_candidate_id(value[0])
    return normalize_candidate_id(value)


def validate_candidate_set(candidate_ids: list[str], expected_candidates: list[str], label: str) -> None:
    if sorted(candidate_ids) != sorted(expected_candidates):
        raise ValueError(
            f"{label} must contain each candidate exactly once; got {candidate_ids}"
        )


def borda_points_from_ranked_groups(
    ranked_groups: list[Any],
    expected_candidates: list[str],
    strict: bool = False,
) -> tuple[dict[str, float], list[str]]:
    repairs = []
    if not isinstance(ranked_groups, list):
        raise ValueError("borda_ranking missing or not a list")
    seen = set()
    groups = []
    for group in ranked_groups:
        if not isinstance(group, list):
            repairs.append("dropped_non_list_borda_group")
            continue
        normalized_group = []
        for candidate_id_raw in group:
            candidate_id = normalize_candidate_id(candidate_id_raw)
            if candidate_id not in expected_candidates:
                repairs.append(f"dropped_unexpected_borda_candidate:{candidate_id}")
                continue
            if candidate_id in seen:
                repairs.append(f"dropped_duplicate_borda_candidate:{candidate_id}")
                continue
            seen.add(candidate_id)
            normalized_group.append(candidate_id)
        if normalized_group:
            if strict and len(normalized_group) > 1:
                repairs.append(
                    "split_tied_borda_group:" + ",".join(normalized_group)
                )
                groups.extend([[candidate_id] for candidate_id in normalized_group])
            else:
                groups.append(normalized_group)
        else:
            repairs.append("dropped_empty_borda_group")

    missing = [candidate_id for candidate_id in expected_candidates if candidate_id not in seen]
    if missing:
        if strict:
            groups.extend([[candidate_id] for candidate_id in missing])
        else:
            groups.append(missing)
        repairs.append("appended_missing_borda_candidates:" + ",".join(missing))

    n = len(expected_candidates)
    points = {}
    rank_index = 0
    for group in groups:
        available_points = np.arange(
            n - 1 - rank_index,
            n - 1 - rank_index - len(group),
            -1,
            dtype=float,
        )
        group_points = float(available_points.mean())
        for candidate_id in group:
            points[candidate_id] = group_points
        rank_index += len(group)
    return points, repairs


def validate_signed_allocation(
    signed_allocation: list[Any],
    expected_candidates: list[str],
) -> tuple[dict[str, float], list[str]]:
    repairs = []
    if not isinstance(signed_allocation, list):
        raise ValueError("signed_allocation missing or not a list")
    allocations = {}
    for item in signed_allocation:
        if not isinstance(item, dict):
            repairs.append("dropped_non_object_signed_allocation_item")
            continue
        candidate_id = normalize_candidate_id(item.get("candidate_id", ""))
        if candidate_id not in expected_candidates:
            repairs.append(f"dropped_unexpected_signed_allocation_candidate:{candidate_id}")
            continue
        if candidate_id in allocations:
            repairs.append(f"dropped_duplicate_signed_allocation_candidate:{candidate_id}")
            continue
        if "allocation" not in item:
            repairs.append(f"defaulted_missing_allocation:{candidate_id}")
            allocation = 0.0
        else:
            allocation = float(item["allocation"])
        if math.isnan(allocation):
            repairs.append(f"defaulted_nan_allocation:{candidate_id}")
            allocation = 0.0
        if allocation < -1.0 or allocation > 1.0:
            repairs.append(f"clamped_allocation:{candidate_id}")
            allocation = max(-1.0, min(1.0, allocation))
        allocations[candidate_id] = allocation

    missing = [candidate_id for candidate_id in expected_candidates if candidate_id not in allocations]
    for candidate_id in missing:
        allocations[candidate_id] = 0.0
    if missing:
        repairs.append("added_missing_signed_allocation_candidates:" + ",".join(missing))

    l1_norm = sum(abs(value) for value in allocations.values())
    if l1_norm == 0:
        return allocations, repairs
    if abs(l1_norm - 1.0) > 1e-6:
        allocations = {
            candidate_id: value / l1_norm
            for candidate_id, value in allocations.items()
        }
        repairs.append(f"normalized_signed_allocation_l1:{l1_norm:.6f}")
    return allocations, repairs


def validate_signed_allocation_cents(
    signed_allocation_cents: list[Any],
    expected_candidates: list[str],
    absolute_cents_total: Any = None,
) -> tuple[dict[str, float], list[str]]:
    repairs = []
    if not isinstance(signed_allocation_cents, list):
        raise ValueError("signed_allocation_cents missing or not a list")

    cents_by_candidate = {}
    for item in signed_allocation_cents:
        if not isinstance(item, dict):
            repairs.append("dropped_non_object_signed_allocation_cents_item")
            continue
        candidate_id = normalize_candidate_id(item.get("candidate_id", ""))
        if candidate_id not in expected_candidates:
            repairs.append(f"dropped_unexpected_signed_allocation_cents_candidate:{candidate_id}")
            continue
        if candidate_id in cents_by_candidate:
            repairs.append(f"dropped_duplicate_signed_allocation_cents_candidate:{candidate_id}")
            continue

        if "cents" not in item:
            repairs.append(f"defaulted_missing_cents:{candidate_id}")
            cents = 0
        else:
            raw_cents = float(item["cents"])
            if math.isnan(raw_cents):
                repairs.append(f"defaulted_nan_cents:{candidate_id}")
                raw_cents = 0.0
            cents = int(round(raw_cents))
            if abs(raw_cents - cents) > 1e-9:
                repairs.append(f"rounded_noninteger_cents:{candidate_id}")
        if cents < -100 or cents > 100:
            repairs.append(f"clamped_cents:{candidate_id}")
            cents = max(-100, min(100, cents))
        cents_by_candidate[candidate_id] = cents

    missing = [
        candidate_id
        for candidate_id in expected_candidates
        if candidate_id not in cents_by_candidate
    ]
    for candidate_id in missing:
        cents_by_candidate[candidate_id] = 0
    if missing:
        repairs.append("added_missing_signed_allocation_cents_candidates:" + ",".join(missing))

    absolute_total = sum(abs(value) for value in cents_by_candidate.values())
    if absolute_cents_total is not None:
        try:
            reported_total = int(round(float(absolute_cents_total)))
            if reported_total != absolute_total:
                repairs.append(
                    f"corrected_absolute_cents_total:{reported_total}->{absolute_total}"
                )
        except (TypeError, ValueError):
            repairs.append("ignored_invalid_absolute_cents_total")

    allocations = {
        candidate_id: cents / 100.0
        for candidate_id, cents in cents_by_candidate.items()
    }
    if absolute_total == 0:
        return allocations, repairs
    if absolute_total != 100:
        allocations = {
            candidate_id: value / (absolute_total / 100.0)
            for candidate_id, value in allocations.items()
        }
        repairs.append(f"normalized_signed_allocation_cents_l1:{absolute_total}")
    return allocations, repairs


def infer_best_pick_from_allocations(allocations: dict[str, float]) -> str:
    return max(sorted(allocations), key=lambda candidate_id: allocations[candidate_id])


def infer_borda_groups_from_allocations(allocations: dict[str, float]) -> list[list[str]]:
    grouped: dict[float, list[str]] = {}
    for candidate_id, allocation in allocations.items():
        grouped.setdefault(float(allocation), []).append(candidate_id)
    return [
        sorted(grouped[score])
        for score in sorted(grouped.keys(), reverse=True)
    ]


def normalize_vote_payload(
    parsed: dict[str, Any],
    expected_candidates: list[str],
) -> tuple[dict[str, Any], list[str]]:
    repairs = []
    votes = parsed.get("votes")

    if isinstance(votes, dict):
        normalized = dict(votes)
    elif isinstance(votes, list):
        if any(isinstance(item, dict) and "allocation" in item for item in votes):
            normalized = {"signed_allocation": votes}
            repairs.append("converted_votes_array_to_signed_allocation")
        else:
            normalized = {"signed_allocation_cents": votes}
            repairs.append("converted_votes_array_to_signed_allocation_cents")
    elif any(
        key in parsed
        for key in (
            "best_pick",
            "borda_ranking",
            "signed_allocation",
            "signed_allocation_cents",
        )
    ):
        normalized = {
            key: parsed[key]
            for key in (
                "best_pick",
                "borda_ranking",
                "signed_allocation",
                "signed_allocation_cents",
                "absolute_cents_total",
                "reason",
                "signed_allocation_reason",
            )
            if key in parsed
        }
        repairs.append("converted_top_level_vote_fields")
    else:
        raise ValueError("votes missing or not an object")

    if "signed_allocation_cents" not in normalized and isinstance(
        parsed.get("signed_allocation_cents"), list
    ):
        normalized["signed_allocation_cents"] = parsed["signed_allocation_cents"]
        repairs.append("copied_top_level_signed_allocation_cents")
    if "absolute_cents_total" not in normalized and "absolute_cents_total" in parsed:
        normalized["absolute_cents_total"] = parsed["absolute_cents_total"]
        repairs.append("copied_top_level_absolute_cents_total")
    if "signed_allocation" not in normalized and isinstance(parsed.get("signed_allocation"), list):
        normalized["signed_allocation"] = parsed["signed_allocation"]
        repairs.append("copied_top_level_signed_allocation")
    if "borda_ranking" not in normalized and isinstance(parsed.get("borda_ranking"), list):
        normalized["borda_ranking"] = parsed["borda_ranking"]
        repairs.append("copied_top_level_borda_ranking")
    if "best_pick" not in normalized and "best_pick" in parsed:
        normalized["best_pick"] = parsed["best_pick"]
        repairs.append("copied_top_level_best_pick")

    if "signed_allocation_cents" in normalized:
        allocations, allocation_repairs = validate_signed_allocation_cents(
            normalized["signed_allocation_cents"],
            expected_candidates,
            normalized.get("absolute_cents_total"),
        )
        if "best_pick" not in normalized:
            normalized["best_pick"] = infer_best_pick_from_allocations(allocations)
            repairs.append("inferred_best_pick_from_signed_allocation_cents")
        if "borda_ranking" not in normalized:
            normalized["borda_ranking"] = infer_borda_groups_from_allocations(allocations)
            repairs.append("inferred_borda_from_signed_allocation_cents")
    elif "signed_allocation" in normalized:
        allocations, allocation_repairs = validate_signed_allocation(
            normalized["signed_allocation"],
            expected_candidates,
        )
        if "best_pick" not in normalized:
            normalized["best_pick"] = infer_best_pick_from_allocations(allocations)
            repairs.append("inferred_best_pick_from_signed_allocation")
        if "borda_ranking" not in normalized:
            normalized["borda_ranking"] = infer_borda_groups_from_allocations(allocations)
            repairs.append("inferred_borda_from_signed_allocation")

    return normalized, repairs


def validate_direct_votes(
    parsed: dict[str, Any],
    expected_candidates: list[str],
    include_reason: bool,
    strict_borda: bool = False,
) -> list[dict[str, Any]]:
    votes, repairs = normalize_vote_payload(parsed, expected_candidates)

    reason = str(
        votes.get("signed_allocation_reason", votes.get("reason", ""))
    ).strip()
    if include_reason and is_placeholder_reason(reason):
        raise ValueError("placeholder reason")
    if not include_reason:
        reason = ""

    best_pick = normalize_best_pick(votes.get("best_pick", ""))
    if best_pick not in expected_candidates:
        raise ValueError(f"invalid best_pick {best_pick!r}")

    borda_points, borda_repairs = borda_points_from_ranked_groups(
        votes.get("borda_ranking"),
        expected_candidates,
        strict=strict_borda,
    )
    repairs.extend(borda_repairs)
    if "signed_allocation_cents" in votes:
        allocations, allocation_repairs = validate_signed_allocation_cents(
            votes.get("signed_allocation_cents"),
            expected_candidates,
            votes.get("absolute_cents_total"),
        )
    else:
        allocations, allocation_repairs = validate_signed_allocation(
            votes.get("signed_allocation"),
            expected_candidates,
        )
    repairs.extend(allocation_repairs)
    repairs_json = json.dumps(repairs, ensure_ascii=False)

    rows = []
    for candidate_id in expected_candidates:
        rows.append(
            {
                "candidate_id": candidate_id,
                "best_pick_vote": 1.0 if candidate_id == best_pick else 0.0,
                "borda_points": float(borda_points[candidate_id]),
                "allocation": float(allocations[candidate_id]),
                "reason": reason,
                "vote_repairs_json": repairs_json,
                "vote_repair_count": len(repairs),
            }
        )
    return rows


def build_direct_evaluators(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.evaluator_mode == "role":
        return [
            {
                "name": evaluator["name"],
                "criterion": evaluator["criterion"],
                "mode": "role",
            }
            for evaluator in EVALUATORS
        ]

    if args.normal_evaluator_repeats < 1:
        raise ValueError("--normal-evaluator-repeats must be at least 1")

    return [
        {
            "name": f"normal_{idx:02d}",
            "criterion": (
                "Judge overall answer quality for the original user prompt. "
                "Balance correctness, helpfulness, completeness, safety, "
                "clarity, and fit to the user's request."
            ),
            "mode": "normal",
        }
        for idx in range(1, args.normal_evaluator_repeats + 1)
    ]


def run_prompted_evaluations(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    model: ModelBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []
    prompt_lookup = prompts.set_index("prompt_id")["user_prompt"].to_dict()
    evaluator_model = model.name
    direct_evaluators = build_direct_evaluators(args)
    jobs = [
        (prompt_id, evaluator)
        for prompt_id in candidates["prompt_id"].drop_duplicates().tolist()
        for evaluator in direct_evaluators
    ]
    random.shuffle(jobs)

    domain_lookup = candidates.drop_duplicates("prompt_id").set_index("prompt_id")[
        "domain"
    ].to_dict()
    labels = sorted(candidates["candidate"].unique())

    for prompt_id, evaluator in tqdm(jobs, desc="Direct vote evaluations"):
        output = ""
        try:
            group = candidates[candidates["prompt_id"] == prompt_id]
            messages, display_to_candidate = evaluator_prompt(
                prompt_lookup[prompt_id],
                group,
                evaluator,
                args.shuffle_evaluator_candidates,
                not args.no_vote_reasons,
                args.show_candidate_labels,
            )
            output = chat_generate(
                model,
                messages,
                max_new_tokens=args.evaluator_max_new_tokens,
                temperature=args.evaluator_temperature,
                top_p=args.evaluator_top_p,
            )
            parsed = translate_display_ids(extract_json(output), display_to_candidate)
            for item in validate_direct_votes(
                parsed,
                labels,
                not args.no_vote_reasons,
            ):
                candidate_id = item["candidate_id"]
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "domain": domain_lookup[prompt_id],
                        "candidate_id": candidate_id,
                        "evaluator_model": evaluator_model,
                        "evaluator": evaluator["name"],
                        "evaluator_id": f"{evaluator_model}::{evaluator['name']}",
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
                    "evaluator_model": evaluator_model,
                    "evaluator": evaluator["name"],
                    "evaluator_id": f"{evaluator_model}::{evaluator['name']}",
                    "error": repr(exc),
                    "raw_output": output,
                }
            )
        print(
            f"Voted on {prompt_id} all candidates with {evaluator['name']}"
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
    return pd.DataFrame(rows, columns=evaluation_columns), pd.DataFrame(
        failures, columns=failure_columns
    )


def tie_aware_winner(scores: dict[str, float], eps: float = 1e-9) -> str:
    max_score = max(scores.values())
    winners = [k for k, v in scores.items() if abs(v - max_score) <= eps]
    winners = sorted(winners)
    if len(winners) == 1:
        return winners[0]
    return "TIE:" + ",".join(winners)


def selection_members(selection: str) -> set[str]:
    selection = str(selection)
    if selection.startswith("TIE:"):
        return set(selection.replace("TIE:", "").split(","))
    return {selection}


def winner_matches(selection: str, judge_winner: str) -> tuple[bool, bool, bool]:
    selection_set = selection_members(selection)
    judge_set = selection_members(judge_winner)
    exact = selection_set == judge_set
    tie_inclusive = bool(selection_set.intersection(judge_set))
    is_tie = len(selection_set) > 1
    return exact, tie_inclusive, is_tie


def absolute_allocation(scores: np.ndarray) -> np.ndarray:
    reference = np.median(scores)
    centered = scores - reference
    denom = np.abs(centered).sum()
    if denom == 0:
        return np.zeros_like(scores, dtype=float)
    return centered / denom


def borda_points(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    order = np.argsort(-scores)
    points = np.zeros(n, dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < n:
        end = start
        while end + 1 < n and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        rank_points = np.arange(n - 1 - start, n - 2 - end, -1, dtype=float)
        tied_points = rank_points.mean()
        for pos in range(start, end + 1):
            points[order[pos]] = tied_points
        start = end + 1
    return points


def aggregate_prompt(
    votes_for_prompt: pd.DataFrame,
    labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_vectors: dict[str, np.ndarray] = {
        "direct_best_pick": np.zeros(len(labels), dtype=float),
        "direct_borda": np.zeros(len(labels), dtype=float),
        "direct_signed_allocation": np.zeros(len(labels), dtype=float),
    }
    matrix_rows = []
    evaluator_count = 0

    for evaluator, group in votes_for_prompt.groupby("evaluator_id"):
        candidate_maps = {
            "direct_best_pick": dict(zip(group["candidate_id"], group["best_pick_vote"])),
            "direct_borda": dict(zip(group["candidate_id"], group["borda_points"])),
            "direct_signed_allocation": dict(zip(group["candidate_id"], group["allocation"])),
        }
        if not all(
            all(label in candidate_map for label in labels)
            for candidate_map in candidate_maps.values()
        ):
            continue
        evaluator_count += 1
        for method, candidate_map in candidate_maps.items():
            method_vectors[method] += np.array(
                [candidate_map[label] for label in labels],
                dtype=float,
            )

        for label in labels:
            matrix_rows.append(
                {
                    "evaluator": evaluator,
                    "candidate_id": label,
                    "best_pick_vote": candidate_maps["direct_best_pick"][label],
                    "borda_points": candidate_maps["direct_borda"][label],
                    "allocation": candidate_maps["direct_signed_allocation"][label],
                }
            )

    rows = []
    if evaluator_count == 0:
        return rows, matrix_rows

    for method, vector in method_vectors.items():
        vector = vector / evaluator_count
        score_dict = {label: float(value) for label, value in zip(labels, vector)}
        rows.append(
            {
                "method": method,
                "selection": tie_aware_winner(score_dict),
                "scores_json": json.dumps(score_dict, sort_keys=True),
            }
        )
    return rows, matrix_rows


def aggregate_all(evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregation_rows = []
    matrix_rows = []
    aggregation_columns = ["method", "selection", "scores_json", "prompt_id", "domain"]
    matrix_columns = [
        "evaluator",
        "candidate_id",
        "best_pick_vote",
        "borda_points",
        "allocation",
        "prompt_id",
        "domain",
    ]
    if evaluations.empty:
        return pd.DataFrame(columns=aggregation_columns), pd.DataFrame(
            columns=matrix_columns
        )
    labels = sorted(evaluations["candidate_id"].unique())

    for (prompt_id, domain), group in evaluations.groupby(["prompt_id", "domain"]):
        rows, prompt_matrix_rows = aggregate_prompt(group, labels)
        for row in rows:
            row["prompt_id"] = prompt_id
            row["domain"] = domain
            aggregation_rows.append(row)
        for row in prompt_matrix_rows:
            row["prompt_id"] = prompt_id
            row["domain"] = domain
            matrix_rows.append(row)

    return pd.DataFrame(aggregation_rows, columns=aggregation_columns), pd.DataFrame(
        matrix_rows, columns=matrix_columns
    )


def judge_prompt(
    user_prompt: str,
    candidates: pd.DataFrame,
    shuffle_candidates: bool,
    show_candidate_labels: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    shown_candidates, display_to_candidate = displayed_candidates(
        candidates,
        shuffle_candidates,
        show_candidate_labels,
    )
    labels = sorted(display_to_candidate.keys())
    candidate_text = "\n\n".join(
        f"Candidate {row.display_id}:\n{row.candidate_answer}"
        for row in shown_candidates.itertuples(index=False)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful external judge. Select the single best "
                "candidate answer for the user's prompt. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Candidate answers:\n{candidate_text}\n\n"
                "Return a JSON object with exactly these fields: "
                "best_candidate, reason. The best_candidate value must be one "
                f"of these candidate IDs: {', '.join(labels)}. The reason must "
                "be specific to the selected answer."
            ),
        },
    ]
    return messages, display_to_candidate


def weak_selector_prompt(
    user_prompt: str,
    candidates: pd.DataFrame,
    shuffle_candidates: bool,
    show_candidate_labels: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    shown_candidates, display_to_candidate = displayed_candidates(
        candidates,
        shuffle_candidates,
        show_candidate_labels,
    )
    labels = sorted(display_to_candidate.keys())
    candidate_text = "\n\n".join(
        f"Candidate {row.display_id}:\n{row.candidate_answer}"
        for row in shown_candidates.itertuples(index=False)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are selecting the best answer to a user's request. Return "
                "only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Candidate answers:\n{candidate_text}\n\n"
                "Return a JSON object with exactly these fields: "
                "best_candidate, reason. The best_candidate value must be one "
                f"of these candidate IDs: {', '.join(labels)}. The reason must "
                "be specific to the selected answer."
            ),
        },
    ]
    return messages, display_to_candidate


def run_weak_selector(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    model: ModelBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    vote_rows = []
    failures = []
    candidate_labels = set(candidates["candidate"].unique())

    for prompt in tqdm(list(prompts.itertuples(index=False)), desc="Weak selector"):
        group = candidates[candidates["prompt_id"] == prompt.prompt_id]
        prompt_votes = []
        for repeat_idx in range(args.weak_selector_repeats):
            output = ""
            try:
                messages, display_to_candidate = weak_selector_prompt(
                    prompt.user_prompt,
                    group,
                    args.shuffle_weak_selector_candidates,
                    args.show_candidate_labels,
                )
                output = chat_generate(
                    model,
                    messages,
                    max_new_tokens=args.judge_max_new_tokens,
                    temperature=args.weak_selector_temperature,
                    top_p=args.weak_selector_top_p,
                )
                parsed = translate_display_ids(extract_json(output), display_to_candidate)
                best = str(parsed["best_candidate"]).strip().upper()
                if best not in candidate_labels:
                    raise ValueError(f"weak selector returned invalid candidate {best!r}")
                reason = str(parsed.get("reason", "")).strip()
                if is_placeholder_reason(reason):
                    raise ValueError("placeholder weak selector reason")
                prompt_votes.append(best)
                vote_rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "domain": prompt.domain,
                        "selector_model": model.name,
                        "repeat_idx": repeat_idx,
                        "selection": best,
                        "reason": reason,
                        "raw_output": output,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "repeat_idx": repeat_idx,
                        "selector_model": model.name,
                        "error": repr(exc),
                        "raw_output": output,
                    }
                )

        if prompt_votes:
            vote_counts = {
                label: prompt_votes.count(label)
                for label in sorted(set(prompt_votes))
            }
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "method": f"single_weak_selector:{model.name}",
                    "selection": tie_aware_winner(vote_counts),
                    "scores_json": json.dumps(vote_counts, sort_keys=True),
                    "selector_repeats": len(prompt_votes),
                    "selector_consensus_share": max(vote_counts.values())
                    / len(prompt_votes),
                }
            )

    selection_columns = [
        "prompt_id",
        "domain",
        "method",
        "selection",
        "scores_json",
        "selector_repeats",
        "selector_consensus_share",
    ]
    vote_columns = [
        "prompt_id",
        "domain",
        "selector_model",
        "repeat_idx",
        "selection",
        "reason",
        "raw_output",
    ]
    failure_columns = [
        "prompt_id",
        "repeat_idx",
        "selector_model",
        "error",
        "raw_output",
    ]
    return (
        pd.DataFrame(rows, columns=selection_columns),
        pd.DataFrame(vote_rows, columns=vote_columns),
        pd.DataFrame(failures, columns=failure_columns),
    )


def run_external_judge(
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    model: ModelBundle,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    vote_rows = []
    failures = []
    candidate_labels = set(candidates["candidate"].unique())

    for prompt in tqdm(list(prompts.itertuples(index=False)), desc="External judge"):
        group = candidates[candidates["prompt_id"] == prompt.prompt_id]
        prompt_votes = []
        for repeat_idx in range(args.judge_repeats):
            output = ""
            try:
                messages, display_to_candidate = judge_prompt(
                    prompt.user_prompt,
                    group,
                    args.shuffle_judge_candidates,
                    args.show_candidate_labels,
                )
                output = chat_generate(
                    model,
                    messages,
                    max_new_tokens=args.judge_max_new_tokens,
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                )
                parsed = translate_display_ids(extract_json(output), display_to_candidate)
                best = str(parsed["best_candidate"]).strip().upper()
                if best not in candidate_labels:
                    raise ValueError(f"judge returned invalid candidate {best!r}")
                reason = str(parsed.get("reason", "")).strip()
                if is_placeholder_reason(reason):
                    raise ValueError("placeholder judge reason")
                prompt_votes.append(best)
                vote_rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "domain": prompt.domain,
                        "judge_model": model.name,
                        "repeat_idx": repeat_idx,
                        "judge_winner": best,
                        "reason": reason,
                        "raw_output": output,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "repeat_idx": repeat_idx,
                        "error": repr(exc),
                        "raw_output": output,
                    }
                )

        if prompt_votes:
            vote_counts = {
                label: prompt_votes.count(label)
                for label in sorted(set(prompt_votes))
            }
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "judge_model": model.name,
                    "judge_winner": tie_aware_winner(vote_counts),
                    "judge_repeats": len(prompt_votes),
                    "judge_vote_counts_json": json.dumps(vote_counts, sort_keys=True),
                    "judge_consensus_share": max(vote_counts.values())
                    / len(prompt_votes),
                }
            )
        print(f"Judged {prompt.prompt_id}")
    judge_columns = [
        "prompt_id",
        "domain",
        "judge_model",
        "judge_winner",
        "judge_repeats",
        "judge_vote_counts_json",
        "judge_consensus_share",
    ]
    vote_columns = [
        "prompt_id",
        "domain",
        "judge_model",
        "repeat_idx",
        "judge_winner",
        "reason",
        "raw_output",
    ]
    failure_columns = ["prompt_id", "repeat_idx", "error", "raw_output"]
    return (
        pd.DataFrame(rows, columns=judge_columns),
        pd.DataFrame(vote_rows, columns=vote_columns),
        pd.DataFrame(failures, columns=failure_columns),
    )


def build_summaries(
    aggregations: pd.DataFrame,
    judge_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_columns = [
        "prompt_id",
        "domain",
        "method",
        "selection",
        "judge_winner",
        "exact_match",
        "tie_inclusive_match",
        "is_tie",
    ]
    summary_columns = [
        "method",
        "n",
        "exact_match_rate",
        "tie_inclusive_match_rate",
        "tie_rate",
    ]
    domain_columns = [
        "domain",
        "method",
        "n",
        "exact_match_rate",
        "tie_inclusive_match_rate",
        "tie_rate",
    ]
    if aggregations.empty or judge_results.empty:
        return (
            pd.DataFrame(columns=selection_columns),
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=domain_columns),
        )

    selection_table = aggregations.merge(judge_results, on=["prompt_id", "domain"])
    match_rows = []
    for row in selection_table.itertuples(index=False):
        exact, tie_inclusive, tied = winner_matches(row.selection, row.judge_winner)
        match_rows.append(
            {
                "prompt_id": row.prompt_id,
                "domain": row.domain,
                "method": row.method,
                "selection": row.selection,
                "judge_winner": row.judge_winner,
                "exact_match": exact,
                "tie_inclusive_match": tie_inclusive,
                "is_tie": tied,
            }
        )
    selection_table = pd.DataFrame(match_rows, columns=selection_columns)

    if selection_table.empty:
        return (
            selection_table,
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=domain_columns),
        )

    method_summary = (
        selection_table.groupby("method")
        .agg(
            n=("prompt_id", "count"),
            exact_match_rate=("exact_match", "mean"),
            tie_inclusive_match_rate=("tie_inclusive_match", "mean"),
            tie_rate=("is_tie", "mean"),
        )
        .reset_index()
    )

    domain_summary = (
        selection_table.groupby(["domain", "method"])
        .agg(
            n=("prompt_id", "count"),
            exact_match_rate=("exact_match", "mean"),
            tie_inclusive_match_rate=("tie_inclusive_match", "mean"),
            tie_rate=("is_tie", "mean"),
        )
        .reset_index()
    )
    return selection_table, method_summary, domain_summary


def diagnostics(
    evaluations: pd.DataFrame,
    candidates: pd.DataFrame,
    aggregations: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    eval_diag_columns = [
        "evaluator_model",
        "evaluator",
        "evaluator_id",
        "n",
        "mean_allocation",
        "std_allocation",
        "min_allocation",
        "max_allocation",
    ]
    candidate_score_columns = [
        "candidate_id",
        "n",
        "mean_allocation",
        "std_allocation",
        "min_allocation",
        "max_allocation",
    ]
    completeness_columns = ["prompt_id", "evaluator_id", "complete_candidate_rows"]
    flat_columns = [
        "prompt_id",
        "evaluator_id",
        "candidate_rows",
        "unique_scores",
        "is_flat",
    ]
    duplicate_columns = [
        "prompt_id",
        "evaluator_id",
        "candidate_rows",
        "unique_scores",
        "duplicate_score_count",
        "duplicate_score_rate",
    ]
    correlation_columns = ["evaluator_a", "evaluator_b", "pearson_correlation"]
    position_columns = [
        "candidate",
        "generation_count",
        "selection_count",
        "tie_inclusion_count",
    ]
    if evaluations.empty:
        return (
            pd.DataFrame(columns=eval_diag_columns),
            pd.DataFrame(columns=candidate_score_columns),
            pd.DataFrame(columns=completeness_columns),
            pd.DataFrame(columns=flat_columns),
            pd.DataFrame(columns=duplicate_columns),
            pd.DataFrame(columns=correlation_columns),
            pd.DataFrame(columns=position_columns),
        )

    eval_diag = (
        evaluations.groupby(["evaluator_model", "evaluator", "evaluator_id"])
        .agg(
            n=("allocation", "count"),
            mean_allocation=("allocation", "mean"),
            std_allocation=("allocation", "std"),
            min_allocation=("allocation", "min"),
            max_allocation=("allocation", "max"),
        )
        .reset_index()
    )

    candidate_score_diag = (
        evaluations.groupby("candidate_id")
        .agg(
            n=("allocation", "count"),
            mean_allocation=("allocation", "mean"),
            std_allocation=("allocation", "std"),
            min_allocation=("allocation", "min"),
            max_allocation=("allocation", "max"),
        )
        .reset_index()
    )

    completeness = (
        evaluations.groupby(["prompt_id", "evaluator_id"])
        .agg(complete_candidate_rows=("candidate_id", "nunique"))
        .reset_index()
    )

    flat_rows = []
    duplicate_rows = []
    for (prompt_id, evaluator), group in evaluations.groupby(["prompt_id", "evaluator_id"]):
        unique_scores = group["allocation"].nunique(dropna=False)
        candidate_rows = int(group["candidate_id"].nunique())
        duplicate_count = max(0, candidate_rows - int(unique_scores))
        flat_rows.append(
            {
                "prompt_id": prompt_id,
                "evaluator_id": evaluator,
                "candidate_rows": candidate_rows,
                "unique_scores": int(unique_scores),
                "is_flat": bool(unique_scores <= 1),
            }
        )
        duplicate_rows.append(
            {
                "prompt_id": prompt_id,
                "evaluator_id": evaluator,
                "candidate_rows": candidate_rows,
                "unique_scores": int(unique_scores),
                "duplicate_score_count": duplicate_count,
                "duplicate_score_rate": duplicate_count / candidate_rows
                if candidate_rows
                else 0.0,
            }
        )
    flat_diag = pd.DataFrame(flat_rows, columns=flat_columns)
    duplicate_diag = pd.DataFrame(duplicate_rows, columns=duplicate_columns)

    correlation_rows = []
    if not evaluations.empty:
        pivot = evaluations.pivot_table(
            index=["prompt_id", "candidate_id"],
            columns="evaluator_id",
            values="allocation",
            aggfunc="mean",
        )
        corr = pivot.corr()
        for evaluator_a in corr.index:
            for evaluator_b in corr.columns:
                correlation_rows.append(
                    {
                        "evaluator_a": evaluator_a,
                        "evaluator_b": evaluator_b,
                        "pearson_correlation": corr.loc[evaluator_a, evaluator_b],
                    }
                )
    correlation_diag = pd.DataFrame(
        correlation_rows,
        columns=correlation_columns,
    )

    position_rows = []
    for label in sorted(candidates["candidate"].unique()):
        position_rows.append(
            {
                "candidate": label,
                "generation_count": int((candidates["candidate"] == label).sum()),
                "selection_count": int(
                    aggregations["selection"].eq(label).sum()
                ),
                "tie_inclusion_count": int(
                    aggregations["selection"].str.contains(label, regex=False).sum()
                ),
            }
        )
    return (
        eval_diag,
        candidate_score_diag,
        completeness,
        flat_diag,
        duplicate_diag,
        correlation_diag,
        pd.DataFrame(position_rows, columns=position_columns),
    )


def classify_failure(error: str) -> str:
    lower = error.lower()
    if "no json object" in lower or "jsondecodeerror" in lower:
        return "json_parse_failure"
    if "candidate_id mismatch" in lower:
        return "candidate_id_mismatch"
    if "allocation missing" in lower or "could not convert" in lower:
        return "allocation_missing_or_nonnumeric"
    if "votes missing" in lower or "borda_ranking" in lower:
        return "vote_schema_failure"
    if "invalid" in lower or "allocation out of range" in lower:
        return "invalid_vote"
    if "placeholder" in lower:
        return "placeholder_or_copied_schema"
    return "other"


def failure_diagnostics(failures: pd.DataFrame) -> pd.DataFrame:
    columns = ["failure_type", "n"]
    if failures.empty:
        return pd.DataFrame(columns=columns)
    rows = (
        failures.assign(failure_type=failures["error"].map(classify_failure))
        .groupby("failure_type")
        .size()
        .reset_index(name="n")
    )
    return rows


def vote_repair_diagnostics(evaluations: pd.DataFrame) -> pd.DataFrame:
    columns = ["repair_type", "n", "n_events"]
    if evaluations.empty or "vote_repairs_json" not in evaluations.columns:
        return pd.DataFrame(columns=columns)

    ballot_keys = [
        column
        for column in ("evaluator_model", "prompt_id", "evaluator_id")
        if column in evaluations.columns
    ]
    if not ballot_keys:
        ballot_rows = evaluations[["vote_repairs_json"]].copy()
    else:
        ballot_rows = evaluations.drop_duplicates(ballot_keys)[
            ballot_keys + ["vote_repairs_json"]
        ]

    ballot_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    for repairs_json in ballot_rows["vote_repairs_json"].dropna():
        try:
            repairs = json.loads(repairs_json)
        except Exception:
            repairs = []
        ballot_types: set[str] = set()
        for repair in repairs:
            repair_type = str(repair).split(":", 1)[0]
            ballot_types.add(repair_type)
            event_counts[repair_type] = event_counts.get(repair_type, 0) + 1
        for repair_type in ballot_types:
            ballot_counts[repair_type] = ballot_counts.get(repair_type, 0) + 1
    return pd.DataFrame(
        [
            {
                "repair_type": key,
                "n": ballot_counts[key],
                "n_events": event_counts.get(key, 0),
            }
            for key in sorted(ballot_counts)
        ],
        columns=columns,
    )


def ballot_quality_diagnostics(
    evaluations: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "evaluator_model",
        "attempted_ballots",
        "parsed_ballots",
        "clean_ballots",
        "repaired_ballots",
        "failed_ballots",
        "clean_rate_attempted",
        "repaired_rate_attempted",
        "failure_rate_attempted",
        "parsed_candidate_rows",
    ]

    model_names: set[str] = set()
    for frame in (evaluations, failures):
        if not frame.empty and "evaluator_model" in frame.columns:
            model_names.update(frame["evaluator_model"].dropna().astype(str))
    if not model_names:
        model_names = {"ALL"}

    rows = []
    for model_name in sorted(model_names):
        eval_part = evaluations
        failure_part = failures
        if model_name != "ALL":
            if "evaluator_model" in eval_part.columns:
                eval_part = eval_part[eval_part["evaluator_model"].astype(str) == model_name]
            if "evaluator_model" in failure_part.columns:
                failure_part = failure_part[
                    failure_part["evaluator_model"].astype(str) == model_name
                ]

        ballot_keys = [
            column
            for column in ("prompt_id", "evaluator_id")
            if column in eval_part.columns
        ]
        if eval_part.empty:
            parsed = pd.DataFrame(columns=ballot_keys + ["vote_repair_count"])
        elif ballot_keys:
            parsed = (
                eval_part.groupby(ballot_keys, dropna=False, as_index=False)
                .agg(vote_repair_count=("vote_repair_count", "max"))
            )
        else:
            parsed = eval_part[["vote_repair_count"]].copy()

        failure_keys = [
            column
            for column in ("prompt_id", "evaluator_id")
            if column in failure_part.columns
        ]
        if failure_part.empty:
            failed_ballots = 0
        elif failure_keys:
            failed_ballots = int(failure_part.drop_duplicates(failure_keys).shape[0])
        else:
            failed_ballots = int(len(failure_part))

        repair_counts = pd.to_numeric(
            parsed.get("vote_repair_count", pd.Series(dtype=float)),
            errors="coerce",
        )
        parsed_ballots = int(len(parsed))
        clean_ballots = int(repair_counts.eq(0).sum())
        repaired_ballots = int(repair_counts.gt(0).sum())
        attempted_ballots = parsed_ballots + failed_ballots

        rows.append(
            {
                "evaluator_model": model_name,
                "attempted_ballots": attempted_ballots,
                "parsed_ballots": parsed_ballots,
                "clean_ballots": clean_ballots,
                "repaired_ballots": repaired_ballots,
                "failed_ballots": failed_ballots,
                "clean_rate_attempted": clean_ballots / attempted_ballots
                if attempted_ballots
                else math.nan,
                "repaired_rate_attempted": repaired_ballots / attempted_ballots
                if attempted_ballots
                else math.nan,
                "failure_rate_attempted": failed_ballots / attempted_ballots
                if attempted_ballots
                else math.nan,
                "parsed_candidate_rows": int(len(eval_part)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df.to_csv(path, index=False)
    else:
        df.sort_index(axis=1).to_csv(path, index=False)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    save_csv(df, out_dir / f"{stem}.csv")
    save_jsonl(df, out_dir / f"{stem}.jsonl")


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if non_empty_frames:
        return pd.concat(non_empty_frames, ignore_index=True)
    return frames[0].copy()


def main() -> None:
    args = parse_args()
    if args.num_candidates != 4:
        raise ValueError("Level 1 uses exactly four candidate IDs: A, B, C, D")
    if args.judge_repeats < 1:
        raise ValueError("--judge-repeats must be at least 1")
    if args.weak_selector_repeats < 1:
        raise ValueError("--weak-selector-repeats must be at least 1")
    set_seed(args.seed)
    evaluator_models = parse_model_list(args.evaluator_models, args.candidate_model)

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"level1_direct_vote_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_csv, args.max_prompts)
    save_table(prompts, out_dir, "prompts")
    save_table(
        pd.DataFrame(
            [
                {
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
                    "judge_temperature": args.judge_temperature,
                    "judge_top_p": args.judge_top_p,
                    "weak_selector_temperature": args.weak_selector_temperature,
                    "weak_selector_top_p": args.weak_selector_top_p,
                    "shuffle_evaluator_candidates": args.shuffle_evaluator_candidates,
                    "shuffle_judge_candidates": args.shuffle_judge_candidates,
                    "shuffle_weak_selector_candidates": args.shuffle_weak_selector_candidates,
                    "show_candidate_labels": args.show_candidate_labels,
                    "judge_repeats": args.judge_repeats,
                    "weak_selector_repeats": args.weak_selector_repeats,
                }
            ]
        ),
        out_dir,
        "run_config",
    )

    if args.candidates_csv:
        candidates = load_candidates_csv(args.candidates_csv, prompts, args.num_candidates)
    elif args.skip_generation:
        candidates_path = out_dir / "candidates.csv"
        if not candidates_path.exists():
            raise FileNotFoundError(f"--skip-generation needs {candidates_path}")
        candidates = pd.read_csv(candidates_path)
    else:
        candidate_model = load_model(args.candidate_model, args.fallback_model)
        candidates = generate_candidates(prompts, candidate_model, args)
        release_model(candidate_model)
    save_table(candidates, out_dir, "candidates")

    evaluation_frames = []
    failed_evaluation_frames = []
    weak_selection_frames = []
    weak_vote_frames = []
    weak_failure_frames = []
    for evaluator_model_name in evaluator_models:
        evaluator_model = load_model(evaluator_model_name, args.fallback_model)
        evaluations_part, failed_evaluations_part = run_prompted_evaluations(
            prompts, candidates, evaluator_model, args
        )
        weak_selections_part, weak_votes_part, weak_failures_part = run_weak_selector(
            prompts, candidates, evaluator_model, args
        )
        evaluation_frames.append(evaluations_part)
        failed_evaluation_frames.append(failed_evaluations_part)
        weak_selection_frames.append(weak_selections_part)
        weak_vote_frames.append(weak_votes_part)
        weak_failure_frames.append(weak_failures_part)
        release_model(evaluator_model)

    evaluations = concat_frames(evaluation_frames)
    failed_evaluations = concat_frames(failed_evaluation_frames)
    weak_selector_aggregations = concat_frames(weak_selection_frames)
    weak_selector_votes = concat_frames(weak_vote_frames)
    failed_weak_selector = concat_frames(weak_failure_frames)
    save_table(evaluations, out_dir, "direct_votes")
    save_table(evaluations, out_dir, "prompted_evaluations")
    save_table(
        vote_repair_diagnostics(evaluations),
        out_dir,
        "vote_repair_diagnostics",
    )
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

    judge_model = load_model(
        args.judge_model,
        args.fallback_model,
        load_4bit=args.load_judge_4bit,
    )
    judge_results, judge_votes, failed_judge_results = run_external_judge(
        prompts, candidates, judge_model, args
    )
    release_model(judge_model)
    save_table(judge_results, out_dir, "external_judge_results")
    save_table(judge_votes, out_dir, "external_judge_votes")
    save_table(failed_judge_results, out_dir, "failed_external_judge_results")
    save_table(
        failure_diagnostics(failed_judge_results),
        out_dir,
        "failed_external_judge_diagnostics",
    )

    selection_table, method_summary, domain_summary = build_summaries(
        comparison_selections, judge_results
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

    archive_base = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Saved outputs to {out_dir}")
    print(f"Created archive {archive_base}")


if __name__ == "__main__":
    main()
