#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize the `user` role content for MELD-style audio emotion recognition dataset,
following the prompt design in R1-Omni (arXiv:2503.21480):

    "R1-Omni: Explainable Omni-Multimodal Emotion Recognition
     with Reinforcement Learning"

Key ideas from the paper that we adopt for the user prompt:
    1. Position the model as an "emotional recognition expert".
    2. Explicitly ask the model to reason over auditory cues
       (tone, pitch, speech rate, energy, intonation, pauses, vocal quality, etc.).
    3. Constrain the final answer to a fixed emotion label set.

NOTE: We intentionally DROP the <think>...</think><answer>...</answer>
structured-output requirement here, because the current dataset's assistant
turn only contains a single label (e.g. "sadness") with no gold reasoning
chain. Asking the model to produce <think>/<answer> while training targets
are bare labels would cause a train/inference format mismatch. The prompt
therefore only asks for a single-word emotion label as the final answer,
which matches the existing assistant supervision.

This script reads the original JSONL (each line contains:
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "<label>"}],
     "audios": [...], "label": "...", ...}
and rewrites the FIRST user turn's content, keeping everything else unchanged.

Usage:
    python optimize_user_prompt_r1omni.py \
        --input  project/stepaudio/data_meld/dev.jsonl \
        --output project/stepaudio/data_meld/dev.r1omni.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# The 7 emotion classes used by MELD, matching the original dataset prompt.
EMOTION_LABELS: List[str] = [
    "surprise",
    "anger",
    "neutral",
    "joy",
    "sadness",
    "fear",
    "disgust",
]

# ---------------------------------------------------------------------------
# R1-Omni inspired prompt for the user turn (label-only variant).
#
# Design choices:
#   * Explicit expert role  ->  "As an emotion recognition expert"
#   * Enumerate acoustic cues the model should attend to (paper Sec. 3 / 4).
#   * Constrain the final answer to a fixed MELD 7-class label set.
#   * Ask for a single-word answer only (NO <think>/<answer> tags), so that
#     the prompt is consistent with the existing single-label supervision.
#   * Keep <audio> placeholder at the beginning so it matches the existing
#     ms-swift multimodal template.
# ---------------------------------------------------------------------------
R1OMNI_USER_PROMPT = (
    "<audio>"
    "As an emotion recognition expert, carefully listen to the given speech clip. "
    "Analyze the speaker's vocal cues, including tone, pitch, loudness, speech rate, "
    "intonation, pauses, emphasis and overall voice quality, and infer the speaker's "
    "emotional state. "
    "The predicted emotion label MUST be exactly one of "
    "[surprise, anger, neutral, joy, sadness, fear, disgust]. "
    "Answer with the single emotion word only, without any explanation or extra text."
)


def rewrite_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite one JSONL sample (returns a new dict). Only the first user
    turn's content is replaced; assistant / audios / label are untouched."""
    new_sample = dict(sample)  # shallow copy
    messages = new_sample.get("messages", [])
    if not messages:
        return new_sample

    new_messages: List[Dict[str, Any]] = []
    user_rewritten = False
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and not user_rewritten:
            new_messages.append({"role": "user", "content": R1OMNI_USER_PROMPT})
            user_rewritten = True
        else:
            new_messages.append({"role": role, "content": content})

    new_sample["messages"] = new_messages
    # Book-keeping: remember which prompt version this file uses.
    new_sample["prompt_version"] = "r1omni_label_only_v2"
    return new_sample


def process(input_path: str, output_path: str) -> None:
    assert os.path.isfile(input_path), f"Input file not found: {input_path}"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    n_total = 0
    n_ok = 0
    n_err = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            n_total += 1
            try:
                sample = json.loads(raw)
            except json.JSONDecodeError as e:
                n_err += 1
                print(
                    f"[warn] line {line_no}: JSON decode error: {e}",
                    file=sys.stderr,
                )
                continue
            new_sample = rewrite_sample(sample)
            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            n_ok += 1

    print(
        f"[done] input={input_path}  output={output_path}\n"
        f"       total={n_total}  ok={n_ok}  err={n_err}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Optimize user-role prompt in a MELD-style JSONL dataset following "
            "the R1-Omni (arXiv:2503.21480) prompting scheme."
        )
    )
    default_in = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_meld",
        "dev.jsonl",
    )
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_meld",
        "dev.r1omni.jsonl",
    )
    p.add_argument("--input", "-i", default=default_in,
                   help=f"Input JSONL path. default={default_in}")
    p.add_argument("--output", "-o", default=default_out,
                   help=f"Output JSONL path. default={default_out}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
