#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a "self-ASR then jointly-reason" variant of the MELD-style dataset.

Motivation
----------
Empirically we observed:
    * Prompt WITHOUT ASR transcript injected  ->  test accuracy ~59%
    * Prompt WITH  ASR transcript injected    ->  test accuracy ~69% (+10%)
which shows the spoken-content text is a very strong cue for emotion
recognition on MELD. However at inference time we typically do NOT have a
ground-truth transcript, so we cannot rely on
`optimize_user_prompt_r1omni_asr.py`-style prompts in production.

This script therefore INTERNALIZES the ASR benefit into the model itself:
    * The USER prompt only contains the audio and a fixed instruction --
      no external transcript is provided.
    * The instruction asks the model to
        (1) transcribe what the speaker says,
        (2) output the final emotion label.
    * The ASSISTANT target (used for SFT) is rewritten with a structured
      "chain" so the model can actually learn the above behavior:
          <transcript>gold ASR text from MELD CSV</transcript>
          <answer>gold label</answer>
      The gold transcript comes from `data/MELD_audio/{split}.csv`.

Compared to the previous two scripts:
    * `optimize_user_prompt_r1omni.py`      : rewrites USER only,
                                              assistant stays a bare label.
    * `optimize_user_prompt_r1omni_asr.py`  : rewrites USER only, INJECTS
                                              gold ASR into the prompt
                                              (requires transcript at
                                              inference time; not realistic).
    * THIS SCRIPT                           : rewrites both USER and
                                              ASSISTANT so the model learns
                                              to do ASR + emotion jointly
                                              from audio alone.

Compatibility with evaluator
----------------------------
`eval_classification_meld.py::normalize_label` (word mode) matches the
predicted label by lower-cased substring inclusion. Since the assistant
target ends with `<answer>{label}</answer>`, the evaluator can still
extract the correct label whether the model outputs the full structure
or only the final `<answer>` segment.

Usage
-----
    # dev split (default paths)
    python optimize_user_prompt_r1omni_self_asr.py

    # explicit split
    python optimize_user_prompt_r1omni_self_asr.py \
        --input   project/stepaudio/data_meld/train.jsonl \
        --output  project/stepaudio/data_meld/train.r1omni_self_asr.jsonl \
        --meta    data/MELD_audio/train.csv

    # process all three splits in one shot
    python optimize_user_prompt_r1omni_self_asr.py --all
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# The 7 emotion classes used by MELD.
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
# USER PROMPT
# ---------------------------------------------------------------------------
# Notes:
#   * Only audio + fixed instruction; NO external transcript is provided.
#   * The 3-step chain is explicit so the model knows what to emit.
#   * We keep it short but unambiguous; each step maps to one XML-like tag
#     so downstream parsing is trivial.
#   * The final line reiterates the closed label set to avoid answer drift.
# ---------------------------------------------------------------------------
R1OMNI_SELF_ASR_USER_PROMPT = (
    "<audio>"
    "You are an emotion recognition expert. You are given only a short speech clip. "
    "Follow these two steps and output them in order using the exact XML-style tags below:\n"
    "1. Transcribe what the speaker says as accurately as possible and put it inside "
    "<transcript>...</transcript>.\n"
    "2. Output the final emotion label inside <answer>...</answer>. "
    "The label MUST be exactly one of "
    "[surprise, anger, neutral, joy, sadness, fear, disgust]."
)


# ---------------------------------------------------------------------------
# ASSISTANT TARGET
# ---------------------------------------------------------------------------
# The assistant target only contains two XML-style segments:
#   * <transcript>...</transcript>  -- gold ASR text from MELD CSV,
#   * <answer>...</answer>          -- gold emotion label.
# There is intentionally NO <analysis> segment: we do not have gold
# paralinguistic descriptions and adding a templated analysis would only
# teach the model to memorize boilerplate. The SFT signal therefore focuses
# on faithful transcription plus correct label prediction.
# ---------------------------------------------------------------------------
def build_assistant_target(transcript: str, label: str) -> str:
    """Assemble the two-tag structured assistant target."""
    label_norm = (label or "").strip().lower()
    # Keep transcript on a single line to avoid confusing tag parsers.
    transcript_line = re.sub(r"\s+", " ", transcript or "").strip()
    return (
        f"<transcript>{transcript_line}</transcript>\n"
        f"<answer>{label_norm}</answer>"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"dia(\d+)_utt(\d+)")


def load_transcripts(meta_csv: str) -> Dict[str, str]:
    """Load MELD metadata CSV -> {"dia{DID}_utt{UID}": Utterance}."""
    assert os.path.isfile(meta_csv), f"Meta CSV not found: {meta_csv}"
    id2text: Dict[str, str] = {}
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = str(row.get("Dialogue_ID", "")).strip()
            uid = str(row.get("Utterance_ID", "")).strip()
            utt = (row.get("Utterance", "") or "").strip()
            if did == "" or uid == "":
                continue
            id2text[f"dia{did}_utt{uid}"] = utt
    return id2text


def clean_transcript(text: str) -> str:
    """Light cleanup of a transcript so it can safely be embedded in
    XML-like tags."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    # Guard against accidental early-closing tags inside the transcript.
    text = text.replace("</transcript>", "</ transcript>")
    text = text.replace("</answer>", "</ answer>")
    return text


def extract_key(sample: Dict[str, Any]) -> Optional[str]:
    """Extract dia{X}_utt{Y} key from sample.key or audio path."""
    key = sample.get("key")
    if isinstance(key, str):
        m = _KEY_RE.search(key)
        if m:
            return f"dia{m.group(1)}_utt{m.group(2)}"
    for a in sample.get("audios") or []:
        if isinstance(a, str):
            m = _KEY_RE.search(a)
            if m:
                return f"dia{m.group(1)}_utt{m.group(2)}"
    return None


def extract_gold_label(sample: Dict[str, Any]) -> str:
    """Best-effort extraction of gold label (falls back to assistant content)."""
    for k in ("label", "labels", "label_str_origin"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            c = (msg.get("content") or "").strip().lower()
            if c:
                return c
    return ""


# ---------------------------------------------------------------------------
# Core rewrite
# ---------------------------------------------------------------------------
def rewrite_sample(sample: Dict[str, Any],
                   id2text: Dict[str, str],
                   stats: Dict[str, int]) -> Dict[str, Any]:
    """Rewrite ONE JSONL sample: user prompt + assistant target."""
    new_sample = dict(sample)
    messages = new_sample.get("messages", []) or []
    if not messages:
        return new_sample

    key = extract_key(new_sample)
    transcript_raw = id2text.get(key) if key else None
    transcript = clean_transcript(transcript_raw) if transcript_raw else ""
    label = extract_gold_label(new_sample)

    if transcript:
        stats["asr_hit"] += 1
    else:
        stats["asr_miss"] += 1

    assistant_target = build_assistant_target(transcript, label)

    new_messages: List[Dict[str, Any]] = []
    user_done = False
    asst_done = False
    for msg in messages:
        role = msg.get("role", "")
        if role == "user" and not user_done:
            new_messages.append(
                {"role": "user", "content": R1OMNI_SELF_ASR_USER_PROMPT}
            )
            user_done = True
        elif role == "assistant" and not asst_done:
            new_messages.append(
                {"role": "assistant", "content": assistant_target}
            )
            asst_done = True
        else:
            new_messages.append({"role": role, "content": msg.get("content", "")})

    # If the original had no assistant turn (rare), append one so SFT works.
    if not asst_done:
        new_messages.append({"role": "assistant", "content": assistant_target})

    new_sample["messages"] = new_messages
    new_sample["prompt_version"] = "r1omni_self_asr_v1"
    if transcript:
        new_sample["asr_text"] = transcript
    return new_sample


def process(input_path: str, output_path: str, meta_csv: str) -> Dict[str, int]:
    assert os.path.isfile(input_path), f"Input file not found: {input_path}"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    id2text = load_transcripts(meta_csv)
    print(f"[info] loaded {len(id2text)} transcripts from {meta_csv}")

    stats = {"asr_hit": 0, "asr_miss": 0}
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
                print(f"[warn] line {line_no}: JSON decode error: {e}",
                      file=sys.stderr)
                continue
            new_sample = rewrite_sample(sample, id2text, stats)
            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            n_ok += 1

    print(
        f"[done] input={input_path}\n"
        f"       output={output_path}\n"
        f"       total={n_total}  ok={n_ok}  err={n_err}  "
        f"asr_hit={stats['asr_hit']}  asr_miss={stats['asr_miss']}"
    )
    return {"total": n_total, "ok": n_ok, "err": n_err, **stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _paths_for(split: str) -> Tuple[str, str, str]:
    inp = os.path.join(_HERE, "data_meld", f"{split}.jsonl")
    outp = os.path.join(_HERE, "data_meld", f"{split}.r1omni_self_asr.jsonl")
    meta = os.path.join(_WS_ROOT, "data", "MELD_audio", f"{split}.csv")
    return inp, outp, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a self-ASR + joint-reasoning variant of the MELD dataset. "
            "Both user prompt and assistant target are rewritten so that the "
            "model learns to transcribe first and then decide the emotion."
        )
    )
    default_in, default_out, default_meta = _paths_for("dev")
    p.add_argument("--input", "-i", default=default_in,
                   help=f"Input JSONL. default={default_in}")
    p.add_argument("--output", "-o", default=default_out,
                   help=f"Output JSONL. default={default_out}")
    p.add_argument("--meta", "-m", default=default_meta,
                   help=(
                       "MELD metadata CSV with [Dialogue_ID, Utterance_ID, "
                       f"Utterance, ...]. default={default_meta}"
                   ))
    p.add_argument("--all", action="store_true",
                   help="Process train/dev/test in one shot (ignores -i/-o/-m).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        for split in ("train", "dev", "test"):
            inp, outp, meta = _paths_for(split)
            if not os.path.isfile(inp):
                print(f"[skip] {split}: input not found: {inp}", file=sys.stderr)
                continue
            if not os.path.isfile(meta):
                print(f"[skip] {split}: meta not found: {meta}", file=sys.stderr)
                continue
            process(inp, outp, meta)
    else:
        process(args.input, args.output, args.meta)


if __name__ == "__main__":
    main()
