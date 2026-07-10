#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a "self-ASR REVERSED" variant of the MELD-style dataset.

This is a sibling of `optimize_user_prompt_r1omni_self_asr.py`. Everything
in the pipeline is identical EXCEPT the assistant's output ORDER:

    self_asr (original):    <transcript>...</transcript>\n<answer>xxx</answer>
    self_asr REVERSED:      <answer>xxx</answer>\n<transcript>...</transcript>

Motivation & caveat
-------------------
The original "transcribe first, decide later" ordering lets the label
generation causally attend to the freshly-emitted transcript (autoregressive
decoders can only look leftwards), which is arguably the mechanism behind
the +10pp accuracy gain we observed from explicit ASR injection.

Flipping the order means:
    * Label is emitted with audio-only context.
    * The transcript is produced AFTER the label -- it can no longer
      influence the label's decoding at inference time.
    * Training loss still supervises both, so the model still LEARNS to
      transcribe; but the transcript is no longer a causal prerequisite
      for the label.

This script is therefore mainly useful as a controlled comparison to
quantify how much of the self-ASR benefit comes from
    (i)  putting the transcript BEFORE the answer  (causal ordering effect),
vs.
    (ii) simply having a transcription objective at all  (multi-task effect).

Compatibility with evaluator
----------------------------
`eval_classification_meld.py::normalize_label` (word mode) matches by
lower-cased substring inclusion. The assistant target STARTS with
`<answer>{label}</answer>`, so the label token is present regardless of
whether the generated <transcript> block that follows is truncated.

Usage
-----
    # dev split (default paths)
    python optimize_user_prompt_r1omni_self_asr_reversed.py

    # explicit split
    python optimize_user_prompt_r1omni_self_asr_reversed.py \
        --input   project/stepaudio/data_meld/train.jsonl \
        --output  project/stepaudio/data_meld/train.r1omni_self_asr_reversed.jsonl \
        --meta    data/MELD_audio/train.csv

    # process all three splits in one shot
    python optimize_user_prompt_r1omni_self_asr_reversed.py --all
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
# USER PROMPT (REVERSED ORDER)
# ---------------------------------------------------------------------------
# Same audio-only setup as the non-reversed variant, but the instruction
# explicitly asks the model to emit the emotion label FIRST and then the
# transcript. Keeping the wording symmetric with the original so any
# accuracy delta between the two datasets can be attributed to ordering
# rather than prompt phrasing.
# ---------------------------------------------------------------------------
R1OMNI_SELF_ASR_REVERSED_USER_PROMPT = (
    "<audio>"
    "You are an emotion recognition expert. You are given only a short speech clip. "
    "Follow these two steps and output them in order using the exact XML-style tags below:\n"
    "1. Output the final emotion label inside <answer>...</answer>. "
    "The label MUST be exactly one of "
    "[surprise, anger, neutral, joy, sadness, fear, disgust].\n"
    "2. Transcribe what the speaker says as accurately as possible and put it inside "
    "<transcript>...</transcript>."
)


# ---------------------------------------------------------------------------
# ASSISTANT TARGET (REVERSED ORDER)
# ---------------------------------------------------------------------------
# Two XML-style segments in the OPPOSITE order relative to self_asr:
#   * <answer>...</answer>          -- gold emotion label (first),
#   * <transcript>...</transcript>  -- gold ASR text from MELD CSV (second).
# ---------------------------------------------------------------------------
def build_assistant_target(transcript: str, label: str) -> str:
    """Assemble the reversed two-tag structured assistant target."""
    label_norm = (label or "").strip().lower()
    # Keep transcript on a single line to avoid confusing tag parsers.
    transcript_line = re.sub(r"\s+", " ", transcript or "").strip()
    return (
        f"<answer>{label_norm}</answer>\n"
        f"<transcript>{transcript_line}</transcript>"
    )


# ---------------------------------------------------------------------------
# Helpers (identical to self_asr variant)
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
    """Rewrite ONE JSONL sample: user prompt + reversed assistant target."""
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
                {"role": "user", "content": R1OMNI_SELF_ASR_REVERSED_USER_PROMPT}
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
    new_sample["prompt_version"] = "r1omni_self_asr_reversed_v1"
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
    outp = os.path.join(_HERE, "data_meld", f"{split}.r1omni_self_asr_reversed.jsonl")
    meta = os.path.join(_WS_ROOT, "data", "MELD_audio", f"{split}.csv")
    return inp, outp, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a REVERSED self-ASR variant of the MELD dataset: assistant "
            "outputs <answer>...</answer> FIRST and <transcript>...</transcript> "
            "SECOND. Useful as an ablation vs. the non-reversed self_asr set."
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
