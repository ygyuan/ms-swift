#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize the `user` role content for MELD-style audio emotion recognition
dataset, and additionally inject the ASR transcript of each audio clip into
the user prompt.

Prompt design still follows R1-Omni (arXiv:2503.21480):
    "R1-Omni: Explainable Omni-Multimodal Emotion Recognition
     with Reinforcement Learning"

Extra ingredient over `optimize_user_prompt_r1omni.py`:
    * We look up the per-utterance transcript from a MELD metadata CSV
      (default: data/MELD_audio/{split}.csv), where each row has fields
      `Dialogue_ID`, `Utterance_ID` and `Utterance`.
    * The transcript is appended to the user prompt as an auxiliary text
      cue, e.g.
          "The speaker says: \"<transcript>\". "
      This mirrors R1-Omni's multimodal setup where the model can attend
      to both audio and the aligned text transcript.

Sample matching rule:
    The dataset JSONL contains a `key` field (e.g. "dia0_utt0") which is
    built from Dialogue_ID / Utterance_ID in the MELD_Audio.py loader:
        f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
    If `key` is missing, we fall back to parsing it from the audio filename.

Usage:
    python optimize_user_prompt_r1omni_asr.py \
        --input   project/stepaudio/data_meld/dev.jsonl \
        --output  project/stepaudio/data_meld/dev.r1omni_asr.jsonl \
        --meta    data/MELD_audio/dev.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

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
# Prompt template.
#   * Keep the R1-Omni "emotion recognition expert" persona and the explicit
#     enumeration of vocal cues.
#   * Insert an ASR transcript so the model can jointly leverage audio and
#     the spoken text (paper Sec. 3, R1-Omni uses both modalities).
#   * Still ask for a single-word answer to stay compatible with the
#     existing label-only supervision.
#   * `{transcript}` will be substituted per-sample.
# ---------------------------------------------------------------------------
R1OMNI_USER_PROMPT_TEMPLATE = (
    "<audio>"
    "As an emotion recognition expert, carefully listen to the given speech clip "
    "and read its transcript. "
    "The transcript of what the speaker says is: \"{transcript}\". "
    "Analyze both the speaker's vocal cues (tone, pitch, loudness, speech rate, "
    "intonation, pauses, emphasis and overall voice quality) and the semantic "
    "content of the transcript, then infer the speaker's emotional state. "
    "The predicted emotion label MUST be exactly one of "
    "[surprise, anger, neutral, joy, sadness, fear, disgust]. "
    "Answer with the single emotion word only, without any explanation or extra text."
)

# Fallback prompt when transcript is missing (same as label-only variant).
R1OMNI_USER_PROMPT_NO_ASR = (
    "<audio>"
    "As an emotion recognition expert, carefully listen to the given speech clip. "
    "Analyze the speaker's vocal cues, including tone, pitch, loudness, speech rate, "
    "intonation, pauses, emphasis and overall voice quality, and infer the speaker's "
    "emotional state. "
    "The predicted emotion label MUST be exactly one of "
    "[surprise, anger, neutral, joy, sadness, fear, disgust]. "
    "Answer with the single emotion word only, without any explanation or extra text."
)


# Match "dia<DID>_utt<UID>" pattern in a path or key.
_KEY_RE = re.compile(r"dia(\d+)_utt(\d+)")


def load_transcripts(meta_csv: str) -> Dict[str, str]:
    """Load MELD metadata CSV and return a mapping:
        key = f"dia{Dialogue_ID}_utt{Utterance_ID}"  ->  Utterance text
    """
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
            key = f"dia{did}_utt{uid}"
            id2text[key] = utt
    return id2text


def clean_transcript(text: str) -> str:
    """Light cleanup of a transcript so it can safely be embedded in the
    prompt string (which itself uses double quotes)."""
    if text is None:
        return ""
    # Collapse whitespace / newlines.
    text = re.sub(r"\s+", " ", text).strip()
    # Escape stray double quotes to avoid ambiguity with the surrounding
    # quotation marks in the template.
    text = text.replace('"', "'")
    return text


def extract_key(sample: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the dia{X}_utt{Y} key from a sample."""
    key = sample.get("key")
    if isinstance(key, str) and _KEY_RE.search(key):
        m = _KEY_RE.search(key)
        return f"dia{m.group(1)}_utt{m.group(2)}"
    # Fallback: parse from the audio file path.
    audios = sample.get("audios") or []
    for a in audios:
        if not isinstance(a, str):
            continue
        m = _KEY_RE.search(a)
        if m:
            return f"dia{m.group(1)}_utt{m.group(2)}"
    return None


def build_user_content(transcript: Optional[str]) -> str:
    if transcript:
        return R1OMNI_USER_PROMPT_TEMPLATE.format(transcript=transcript)
    return R1OMNI_USER_PROMPT_NO_ASR


def rewrite_sample(sample: Dict[str, Any],
                   id2text: Dict[str, str],
                   stats: Dict[str, int]) -> Dict[str, Any]:
    """Rewrite one JSONL sample; only the first user turn is replaced."""
    new_sample = dict(sample)
    messages = new_sample.get("messages", [])
    if not messages:
        return new_sample

    key = extract_key(new_sample)
    transcript_raw = id2text.get(key) if key else None
    transcript = clean_transcript(transcript_raw) if transcript_raw else None

    if transcript:
        stats["asr_hit"] += 1
    else:
        stats["asr_miss"] += 1

    new_user_content = build_user_content(transcript)

    new_messages: List[Dict[str, Any]] = []
    user_rewritten = False
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and not user_rewritten:
            new_messages.append({"role": "user", "content": new_user_content})
            user_rewritten = True
        else:
            new_messages.append({"role": role, "content": content})

    new_sample["messages"] = new_messages
    new_sample["prompt_version"] = "r1omni_label_only_asr_v1"
    # Store the transcript for traceability / downstream use.
    if transcript:
        new_sample["asr_text"] = transcript
    return new_sample


def process(input_path: str, output_path: str, meta_csv: str) -> None:
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
                print(
                    f"[warn] line {line_no}: JSON decode error: {e}",
                    file=sys.stderr,
                )
                continue
            new_sample = rewrite_sample(sample, id2text, stats)
            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            n_ok += 1

    print(
        f"[done] input={input_path}  output={output_path}\n"
        f"       total={n_total}  ok={n_ok}  err={n_err}\n"
        f"       asr_hit={stats['asr_hit']}  asr_miss={stats['asr_miss']}"
    )


def _guess_default_meta(input_path: str) -> str:
    """Guess the default MELD metadata CSV path from the input JSONL name.

    Rule: if the input file basename is "<split>.jsonl", try
    "<workspace>/data/MELD_audio/<split>.csv" first; otherwise fall back
    to dev.csv.
    """
    base = os.path.basename(input_path)
    split = "dev"
    m = re.match(r"^(train|dev|test)\.jsonl$", base)
    if m:
        split = m.group(1)
    # Walk up to find a "data/MELD_audio" folder next to the workspace root.
    guess = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "MELD_audio", f"{split}.csv",
        )
    )
    return guess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Optimize user-role prompt in a MELD-style JSONL dataset following "
            "the R1-Omni (arXiv:2503.21480) prompting scheme, with ASR "
            "transcript injected."
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
        "dev.r1omni_asr.jsonl",
    )
    default_meta = _guess_default_meta(default_in)

    p.add_argument("--input", "-i", default=default_in,
                   help=f"Input JSONL path. default={default_in}")
    p.add_argument("--output", "-o", default=default_out,
                   help=f"Output JSONL path. default={default_out}")
    p.add_argument("--meta", "-m", default=default_meta,
                   help=(
                       "MELD metadata CSV path with columns "
                       "[Dialogue_ID, Utterance_ID, Utterance, ...]. "
                       f"default={default_meta}"
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process(args.input, args.output, args.meta)


if __name__ == "__main__":
    main()
