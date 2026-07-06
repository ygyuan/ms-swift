#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan audio duration / estimated encoder-token distribution across a
StepAudio2 GRPO training jsonl (default: ``train_stratified_v7.jsonl``).

Motivation
----------
GRPO run v7-20260702-153719 crashed at step 2 with::

    RuntimeError: CUDA driver error: invalid argument
    (in transformers/models/qwen2/modeling_qwen2.py :: eager_attention_forward
     -> nn.functional.softmax)

The prime suspect is a single over-long audio in the newly-generated
``train_stratified_v7.jsonl`` that inflates the qwen2 eager-attention
attention-weight tensor beyond what the CUDA kernel launch parameters can
address (grid.x * block.x saturates when H * L crosses 65535, or
[B, H, L, L] softmax hits int32 element-count issues at very large L).
Whatever the exact overflow, the fix path is the same: find the outlier
samples first, then either drop them or lower ``MAX_LENGTH``.

What this script does
---------------------
1. Read a jsonl (rows shaped like the ones produced by
   ``balance_train_jsonl.py``: ``{audios:[path], label:str, key:str, ...}``).
2. For every audio path referenced (dedup'd), probe its duration in seconds.
   * Tries ``soundfile.info`` first (fast, no decode, works on wav/flac).
   * Falls back to ``librosa.get_duration(path=...)`` (slower, mp3/ogg).
   * Final fallback: WAV RIFF header parse (zero-dep, wav-only).
3. Convert duration -> estimated encoder token count using StepAudio2's
   audio encoder rate (default 25 tokens/sec, matches Step-Audio-2-mini's
   50 Hz mel -> 25 Hz semantic downsample; override via
   ``--tokens-per-sec``).
4. Add a fixed ``--prompt-tokens`` overhead (default 96) representing the
   text prompt + special tokens; the effective sequence length ``L`` fed
   into eager attention is ``est_audio_tokens + prompt_tokens``.
5. Report per-class stats (count, mean/median/p95/p99/max seconds and
   estimated L) plus a global histogram, and print the top-K longest rows.
6. Emit warnings when any row's estimated ``L`` exceeds ``--max-length``
   (default 3072, matches the training script) or ``--soft-limit``
   (default 2048, the value we recommend downgrading MAX_LENGTH to for
   the eager-attention path).
7. Optional ``--emit-safe-output``: write a filtered jsonl that drops
   rows whose estimated ``L`` >= ``--drop-threshold`` (default = value of
   ``--soft-limit``); the resulting file can be plugged straight into
   ``TRAIN_JSONL=... bash run_train_grpo.sh`` for a re-run.
8. Optional ``--emit-report-json``: dump full per-row records
   (path, seconds, est_L, label) so downstream tools can visualize.

Usage
-----
    # Sanity scan (fast, uses soundfile only):
    python project/stepaudio/tools/scan_audio_lengths.py \
        -i project/stepaudio/data/train_stratified_v7.jsonl

    # Full parallel scan with librosa fallback and safe-output emission:
    python project/stepaudio/tools/scan_audio_lengths.py \
        -i project/stepaudio/data/train_stratified_v7.jsonl \
        --workers 16 --use-librosa \
        --emit-safe-output project/stepaudio/data/train_stratified_v7_safe.jsonl \
        --drop-threshold 2048

    # Report top-30 longest audios and dump a per-row json:
    python project/stepaudio/tools/scan_audio_lengths.py \
        -i project/stepaudio/data/train_stratified_v7.jsonl \
        --top 30 --emit-report-json /tmp/v7_audio_lengths.json

Notes on the token rate
-----------------------
StepAudio2's audio front-end downsamples 16 kHz mel features to a
25 Hz representation before entering the LLM decoder, so
``tokens_per_sec ≈ 25`` is a safe upper-bound estimate.  If your fork
uses a different rate (e.g. 12.5 or 50), pass ``--tokens-per-sec`` to
match; the *ranking* of samples by length is invariant to this scalar
so the outlier-hunting still works.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Duration probing
# --------------------------------------------------------------------------- #

def _duration_wav_header(path: str) -> Optional[float]:
    """Zero-dep RIFF-WAV header parse. Returns seconds or None."""
    try:
        with open(path, "rb") as f:
            header = f.read(44)
        if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        # scan chunks to find "fmt " and "data"
        # standard 44-byte layout: fmt at offset 12, data at offset 36
        fmt_id = header[12:16]
        if fmt_id != b"fmt ":
            return None
        num_channels = struct.unpack("<H", header[22:24])[0]
        sample_rate = struct.unpack("<I", header[24:28])[0]
        bits_per_sample = struct.unpack("<H", header[34:36])[0]
        # data chunk size might not be at fixed 40; but for typical training
        # data (fixed-length clips) the standard offset works. Fallback: use
        # file size minus 44 as an approximation.
        data_id = header[36:40]
        if data_id == b"data":
            data_size = struct.unpack("<I", header[40:44])[0]
        else:
            data_size = os.path.getsize(path) - 44
        if sample_rate <= 0 or num_channels <= 0 or bits_per_sample <= 0:
            return None
        bytes_per_sample = bits_per_sample // 8
        num_frames = data_size / (num_channels * bytes_per_sample)
        return num_frames / sample_rate
    except Exception:
        return None


def _duration_soundfile(path: str) -> Optional[float]:
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def _duration_librosa(path: str) -> Optional[float]:
    try:
        import librosa  # type: ignore
        return float(librosa.get_duration(path=path))
    except Exception:
        return None


def probe_duration(path: str, use_librosa: bool) -> Optional[float]:
    """Try soundfile -> WAV header -> librosa (if enabled)."""
    d = _duration_soundfile(path)
    if d is not None and d > 0:
        return d
    d = _duration_wav_header(path)
    if d is not None and d > 0:
        return d
    if use_librosa:
        d = _duration_librosa(path)
        if d is not None and d > 0:
            return d
    return None


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] line {i} JSON decode failed: {e}", file=sys.stderr)
    return rows


def extract_audio_paths(row: dict) -> List[str]:
    """Rows use ``audios: [path, ...]``; a couple of rows may use ``audio``."""
    if "audios" in row and isinstance(row["audios"], (list, tuple)):
        return [str(p) for p in row["audios"] if p]
    if "audio" in row:
        v = row["audio"]
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(p) for p in v if p]
    return []


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    k = (len(xs_sorted) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs_sorted[int(k)]
    return xs_sorted[lo] * (hi - k) + xs_sorted[hi] * (k - lo)


def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", "-i", required=True, help="input training jsonl")
    ap.add_argument("--label-key", default="label", help="row field for class label (default: label)")
    ap.add_argument("--tokens-per-sec", type=float, default=25.0,
                    help="estimated audio-encoder token rate (default 25 Hz, StepAudio2-mini)")
    ap.add_argument("--prompt-tokens", type=int, default=96,
                    help="estimated fixed prompt overhead in tokens (default 96)")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="hard MAX_LENGTH used at training time (default 4096, matches run_train_grpo.sh)")
    ap.add_argument("--soft-limit", type=int, default=2048,
                    help="soft warning threshold for est_L (default 2048; the value we recommend "
                         "downgrading MAX_LENGTH to when eager-attention triggers CUDA errors)")
    ap.add_argument("--top", type=int, default=20, help="print top-K longest samples (default 20)")
    ap.add_argument("--workers", type=int, default=8, help="parallel probe workers (default 8)")
    ap.add_argument("--use-librosa", action="store_true",
                    help="allow librosa fallback for non-wav formats (slower, but more robust)")
    ap.add_argument("--emit-safe-output", default="",
                    help="if set, write a filtered jsonl dropping rows with est_L >= --drop-threshold")
    ap.add_argument("--drop-threshold", type=int, default=0,
                    help="est_L threshold for filtering into --emit-safe-output "
                         "(default 0 -> use --soft-limit)")
    ap.add_argument("--emit-report-json", default="",
                    help="if set, dump full per-row records (path, sec, est_L, label) as json")
    ap.add_argument("--limit", type=int, default=0,
                    help="only scan the first N rows (debug convenience; 0 = all)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"[FATAL] input not found: {args.input}")

    print(f"[INFO] loading {args.input} ...")
    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[INFO] --limit {args.limit} -> only scanning first {len(rows)} rows")
    print(f"[INFO] loaded {len(rows)} rows")

    # Build (row_idx, path) pairs; dedup path lookup so we probe each unique
    # audio file at most once even if a row appears multiple times after
    # up-sampling (the balancer copies rows verbatim, so paths repeat).
    row_paths: List[Tuple[int, str]] = []
    unique_paths: Dict[str, None] = {}
    for i, r in enumerate(rows):
        for p in extract_audio_paths(r):
            row_paths.append((i, p))
            unique_paths.setdefault(p, None)
    n_unique = len(unique_paths)
    n_refs = len(row_paths)
    print(f"[INFO] {n_refs} audio references over {n_unique} unique files")

    # Probe unique files in parallel.
    print(f"[INFO] probing durations (workers={args.workers}, use_librosa={args.use_librosa}) ...")
    durations: Dict[str, Optional[float]] = {}
    missing: List[str] = []

    def _probe(p: str) -> Tuple[str, Optional[float], bool]:
        exists = os.path.isfile(p)
        if not exists:
            return (p, None, False)
        d = probe_duration(p, use_librosa=args.use_librosa)
        return (p, d, True)

    done = 0
    step = max(1, n_unique // 20)
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for p, d, exists in ex.map(_probe, unique_paths.keys()):
            durations[p] = d if exists else None
            if not exists:
                missing.append(p)
            done += 1
            if done % step == 0:
                print(f"       probed {done}/{n_unique}", file=sys.stderr)

    n_missing = len(missing)
    n_probefail = sum(1 for p, d in durations.items()
                      if d is None and p not in set(missing))
    if n_missing:
        print(f"[WARN] {n_missing} audio files missing on disk (will be excluded from stats)")
        for p in missing[:10]:
            print(f"       missing: {p}")
        if n_missing > 10:
            print(f"       ... ({n_missing - 10} more)")
    if n_probefail:
        print(f"[WARN] {n_probefail} audio files failed to probe (bad header?). "
              f"Consider --use-librosa.")

    # Per-row records with derived est_L.
    records: List[dict] = []
    for i, r in enumerate(rows):
        label = str(r.get(args.label_key, "?"))
        key = str(r.get("key", ""))
        paths = extract_audio_paths(r)
        if not paths:
            continue
        # A row may have >1 audio; sum durations to bound sequence length.
        secs_list = [durations.get(p) for p in paths]
        if any(s is None for s in secs_list):
            continue  # skip rows we couldn't fully measure
        secs = float(sum(secs_list))  # type: ignore[arg-type]
        est_audio_tok = secs * args.tokens_per_sec
        est_L = int(round(est_audio_tok + args.prompt_tokens))
        records.append({
            "row_idx": i,
            "label": label,
            "key": key,
            "paths": paths,
            "seconds": secs,
            "est_audio_tokens": est_audio_tok,
            "est_L": est_L,
        })

    n_ok = len(records)
    print(f"[INFO] measured {n_ok}/{len(rows)} rows ({n_ok / max(1, len(rows)) * 100:.2f}%)")

    if not records:
        raise SystemExit("[FATAL] no rows measured -- cannot report anything")

    # -------- Global stats --------
    all_secs = [r["seconds"] for r in records]
    all_L = [r["est_L"] for r in records]
    print()
    print("=" * 78)
    print(f"Global stats  (tokens_per_sec={args.tokens_per_sec}, "
          f"prompt_tokens={args.prompt_tokens})")
    print("=" * 78)
    s = summarize(all_secs)
    print(f"  duration_sec   n={s['n']:>6}  mean={s['mean']:6.2f}  "
          f"p50={s['p50']:6.2f}  p95={s['p95']:6.2f}  p99={s['p99']:6.2f}  "
          f"max={s['max']:7.2f}")
    sL = summarize([float(x) for x in all_L])
    print(f"  est_L (tokens) n={sL['n']:>6}  mean={sL['mean']:6.0f}  "
          f"p50={sL['p50']:6.0f}  p95={sL['p95']:6.0f}  p99={sL['p99']:6.0f}  "
          f"max={sL['max']:7.0f}")

    # -------- Per-class stats --------
    per_class: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in records:
        per_class[r["label"]].append(r)
    print()
    print("=" * 78)
    print("Per-class duration / est_L")
    print("=" * 78)
    print(f"  {'label':<10} {'n':>6}  {'mean_s':>7} {'p95_s':>7} {'p99_s':>7} "
          f"{'max_s':>7}   {'mean_L':>7} {'p95_L':>7} {'p99_L':>7} {'max_L':>7}")
    print("  " + "-" * 76)
    for k in sorted(per_class, key=lambda x: -len(per_class[x])):
        secs = [x["seconds"] for x in per_class[k]]
        Ls = [float(x["est_L"]) for x in per_class[k]]
        ss = summarize(secs)
        sl = summarize(Ls)
        print(f"  {k:<10} {ss['n']:>6}  {ss['mean']:>7.2f} {ss['p95']:>7.2f} "
              f"{ss['p99']:>7.2f} {ss['max']:>7.2f}   "
              f"{sl['mean']:>7.0f} {sl['p95']:>7.0f} {sl['p99']:>7.0f} "
              f"{sl['max']:>7.0f}")

    # -------- Threshold checks --------
    over_hard = [r for r in records if r["est_L"] > args.max_length]
    over_soft = [r for r in records if r["est_L"] > args.soft_limit]
    print()
    print("=" * 78)
    print(f"Threshold flags")
    print("=" * 78)
    print(f"  est_L > MAX_LENGTH ({args.max_length}) : {len(over_hard):>6} rows "
          f"({len(over_hard) / n_ok * 100:.2f}%)")
    print(f"  est_L > soft_limit ({args.soft_limit}) : {len(over_soft):>6} rows "
          f"({len(over_soft) / n_ok * 100:.2f}%)")

    if over_hard:
        print(f"\n[!] {len(over_hard)} rows are LIKELY to be truncated/dropped by "
              f"MAX_LENGTH={args.max_length}.")
        print(f"    Under truncation_strategy='delete' they are silently removed;")
        print(f"    under 'right' they lose the tail, which for audio classification")
        print(f"    means the model may see incomplete acoustic evidence.")
    if over_soft and not over_hard:
        print(f"\n[!] {len(over_soft)} rows are safely within MAX_LENGTH={args.max_length}")
        print(f"    but exceed soft_limit={args.soft_limit}. If you hit CUDA")
        print(f"    'invalid argument' errors in eager attention, consider lowering")
        print(f"    MAX_LENGTH to {args.soft_limit}.")

    # -------- Top-K longest --------
    print()
    print("=" * 78)
    print(f"Top-{args.top} longest audio rows (by est_L)")
    print("=" * 78)
    top = sorted(records, key=lambda r: -r["est_L"])[: args.top]
    for r in top:
        p_short = r["paths"][0]
        if len(p_short) > 70:
            p_short = "..." + p_short[-67:]
        print(f"  row={r['row_idx']:<6} label={r['label']:<7} "
              f"sec={r['seconds']:7.2f}  est_L={r['est_L']:>6}  {p_short}")

    # -------- Missing / probe-failed rows --------
    if missing:
        print()
        print(f"[INFO] {len(missing)} audio files could not be found; the first 5:")
        for p in missing[:5]:
            print(f"       {p}")

    # -------- Emit safe output --------
    if args.emit_safe_output:
        drop_thr = args.drop_threshold if args.drop_threshold > 0 else args.soft_limit
        # Keep set: rows we successfully measured AND est_L < drop_thr.
        # Rows we could NOT measure are kept by default (conservative -- don't
        # silently drop training data due to a probe hiccup); emit an [INFO]
        # so the user knows.
        measured_row_est: Dict[int, int] = {r["row_idx"]: r["est_L"] for r in records}
        kept = 0
        dropped = 0
        unmeasured_kept = 0
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_safe_output)) or ".",
                    exist_ok=True)
        tmp = args.emit_safe_output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                estL = measured_row_est.get(i)
                if estL is None:
                    unmeasured_kept += 1
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if estL >= drop_thr:
                    dropped += 1
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        os.replace(tmp, args.emit_safe_output)
        print()
        print(f"[OK  ] wrote safe-output {args.emit_safe_output}: "
              f"kept={kept} dropped={dropped} (drop_threshold est_L>={drop_thr}); "
              f"of the kept rows, {unmeasured_kept} were retained without measurement.")

    # -------- Emit per-row json --------
    if args.emit_report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_report_json)) or ".",
                    exist_ok=True)
        payload = {
            "input": os.path.abspath(args.input),
            "tokens_per_sec": args.tokens_per_sec,
            "prompt_tokens": args.prompt_tokens,
            "max_length": args.max_length,
            "soft_limit": args.soft_limit,
            "n_rows": len(rows),
            "n_measured": n_ok,
            "n_missing_files": n_missing,
            "records": records,
        }
        with open(args.emit_report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK  ] wrote per-row report -> {args.emit_report_json}")


if __name__ == "__main__":
    main()
