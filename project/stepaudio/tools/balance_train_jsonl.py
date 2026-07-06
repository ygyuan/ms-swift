#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balance train.jsonl by upsampling minority classes.

Motivation
----------
The original ``train.jsonl`` for the StepAudio2 classifier is heavily skewed:

    speech  39066 (69.39%)
    noise    9231 (16.40%)
    music    3217 ( 5.71%)
    porn     2725 ( 4.84%)
    song     2063 ( 3.66%)

GRPO with G=4 and per-device-batch=1 samples ~4-32 prompts per optimizer step.
With <5% porn/song/music, entire batches contain no minority-class prompts, so
the policy has no signal to learn them and quickly collapses to always
predicting "speech" (the majority-class attractor). We observed exactly this in
run v0-20260630-141430 (porn/song/music recall = 0 at ckpt-2800) and again in
v2-20260701-105056 (0 porn GT sampled across 374 steps).

Strategy (this script)
----------------------
1. Group rows by ``label``.
2. Upsample every non-majority class to ``target_count`` via *replacement*
   sampling with fixed seed (deterministic, reproducible).
3. Optionally cap the majority class (``--cap-majority``) to reduce total size.
4. Global shuffle so that adjacent rows in the resulting jsonl come from
   different classes (crucial for GRPO: adjacent rows form the same DP batch
   under ``per_device_train_batch_size=1`` + gradient accumulation).

Output rows are byte-identical copies of the source rows (no field mutation),
just with different multiplicities.

Usage
-----
    python balance_train_jsonl.py \
        --input  project/stepaudio/data/train.jsonl \
        --output project/stepaudio/data/train_balanced.jsonl \
        --target max            # upsample all classes to majority count (default)

    # Or cap majority to 20000 and upsample others to 20000 as well:
    python balance_train_jsonl.py --target 20000 --cap-majority 20000

    # Or match to median class size (compromise between size and balance):
    python balance_train_jsonl.py --target median

    # Or fully custom per-class targets (v6 recommended, based on v4 analysis):
    python balance_train_jsonl.py \
        --custom-target 'speech=20000,noise=12000,music=10000,porn=12000,song=8000' \
        -i data/train.jsonl -o data/train_balanced_v2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# ---- Optional integration with scan_audio_lengths.py (same tools/ dir) ---- #
# We import lazily so this script keeps working when the sibling module is
# missing (e.g. minimal deployments). The pre-filter feature simply degrades
# to a no-op with a clear warning if the import fails.
try:
    # add current dir to path so `from scan_audio_lengths import ...` works
    # regardless of how this script is invoked.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    from scan_audio_lengths import (  # type: ignore
        probe_duration as _probe_duration,
        extract_audio_paths as _extract_audio_paths,
    )
    _HAS_SCAN = True
except Exception as _e:  # pragma: no cover -- keep balancing usable regardless
    _HAS_SCAN = False
    _SCAN_IMPORT_ERR = _e


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


def group_by_label(rows: List[dict], label_key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = collections.defaultdict(list)
    missing = 0
    for r in rows:
        lab = r.get(label_key)
        if lab is None:
            missing += 1
            continue
        groups[str(lab)].append(r)
    if missing:
        print(f"[WARN] {missing} rows missing '{label_key}' -- skipped", file=sys.stderr)
    return groups


def resolve_target(target_arg: str, sizes: Dict[str, int]) -> int:
    """Resolve --target (max/median/mean/<int>) into a concrete count."""
    if target_arg == "max":
        return max(sizes.values())
    if target_arg == "median":
        vals = sorted(sizes.values())
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) // 2
    if target_arg == "mean":
        return sum(sizes.values()) // len(sizes)
    try:
        return int(target_arg)
    except ValueError:
        raise SystemExit(
            f"[FATAL] --target must be one of {{max,median,mean,<int>}}, got '{target_arg}'"
        )


def resample(pool: List[dict], target: int, rng: random.Random) -> List[dict]:
    """Upsample or downsample pool -> exactly target rows.

    * Upsample: keep all + random-with-replacement fill.
    * Downsample: random-without-replacement subset.
    """
    n = len(pool)
    if target == n:
        return list(pool)
    if target < n:
        return rng.sample(pool, target)
    # target > n: keep every original once, then sample (target - n) with replacement
    extra = [rng.choice(pool) for _ in range(target - n)]
    return list(pool) + extra


def _apply_pre_filter(
    rows: List[dict],
    max_est_L: int,
    tokens_per_sec: float,
    prompt_tokens: int,
    workers: int,
    use_librosa: bool,
    drop_unmeasured: bool,
    label_key: str,
) -> List[dict]:
    """Drop rows whose estimated sequence length exceeds ``max_est_L``.

    est_L = prompt_tokens + sum(duration_of_audio) * tokens_per_sec

    Rows with unmeasurable audio (file missing / decode failed) are kept by
    default (conservative) unless ``drop_unmeasured`` is True.

    Prints a per-class summary of what would be kept vs dropped, so the user
    can quickly detect situations where the pre-filter would gut a class
    (e.g. "song" is dominated by 30s clips).
    """
    import concurrent.futures as cf

    if not _HAS_SCAN:
        print(
            f"[WARN] --pre-filter-est-L requested but scan_audio_lengths.py "
            f"could not be imported ({_SCAN_IMPORT_ERR!r}); pre-filter "
            f"skipped -- returning all rows unchanged.",
            file=sys.stderr,
        )
        return rows

    # Collect unique audio paths (up-sampled / duplicated rows share files).
    unique_paths: Dict[str, None] = {}
    for r in rows:
        for p in _extract_audio_paths(r):
            unique_paths.setdefault(p, None)
    n_unique = len(unique_paths)
    print(
        f"[INFO] pre-filter: probing {n_unique} unique audio files "
        f"(threshold est_L>{max_est_L}, tokens/sec={tokens_per_sec}, "
        f"prompt_tokens={prompt_tokens}, workers={workers}, "
        f"use_librosa={use_librosa}) ..."
    )

    def _probe_one(p: str) -> Tuple[str, Optional[float], bool]:
        exists = os.path.isfile(p)
        if not exists:
            return (p, None, False)
        d = _probe_duration(p, use_librosa=use_librosa)
        return (p, d, True)

    durations: Dict[str, Optional[float]] = {}
    n_missing = 0
    n_probefail = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for p, d, exists in ex.map(_probe_one, list(unique_paths.keys())):
            if not exists:
                n_missing += 1
                durations[p] = None
                continue
            if d is None:
                n_probefail += 1
            durations[p] = d
    if n_missing:
        print(f"[WARN] pre-filter: {n_missing} audio files missing on disk")
    if n_probefail:
        print(f"[WARN] pre-filter: {n_probefail} audio files failed to probe "
              f"(consider --pre-filter-use-librosa)")

    kept: List[dict] = []
    kept_stats: collections.Counter = collections.Counter()
    dropped_stats: collections.Counter = collections.Counter()
    unmeasured_stats: collections.Counter = collections.Counter()
    for r in rows:
        lab = str(r.get(label_key, "?"))
        paths = _extract_audio_paths(r)
        secs_list = [durations.get(p) for p in paths] if paths else []
        # If we cannot fully measure the row, apply drop_unmeasured policy.
        if paths and any(s is None for s in secs_list):
            unmeasured_stats[lab] += 1
            if drop_unmeasured:
                dropped_stats[lab] += 1
                continue
            kept.append(r)
            kept_stats[lab] += 1
            continue
        secs = float(sum(secs_list)) if secs_list else 0.0
        est_L = int(round(secs * tokens_per_sec + prompt_tokens))
        if est_L > max_est_L:
            dropped_stats[lab] += 1
            continue
        kept.append(r)
        kept_stats[lab] += 1

    all_labels = sorted(set(list(kept_stats) + list(dropped_stats)))
    print("[INFO] pre-filter per-class summary:")
    print(f"       {'label':<10} {'kept':>8} {'dropped':>8} {'unmeasured':>11}")
    for lab in all_labels:
        print(
            f"       {lab:<10} {kept_stats[lab]:>8} "
            f"{dropped_stats[lab]:>8} {unmeasured_stats[lab]:>11}"
        )
    total_dropped = sum(dropped_stats.values())
    print(
        f"[INFO] pre-filter total: {sum(kept_stats.values())} kept / "
        f"{total_dropped} dropped "
        f"({total_dropped / max(1, len(rows)) * 100:.2f}% of input)"
    )
    # Emit an actionable warning if any class lost a large fraction:
    for lab in all_labels:
        drop_n = dropped_stats[lab]
        base_n = drop_n + kept_stats[lab]
        if base_n > 0 and drop_n / base_n >= 0.30:
            print(
                f"[WARN] pre-filter dropped {drop_n}/{base_n} "
                f"({drop_n / base_n * 100:.1f}%) rows of class '{lab}' -- "
                f"the resampler will now up-sample from a much smaller pool, "
                f"increasing memorization risk. Consider raising "
                f"--pre-filter-est-L or shortening long audios upstream."
            )
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, help="input train.jsonl")
    ap.add_argument("--output", "-o", required=True, help="output train_balanced.jsonl")
    ap.add_argument("--label-key", default="label", help="row field to group by (default: label)")
    ap.add_argument(
        "--target",
        default="max",
        help="target count per class: 'max' (default, = majority class size), "
             "'median', 'mean', or an explicit integer",
    )
    ap.add_argument(
        "--custom-target",
        default="",
        help="per-class target overrides, comma-separated 'label=count' pairs "
             "(e.g. 'speech=20000,noise=12000,music=10000,porn=12000,song=8000'). "
             "Any class not listed here falls back to --target. Overrides win "
             "over --target and --cap-majority whenever a class appears here.",
    )
    ap.add_argument(
        "--cap-majority",
        type=int,
        default=0,
        help="if >0, downsample the largest class to this count (applied BEFORE resolving 'max' target)",
    )
    ap.add_argument(
        "--min-only",
        action="store_true",
        help="only upsample classes strictly smaller than target; leave equal-or-larger classes untouched",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="just print the plan, do not write output")
    ap.add_argument(
        "--stratified-interleave",
        action="store_true",
        help="After per-class resampling, DO NOT do a global shuffle; instead, "
             "shuffle each class internally and then round-robin interleave "
             "them so that every K consecutive rows (K = number of classes) "
             "covers all classes exactly once. This makes every GRPO "
             "gradient-accumulation window (world_size * grad_accum rows) "
             "contain the full class set as long as that window is a multiple "
             "of K. Strongly recommended when using GRPO with per-device "
             "batch=1 -- see v6 post-mortem in the header docstring.",
    )
    ap.add_argument(
        "--stratified-cycle",
        default="",
        help="Custom cycle order for --stratified-interleave (comma-separated "
             "labels; a label may repeat to weight it more heavily). Example: "
             "'speech,noise,speech,music,porn,song' produces 6-step cycles "
             "with speech oversampled 2x per cycle. Defaults to alphabetical "
             "class order when empty.",
    )
    # ---------------- Pre-filter: drop over-long audios BEFORE resampling ---------- #
    # Rationale (v7 post-mortem, run v7-20260702-153719):
    #   The eager-attention path in Qwen2 (used by Step-Audio-2-mini because
    #   attn_impl='eager' is currently its only supported value) allocates
    #   an [B, H, L, L] softmax tensor whose CUDA kernel launch parameters
    #   overflow when L is too large. In practice we saw a step-2 crash with
    #   'RuntimeError: CUDA driver error: invalid argument' on a run whose
    #   MAX_LENGTH was 3072. Filtering the source jsonl BEFORE the balancer
    #   copies rows means minority-class up-sampling won't multiply the same
    #   over-long audio 4-19x into the training set, and the resulting file
    #   is safe to feed to run_train_grpo.sh at any MAX_LENGTH >= threshold.
    ap.add_argument(
        "--pre-filter-est-L",
        type=int,
        default=0,
        help="if >0, drop rows whose estimated sequence length "
             "(prompt_tokens + duration * tokens_per_sec) EXCEEDS this value "
             "BEFORE resampling. Recommended: 2048 for eager-attention safety, "
             "3072 to just match run_train_grpo.sh MAX_LENGTH. 0 = disabled.",
    )
    ap.add_argument(
        "--pre-filter-tokens-per-sec",
        type=float,
        default=25.0,
        help="tokens/sec used by --pre-filter-est-L (default 25 Hz, matches "
             "Step-Audio-2-mini's semantic-token rate)",
    )
    ap.add_argument(
        "--pre-filter-prompt-tokens",
        type=int,
        default=96,
        help="fixed prompt-token overhead added to the estimated length "
             "(default 96)",
    )
    ap.add_argument(
        "--pre-filter-workers",
        type=int,
        default=8,
        help="parallel probe workers for --pre-filter-est-L (default 8)",
    )
    ap.add_argument(
        "--pre-filter-use-librosa",
        action="store_true",
        help="allow librosa fallback when soundfile / WAV-header probe fails "
             "(slower but robust for mp3/ogg)",
    )
    ap.add_argument(
        "--pre-filter-keep-unmeasured",
        action="store_true",
        help="keep rows whose audio duration could not be probed (missing file "
             "or bad header). Default is to KEEP them (conservative), pass "
             "this flag to make the intent explicit; use "
             "--pre-filter-drop-unmeasured to instead drop such rows.",
    )
    ap.add_argument(
        "--pre-filter-drop-unmeasured",
        action="store_true",
        help="drop rows whose audio duration could not be probed (default is "
             "to keep them). Useful if you want a strictly-clean output.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"[FATAL] input not found: {args.input}")

    rng = random.Random(args.seed)

    print(f"[INFO] loading {args.input} ...")
    rows = load_jsonl(args.input)
    print(f"[INFO] loaded {len(rows)} rows")

    # ------------------------------------------------------------------ #
    # Pre-filter over-long audios BEFORE grouping / resampling.
    # We do this early so that:
    #   * the printed 'original class distribution' reflects what the
    #     downstream balancer will actually see,
    #   * cap_majority / target resolution operate on the trimmed pool,
    #   * and rare-class up-sampling never multiplies a long-audio row.
    # ------------------------------------------------------------------ #
    if args.pre_filter_est_L and args.pre_filter_est_L > 0:
        rows = _apply_pre_filter(
            rows,
            max_est_L=args.pre_filter_est_L,
            tokens_per_sec=args.pre_filter_tokens_per_sec,
            prompt_tokens=args.pre_filter_prompt_tokens,
            workers=args.pre_filter_workers,
            use_librosa=args.pre_filter_use_librosa,
            drop_unmeasured=args.pre_filter_drop_unmeasured,
            label_key=args.label_key,
        )
        print(f"[INFO] after pre-filter: {len(rows)} rows survive")

    groups = group_by_label(rows, args.label_key)
    orig_sizes = {k: len(v) for k, v in groups.items()}
    print("[INFO] original class distribution:")
    total_orig = sum(orig_sizes.values())
    for k in sorted(orig_sizes, key=lambda x: -orig_sizes[x]):
        v = orig_sizes[k]
        print(f"       {k:<10} {v:>7}  ({v / total_orig * 100:5.2f}%)")

    # Optional: cap majority BEFORE target resolution so 'max' takes the capped value.
    if args.cap_majority > 0:
        maj_key = max(orig_sizes, key=lambda k: orig_sizes[k])
        if orig_sizes[maj_key] > args.cap_majority:
            print(f"[INFO] capping majority class '{maj_key}' from {orig_sizes[maj_key]} -> {args.cap_majority}")
            groups[maj_key] = rng.sample(groups[maj_key], args.cap_majority)
            orig_sizes[maj_key] = args.cap_majority

    target = resolve_target(args.target, orig_sizes)
    print(f"[INFO] resolved target per class = {target}  (mode='{args.target}')")

    # Parse --custom-target overrides ('label=N,label=N,...').
    # This lets us pick a *per-class* target instead of forcing all classes to
    # the same size. Motivation (v6, based on v4 analysis):
    #   * 'max' balancing (39066 x 5 = 195330 rows) upsamples porn 14x and
    #     song 19x -> risk of memorizing the tiny minority pool + tripling
    #     epoch time; also creates a sharp train/val distribution mismatch
    #     (train porn 20% vs val porn 5%).
    #   * A gentler distribution like {speech=20000, noise=12000, music=10000,
    #     porn=12000, song=8000} keeps rare classes at ~4x upsampling (still
    #     enough for GRPO to see them 5x per epoch after --dynamic_sample=false)
    #     while capping speech to reduce majority dominance without erasing it.
    custom_targets: Dict[str, int] = {}
    if args.custom_target.strip():
        for part in args.custom_target.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"[FATAL] --custom-target token missing '=' : {part!r}")
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                custom_targets[k] = int(v)
            except ValueError:
                raise SystemExit(f"[FATAL] --custom-target value not int : {part!r}")
        # Warn on unknown labels (typo protection).
        unknown = [k for k in custom_targets if k not in orig_sizes]
        if unknown:
            print(f"[WARN] --custom-target contains labels not present in dataset: {unknown}")
        print(f"[INFO] custom per-class overrides: {custom_targets}")

    balanced: List[dict] = []
    print("[INFO] resampling plan:")
    for k in sorted(orig_sizes, key=lambda x: -orig_sizes[x]):
        n = orig_sizes[k]
        # Per-class override takes precedence; else fall back to global target.
        eff_target = custom_targets.get(k, target)
        if args.min_only and n >= eff_target:
            new_n = n
            action = "keep-as-is"
        else:
            new_n = eff_target
            if new_n > n:
                action = f"upsample x{new_n / n:.2f}"
            elif new_n < n:
                action = f"downsample x{new_n / n:.2f}"
            else:
                action = "unchanged"
        src = "custom" if k in custom_targets else "target"
        print(f"       {k:<10} {n:>7} -> {new_n:>7}   [{action}, from={src}]")
        balanced.extend(resample(groups[k], new_n, rng))

    print(f"[INFO] balanced total = {len(balanced)} rows (was {total_orig})")

    # ------------------------------------------------------------------ #
    # Ordering strategy:
    #   * default : global shuffle (adjacent rows random -- historical behavior)
    #   * --stratified-interleave : per-class shuffle + round-robin interleave
    #     so every K consecutive rows cover all K classes exactly once
    #     (or follow --stratified-cycle if given).
    # ------------------------------------------------------------------ #
    if args.stratified_interleave:
        # Group the (already-resampled) rows back by label.
        by_label = group_by_label(balanced, args.label_key)
        for k in by_label:
            rng.shuffle(by_label[k])

        if args.stratified_cycle.strip():
            cycle = [x.strip() for x in args.stratified_cycle.split(",") if x.strip()]
            unknown = [c for c in cycle if c not in by_label]
            if unknown:
                raise SystemExit(
                    f"[FATAL] --stratified-cycle references unknown labels: {unknown}. "
                    f"Available: {sorted(by_label.keys())}"
                )
        else:
            cycle = sorted(by_label.keys())

        # Round-robin: consume one row per label per cycle step until any
        # class runs out; stop there so we don't emit rows outside the
        # promised interleaving contract. This also naturally trims to a
        # multiple of len(cycle), which is what GRPO wants.
        interleaved: List[dict] = []
        cursors = {k: 0 for k in by_label}
        while True:
            # Check every label appearing in this cycle has at least one
            # remaining row (accounting for repeated labels in the cycle).
            need = collections.Counter(cycle)
            can_go = all(cursors[k] + need[k] <= len(by_label[k]) for k in need)
            if not can_go:
                break
            for k in cycle:
                interleaved.append(by_label[k][cursors[k]])
                cursors[k] += 1

        dropped = len(balanced) - len(interleaved)
        if dropped > 0:
            # Show which classes are the bottleneck so users can retune targets.
            leftover = {k: len(by_label[k]) - cursors[k] for k in by_label}
            print(f"[INFO] stratified-interleave dropped {dropped} tail rows "
                  f"to keep exact class-per-cycle contract (cycle={cycle}). "
                  f"Leftover per class: {leftover}")
        balanced = interleaved
        print(f"[INFO] stratified-interleave: {len(balanced)} rows arranged in "
              f"{len(balanced) // len(cycle)} cycles of {len(cycle)} rows each "
              f"(cycle order = {cycle})")
    else:
        # Global shuffle so that adjacent rows come from mixed classes.
        rng.shuffle(balanced)

    # Sanity-check final distribution
    final_dist = collections.Counter(str(r.get(args.label_key)) for r in balanced)
    print("[INFO] final class distribution:")
    tot = sum(final_dist.values())
    for k in sorted(final_dist, key=lambda x: -final_dist[x]):
        v = final_dist[k]
        print(f"       {k:<10} {v:>7}  ({v / tot * 100:5.2f}%)")

    if args.dry_run:
        print("[INFO] --dry-run set, skipping write.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(f"[OK  ] wrote {args.output}")


if __name__ == "__main__":
    main()
