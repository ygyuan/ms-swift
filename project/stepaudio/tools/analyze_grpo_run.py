#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze a GRPO run's ``completions.jsonl`` to track per-class recall over
training steps -- primary use is early detection of mode collapse.

Why this script exists
----------------------
For the StepAudio2 five-way classifier, GRPO runs like ``v0-20260630-141430`` /
``v2-20260701-105056`` collapsed to always-predict-``speech`` because the
minority classes were never sampled (see ``tools/balance_train_jsonl.py``). We
learned this only *after* running full evaluation on saved checkpoints. This
script lets us spot the problem *during* training by:

* reading the training-time samples that ms-swift dumps to
  ``<output_dir>/completions.jsonl`` (one row per optimizer step, containing
  the prompt, all G completions, and per-completion reward);
* inferring the batch's ground-truth label from the completion whose
  ``StepAudioAccuracy == 1`` (or from the ``solution`` field in the prompt);
* bucketing by step and computing a confusion matrix + per-class recall for
  each bucket.

If any minority class stays at recall == 0 for consecutive buckets, that's the
early-warning signature we want to catch.

Data schema (as of ms-swift GRPO trainer)
-----------------------------------------
Each line of ``completions.jsonl`` looks like::

    {
      "step":               [123, 123, 123, 123],   # per-completion step id
      "prompt":             [ [...chat msgs...], ... ],
      "completion":         ["speech", "song", ...],
      "StepAudioAccuracy":  [1, 0, 0, 0],
      "StepAudioFormat":    [1, 1, 1, 1],
      "advantages":         [ ...G floats... ]
    }

Usage
-----
    # One-shot summary of the whole run
    python analyze_grpo_run.py \
        --run-dir output/stepaudio/grpo/v3-YYYYMMDD-HHMMSS

    # Bucket every 200 steps, only look at the last 1000 steps
    python analyze_grpo_run.py --run-dir <run> --bucket 200 --tail 1000

    # Live mode: refresh every 60 s (Ctrl+C to stop)
    python analyze_grpo_run.py --run-dir <run> --follow --interval 60

    # Emit machine-readable outputs for plotting
    python analyze_grpo_run.py --run-dir <run> \
        --csv timeline.csv --json summary.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
LABELS_DEFAULT = ["speech", "music", "noise", "porn", "song"]
OTHER = "<other>"


# --------------------------------------------------------------------------- #
# GT extraction
# --------------------------------------------------------------------------- #
def _pick_label(text: str, labels: List[str]) -> str:
    """Return the first label found in ``text`` (lower-cased, sub-string
    match); ``OTHER`` if none matches."""
    if not text:
        return OTHER
    t = text.strip().lower()
    for lab in labels:
        if lab in t:
            return lab
    return OTHER


_SOLUTION_RE = re.compile(r'"solution"\s*:\s*"([^"]+)"')
# Match the assistant's target label if the prompt happens to include the full
# original chat template (rare, but seen when swift dumps the raw sample).
_ASSISTANT_LABEL_RE = re.compile(r'"role"\s*:\s*"assistant"\s*,\s*"content"\s*:\s*"([^"]+)"')

def _extract_gt(row: dict, labels: List[str]) -> Optional[str]:
    """Best-effort inference of the batch ground-truth label.

    Priority order (updated 2026-07-02, see analyze_grpo_run mode-collapse
    post-mortem for v0/v2 -- when *all* completions in a batch are wrong,
    reward-based inference silently loses that batch's GT and biases the
    confusion matrix. Prompt/label-based inference does not have that flaw,
    so we now try it first):
      1. Explicit fields from the top-level row -- ``label`` / ``ground_truth``
         / ``solution`` / ``answer`` / ``gt`` (fastest, always right).
      2. ``solution``/``answer`` embedded as JSON inside the prompt turns.
      3. Reward-based fallback: any completion with ``StepAudioAccuracy==1``
         (kept as a last resort for legacy runs whose completions.jsonl does
         not carry the label column).
    Returns None if nothing works (row is skipped in aggregation).
    """
    # (1) explicit top-level GT columns
    for k in ("label", "ground_truth", "gt", "solution", "answer"):
        v = row.get(k)
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, str):
            lab = _pick_label(v, labels)
            if lab != OTHER:
                return lab

    # (2) prompt-embedded solution / assistant target
    prompt = row.get("prompt")
    prompt_texts: List[str] = []
    if isinstance(prompt, list):
        for turn in prompt:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                if isinstance(content, str):
                    prompt_texts.append(content)
            elif isinstance(turn, str):
                prompt_texts.append(turn)
    elif isinstance(prompt, str):
        prompt_texts.append(prompt)
    for txt in prompt_texts:
        m = _SOLUTION_RE.search(txt)
        if m:
            lab = _pick_label(m.group(1), labels)
            if lab != OTHER:
                return lab
        m = _ASSISTANT_LABEL_RE.search(txt)
        if m:
            lab = _pick_label(m.group(1), labels)
            if lab != OTHER:
                return lab

    # (3) reward-based fallback (legacy behaviour)
    acc = row.get("StepAudioAccuracy") or []
    comps = row.get("completion")
    if isinstance(comps, str):
        comps = [comps]
    if isinstance(acc, list) and isinstance(comps, list) and len(acc) == len(comps):
        for a, c in zip(acc, comps):
            try:
                if float(a) == 1.0:
                    lab = _pick_label(c or "", labels)
                    if lab != OTHER:
                        return lab
            except (TypeError, ValueError):
                continue

    return None


def _row_step(row: dict) -> Optional[int]:
    s = row.get("step", row.get("global_step"))
    if isinstance(s, list) and s:
        s = s[0]
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
class Aggregator:
    def __init__(self, labels: List[str], bucket: int):
        self.labels = labels
        self.bucket = max(1, int(bucket))
        # (bucket_id) -> Counter[(gt, pred)]
        self.buckets: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        # per-row totals (regardless of gt)
        self.pred_dist_early: collections.Counter = collections.Counter()
        self.pred_dist_late: collections.Counter = collections.Counter()
        self.n_rows = 0
        self.n_completions = 0
        self.n_with_gt = 0
        self.min_step: Optional[int] = None
        self.max_step: Optional[int] = None
        # bucket_id -> list of per-row advantage-std values (one float per row).
        # Populated only when the row carries an ``advantages`` array.
        # Used by the advantage-collapse warning emitter -- see
        # _advantage_collapse_warnings() below for the exact heuristic.
        self.adv_std_by_bucket: Dict[int, List[float]] = collections.defaultdict(list)
        self.n_rows_with_adv = 0

    def add_row(self, row: dict) -> None:
        step = _row_step(row) or 0
        if self.min_step is None or step < self.min_step:
            self.min_step = step
        if self.max_step is None or step > self.max_step:
            self.max_step = step

        comps = row.get("completion")
        if isinstance(comps, str):
            comps = [comps]
        if not comps:
            return
        self.n_rows += 1
        self.n_completions += len(comps)

        gt = _extract_gt(row, self.labels)
        if gt is not None:
            self.n_with_gt += 1

        bkey = step // self.bucket
        for c in comps:
            pred = _pick_label(c or "", self.labels)
            if gt is not None:
                self.buckets[bkey][(gt, pred)] += 1

        # ---- Advantage-collapse tracker ----
        # ms-swift's GRPO trainer dumps per-completion advantages to the same
        # row. When std(advantages) shrinks toward zero for many rows in a
        # row, the group-relative gradient signal has effectively vanished
        # and the policy is either (a) mode-collapsed (all completions equal)
        # or (b) rewards saturated (all completions correct). Either way the
        # trainer stops learning; we log this so users can spot it early.
        adv = row.get("advantages")
        if isinstance(adv, list) and len(adv) >= 2:
            try:
                fs = [float(x) for x in adv]
                mean = sum(fs) / len(fs)
                var = sum((x - mean) ** 2 for x in fs) / len(fs)
                std = var ** 0.5
                self.adv_std_by_bucket[bkey].append(std)
                self.n_rows_with_adv += 1
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def bucket_stats(self) -> List[dict]:
        """Return a list of dicts, one per non-empty bucket, sorted by step."""
        out = []
        for bkey in sorted(self.buckets):
            cm = self.buckets[bkey]
            step_lo = bkey * self.bucket
            step_hi = step_lo + self.bucket - 1
            per_class = {}
            for gt in self.labels:
                row_total = sum(v for (g, _), v in cm.items() if g == gt)
                correct = cm.get((gt, gt), 0)
                per_class[gt] = {
                    "support": row_total,
                    "correct": correct,
                    "recall": (correct / row_total) if row_total else float("nan"),
                }
            total = sum(cm.values())
            correct_total = sum(cm.get((g, g), 0) for g in self.labels)
            out.append(
                {
                    "bucket_id": bkey,
                    "step_lo": step_lo,
                    "step_hi": step_hi,
                    "total_completions": total,
                    "accuracy": (correct_total / total) if total else float("nan"),
                    "per_class": per_class,
                    "confusion": {f"{g}->{p}": v for (g, p), v in cm.items()},
                }
            )
        return out

    def overall_confusion(self) -> Dict[Tuple[str, str], int]:
        merged: collections.Counter = collections.Counter()
        for cm in self.buckets.values():
            merged.update(cm)
        return dict(merged)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    if x != x:  # nan
        return "  n/a"
    return f"{x * 100:5.1f}%"


def render_terminal(agg: Aggregator, tail_bucket: Optional[int] = None) -> str:
    lines: List[str] = []
    labels = agg.labels
    stats = agg.bucket_stats()
    if not stats:
        return "(no bucket data)"

    if tail_bucket is not None and tail_bucket > 0:
        stats = stats[-tail_bucket:]

    # ---- Timeline table --------------------------------------------------
    lines.append("=" * 78)
    lines.append(
        f"Per-class recall timeline  (bucket={agg.bucket} steps, "
        f"steps={agg.min_step}..{agg.max_step}, "
        f"rows={agg.n_rows}, completions={agg.n_completions}, "
        f"gt-inferred={agg.n_with_gt}/{agg.n_rows})"
    )
    lines.append("=" * 78)
    header = f"{'steps':>13} {'acc':>7}  " + " ".join(f"{lab:>10}" for lab in labels)
    lines.append(header)
    lines.append("-" * len(header))
    for s in stats:
        row = [f"{s['step_lo']:>5}-{s['step_hi']:<5}", _fmt_pct(s["accuracy"])]
        for lab in labels:
            pc = s["per_class"][lab]
            if pc["support"] == 0:
                cell = f"    - / {0:<3}"
            else:
                cell = f"{_fmt_pct(pc['recall'])} /{pc['support']:>3}"
            row.append(f"{cell:>10}")
        lines.append(" ".join(row))
    lines.append("legend: recall/gt-support-in-bucket ('-' = no GT of that class in bucket)")

    # ---- Overall confusion ----------------------------------------------
    lines.append("")
    lines.append("Overall confusion (rows=inferred_gt, cols=pred, counts over completions):")
    labs_full = labels + [OTHER]
    hdr = f"  {'gt\\pred':<10} " + " ".join(f"{x:>8}" for x in labs_full) + f"   {'recall':>7}"
    lines.append(hdr)
    cm = agg.overall_confusion()
    for g in labels:
        row_counts = [cm.get((g, p), 0) for p in labs_full]
        tot = sum(row_counts)
        rec = cm.get((g, g), 0) / tot if tot else float("nan")
        lines.append(
            f"  {g:<10} "
            + " ".join(f"{v:>8}" for v in row_counts)
            + f"   {_fmt_pct(rec):>7}"
        )

    # ---- Mean-recall summary (macro) ------------------------------------
    # Macro-recall = simple average of per-class recall across all classes
    # that had at least one GT sample in the run. This is the correct
    # target metric for a mode-collapse-prone classifier (accuracy is
    # dominated by the majority class); we surface both an overall macro
    # recall AND the last-bucket macro recall so users can quickly answer
    # "is my current policy hitting mean_recall > 0.6?" without having to
    # eyeball the per-class table.
    lines.append("")
    cm = agg.overall_confusion()
    per_class_recall_all = {}
    for g in labels:
        row_counts = [cm.get((g, p), 0) for p in labs_full]
        tot = sum(row_counts)
        if tot > 0:
            per_class_recall_all[g] = cm.get((g, g), 0) / tot
    if per_class_recall_all:
        macro_all = sum(per_class_recall_all.values()) / len(per_class_recall_all)
        lines.append(f"Macro recall (whole run): {_fmt_pct(macro_all)}  "
                     f"(over {len(per_class_recall_all)}/{len(labels)} classes with GT)")
    last = stats[-1]
    last_recalls = [last["per_class"][lab]["recall"]
                    for lab in labels
                    if last["per_class"][lab]["support"] > 0]
    if last_recalls:
        macro_last = sum(last_recalls) / len(last_recalls)
        lines.append(f"Macro recall (last bucket steps {last['step_lo']}..{last['step_hi']}): "
                     f"{_fmt_pct(macro_last)}  (over {len(last_recalls)}/{len(labels)} classes)")

    # ---- Warnings --------------------------------------------------------
    lines.append("")
    warnings = _mode_collapse_warnings(stats, labels)
    warnings.extend(_advantage_collapse_warnings(agg, stats))
    if warnings:
        lines.append("[!] Mode-collapse / anomaly warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    else:
        lines.append("[OK] No obvious mode-collapse patterns detected in the last buckets.")

    if agg.n_rows_with_adv > 0:
        lines.append(f"[INFO] advantage-collapse tracker: analyzed "
                     f"{agg.n_rows_with_adv}/{agg.n_rows} rows with 'advantages' field.")
    return "\n".join(lines)


def _advantage_collapse_warnings(
    agg: "Aggregator",
    stats: List[dict],
    zero_std_eps: float = 1e-3,
    zero_frac_alarm: float = 0.7,
    look_back: int = 3,
) -> List[str]:
    """Emit warnings when a large fraction of rollout groups have
    advantage-std ~= 0 across the last few buckets. This is the earliest
    tell-tale of GRPO's group-relative signal dying.

    * ``zero_std_eps``    : below this, we count the row as "collapsed".
    * ``zero_frac_alarm`` : if the collapsed fraction stays above this in the
                            last ``look_back`` buckets, we emit a warning.
    """
    warnings: List[str] = []
    if not stats or not agg.adv_std_by_bucket:
        return warnings
    tail = stats[-min(look_back, len(stats)):]
    fracs: List[Tuple[int, float, int]] = []
    for s in tail:
        b = s["bucket_id"]
        stds = agg.adv_std_by_bucket.get(b, [])
        if not stds:
            continue
        n_zero = sum(1 for x in stds if x < zero_std_eps)
        fracs.append((b, n_zero / len(stds), len(stds)))
    if not fracs:
        return warnings
    all_high = all(f >= zero_frac_alarm for _, f, _ in fracs)
    if all_high:
        parts = [f"bucket~{b}: {f * 100:.0f}% of {n} groups have std<{zero_std_eps}"
                 for b, f, n in fracs]
        warnings.append(
            "advantage collapse: " + "; ".join(parts) +
            " -- GRPO group-relative gradient is vanishing (either all-correct "
            "saturation or all-wrong rare-class death spiral)."
        )
    else:
        # Softer heads-up: at least one bucket is bad.
        worst = max(fracs, key=lambda t: t[1])
        if worst[1] >= zero_frac_alarm:
            warnings.append(
                f"advantage warning: bucket~{worst[0]} has {worst[1] * 100:.0f}% "
                f"of {worst[2]} groups with std<{zero_std_eps}; keep an eye "
                f"on the trend before it becomes chronic."
            )
    return warnings


def _mode_collapse_warnings(stats: List[dict], labels: List[str]) -> List[str]:
    """Emit human-readable warnings when a class' recall is stuck at 0
    across the last few buckets *and* there is enough support to trust it."""
    warnings: List[str] = []
    if not stats:
        return warnings
    look_back = min(3, len(stats))
    tail = stats[-look_back:]
    for lab in labels:
        supports = [s["per_class"][lab]["support"] for s in tail]
        recalls = [s["per_class"][lab]["recall"] for s in tail]
        # Only warn if we saw at least a few GT samples of this class in the
        # look-back window (otherwise recall is undefined / uninformative).
        if sum(supports) >= 3 and all((r == 0.0) for r in recalls if r == r):
            warnings.append(
                f"class '{lab}' recall==0% over the last {look_back} buckets "
                f"(supports={supports}) -- likely being ignored by the policy."
            )
    # Global support skew warning
    total_support = collections.Counter()
    for s in stats[-look_back:]:
        for lab in labels:
            total_support[lab] += s["per_class"][lab]["support"]
    tot = sum(total_support.values()) or 1
    for lab in labels:
        frac = total_support[lab] / tot
        if frac < 0.05 and total_support[lab] < 5:
            warnings.append(
                f"class '{lab}' saw only {total_support[lab]} GT samples "
                f"({frac * 100:.1f}%) in last {look_back} buckets -- "
                f"training distribution may still be skewed."
            )
    return warnings


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{i} JSON decode failed: {e}", file=sys.stderr)


def resolve_completions_path(run_dir_or_file: str) -> str:
    if os.path.isfile(run_dir_or_file):
        return run_dir_or_file
    if os.path.isdir(run_dir_or_file):
        p = os.path.join(run_dir_or_file, "completions.jsonl")
        if os.path.isfile(p):
            return p
    raise SystemExit(f"[FATAL] cannot locate completions.jsonl from: {run_dir_or_file}")


def dump_csv(csv_path: str, agg: Aggregator) -> None:
    labels = agg.labels
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["step_lo", "step_hi", "total_completions", "accuracy"]
        for lab in labels:
            header += [f"{lab}_support", f"{lab}_correct", f"{lab}_recall"]
        w.writerow(header)
        for s in agg.bucket_stats():
            row = [s["step_lo"], s["step_hi"], s["total_completions"], s["accuracy"]]
            for lab in labels:
                pc = s["per_class"][lab]
                row += [pc["support"], pc["correct"], pc["recall"]]
            w.writerow(row)


def dump_json(json_path: str, agg: Aggregator) -> None:
    payload = {
        "labels": agg.labels,
        "bucket": agg.bucket,
        "n_rows": agg.n_rows,
        "n_completions": agg.n_completions,
        "n_with_gt": agg.n_with_gt,
        "min_step": agg.min_step,
        "max_step": agg.max_step,
        "buckets": agg.bucket_stats(),
        "overall_confusion": {f"{g}->{p}": v for (g, p), v in agg.overall_confusion().items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Optional plotting (matplotlib is imported lazily so the script also runs on
# hosts without it).
# --------------------------------------------------------------------------- #
def dump_plot(plot_path: str, agg: Aggregator) -> bool:
    """Draw per-class recall + accuracy timeline. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable ({e}); skip --plot.", file=sys.stderr)
        return False

    stats = agg.bucket_stats()
    if not stats:
        print("[WARN] no bucket data; skip --plot.", file=sys.stderr)
        return False

    xs = [(s["step_lo"] + s["step_hi"]) / 2.0 for s in stats]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # top: per-class recall
    for lab in agg.labels:
        ys = []
        for s in stats:
            r = s["per_class"][lab]["recall"]
            ys.append(r if r == r else None)  # keep NaN as None (gap)
        ax1.plot(xs, ys, marker="o", label=lab, linewidth=1.6)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("recall")
    ax1.set_title(
        f"GRPO training-time per-class recall  (bucket={agg.bucket} steps, "
        f"rows={agg.n_rows}, completions={agg.n_completions})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9, ncol=len(agg.labels))

    # bottom: overall accuracy + per-class support share
    accs = [s["accuracy"] if s["accuracy"] == s["accuracy"] else 0.0 for s in stats]
    ax2.plot(xs, accs, color="black", marker="s", linewidth=1.8, label="overall acc")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("training step (bucket midpoint)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Optional comparison against a val-set eval_summary.json produced by run_eval
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Optional post-mortem: audio-length distribution of the training jsonl.
# --------------------------------------------------------------------------- #
# Motivation (v7 post-mortem, run v7-20260702-153719 crashed at step 2 with
# 'CUDA driver error: invalid argument' inside qwen2 eager attention):
#   completions.jsonl does NOT record the audio path per row (only prompt +
#   completions + rewards + advantages), so we cannot directly identify the
#   sample that caused the crash. What we CAN do is scan the training jsonl
#   the run was fed with -- if it contains audio whose estimated sequence
#   length exceeds the eager-attention safe zone, that's the leading suspect.
#   This helper reuses probe_duration / extract_audio_paths from
#   scan_audio_lengths.py so the two tools stay in sync.
def render_audio_length_diagnosis(
    train_jsonl: str,
    tokens_per_sec: float,
    prompt_tokens: int,
    max_length: int,
    soft_limit: int,
    workers: int,
    use_librosa: bool,
    label_key: str = "label",
    top_k: int = 10,
) -> str:
    """Read the training jsonl (same file that fed the crashed run) and
    return a rendered post-mortem section.

    Returns an empty-with-explanation block if the sibling module is missing.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"Post-mortem: audio-length distribution of training jsonl "
        f"(tokens/sec={tokens_per_sec}, prompt_tokens={prompt_tokens})"
    )
    lines.append("=" * 78)

    if not os.path.isfile(train_jsonl):
        lines.append(f"[WARN] training jsonl not found: {train_jsonl} -- skip.")
        return "\n".join(lines)

    # Import the sibling scanner lazily so this file is still runnable when
    # scan_audio_lengths.py has been removed / relocated.
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from scan_audio_lengths import (  # type: ignore
            probe_duration as _probe,
            extract_audio_paths as _extract,
            summarize as _summ,
        )
    except Exception as e:
        lines.append(f"[WARN] cannot import scan_audio_lengths.py ({e!r}); "
                     f"skip audio-length diagnosis.")
        return "\n".join(lines)

    import concurrent.futures as cf

    # Load rows.
    rows: List[dict] = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                lines.append(f"[WARN] {train_jsonl}:{i} JSON decode: {e}")
    if not rows:
        lines.append(f"[WARN] no rows loaded from {train_jsonl}")
        return "\n".join(lines)

    # Unique-path probing.
    unique_paths: Dict[str, None] = {}
    for r in rows:
        for p in _extract(r):
            unique_paths.setdefault(p, None)
    n_unique = len(unique_paths)
    if n_unique == 0:
        lines.append(f"[WARN] no audio paths found in {train_jsonl} -- "
                     f"is this the right jsonl?")
        return "\n".join(lines)

    def _probe_one(p: str) -> Tuple[str, Optional[float], bool]:
        exists = os.path.isfile(p)
        if not exists:
            return (p, None, False)
        d = _probe(p, use_librosa=use_librosa)
        return (p, d, True)

    durations: Dict[str, Optional[float]] = {}
    n_missing = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for p, d, exists in ex.map(_probe_one, list(unique_paths.keys())):
            if not exists:
                n_missing += 1
            durations[p] = d if exists else None

    # Build per-row records.
    records = []
    for i, r in enumerate(rows):
        paths = _extract(r)
        if not paths:
            continue
        secs_list = [durations.get(p) for p in paths]
        if any(s is None for s in secs_list):
            continue
        secs = float(sum(secs_list))  # type: ignore[arg-type]
        est_L = int(round(secs * tokens_per_sec + prompt_tokens))
        records.append({
            "row_idx": i,
            "label": str(r.get(label_key, "?")),
            "seconds": secs,
            "est_L": est_L,
            "path0": paths[0],
        })
    if not records:
        lines.append("[WARN] could not measure any row -- probe failed for "
                     "all files (missing on disk or bad headers).")
        return "\n".join(lines)

    all_secs = [r["seconds"] for r in records]
    all_L = [float(r["est_L"]) for r in records]
    ss = _summ(all_secs)
    sl = _summ(all_L)
    lines.append(
        f"  training rows measured : {len(records)}/{len(rows)}  "
        f"(missing on disk: {n_missing})"
    )
    lines.append(
        f"  duration_sec           : mean={ss['mean']:.2f} p50={ss['p50']:.2f} "
        f"p95={ss['p95']:.2f} p99={ss['p99']:.2f} max={ss['max']:.2f}"
    )
    lines.append(
        f"  est_L (audio+prompt)   : mean={sl['mean']:.0f} p50={sl['p50']:.0f} "
        f"p95={sl['p95']:.0f} p99={sl['p99']:.0f} max={sl['max']:.0f}"
    )

    over_hard = [r for r in records if r["est_L"] > max_length]
    over_soft = [r for r in records if r["est_L"] > soft_limit]
    lines.append(
        f"  est_L > MAX_LENGTH ({max_length}) : {len(over_hard)} rows "
        f"({len(over_hard) / len(records) * 100:.2f}%)"
    )
    lines.append(
        f"  est_L > soft_limit  ({soft_limit}) : {len(over_soft)} rows "
        f"({len(over_soft) / len(records) * 100:.2f}%)"
    )

    if over_hard:
        lines.append(
            f"[!] {len(over_hard)} training rows exceed MAX_LENGTH -- prime"
            f" suspect for a CUDA-invalid-argument crash under"
            f" attn_impl='eager'. Consider re-running:"
        )
        lines.append(
            f"    python project/stepaudio/tools/scan_audio_lengths.py "
            f"-i {train_jsonl} "
            f"--emit-safe-output <safe.jsonl> --drop-threshold {soft_limit}"
        )
    elif over_soft:
        lines.append(
            f"[!] {len(over_soft)} rows exceed soft_limit={soft_limit}. "
            f"Not fatal, but if you're chasing 'CUDA invalid argument' errors, "
            f"try lowering MAX_LENGTH to {soft_limit} or emit a safe jsonl "
            f"as above."
        )
    else:
        lines.append(
            "[OK] No rows exceed the safe-length threshold in this jsonl -- "
            "crashes (if any) are unlikely to be caused by long-audio kernel "
            "overflow."
        )

    # Top-K longest.
    if top_k > 0 and records:
        top = sorted(records, key=lambda r: -r["est_L"])[:top_k]
        lines.append("")
        lines.append(f"  Top-{len(top)} longest rows in the training jsonl:")
        for r in top:
            p = r["path0"]
            if len(p) > 62:
                p = "..." + p[-59:]
            lines.append(
                f"    row={r['row_idx']:<7} label={r['label']:<7} "
                f"sec={r['seconds']:7.2f}  est_L={r['est_L']:>6}  {p}"
            )
    return "\n".join(lines)


def load_val_recall(eval_dir_or_json: str, labels: List[str]) -> Dict[str, float]:
    """Load per-class recall from a run_eval.sh output. Accepts either a
    directory containing ``eval_summary.json`` or the JSON file itself.
    Returns {label: recall_float}; missing classes are omitted."""
    p = eval_dir_or_json
    if os.path.isdir(p):
        p = os.path.join(p, "eval_summary.json")
    if not os.path.isfile(p):
        raise SystemExit(f"[FATAL] eval_summary.json not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # run_eval.sh writes ``multiclass_report.per_class[label] = {precision,
    # recall, f1, support, tp, fp, fn}``. Older/other schemas are also
    # tolerated so this stays useful if the eval script evolves.
    per_class = (
        data.get("per_class")
        or data.get("multiclass_report", {}).get("per_class")
        or data.get("multiclass", {}).get("per_class")
        or {}
    )
    out: Dict[str, float] = {}
    for lab in labels:
        if lab in per_class and isinstance(per_class[lab], dict):
            r = per_class[lab].get("recall")
            if isinstance(r, (int, float)):
                out[lab] = float(r)
    return out


def render_compare(agg: Aggregator, val_recall: Dict[str, float]) -> str:
    """Side-by-side compare: last-bucket training recall vs val recall."""
    stats = agg.bucket_stats()
    if not stats:
        return "(no bucket data to compare)"
    last = stats[-1]
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"Train-vs-Val recall comparison  "
        f"(train bucket = steps {last['step_lo']}..{last['step_hi']})"
    )
    lines.append("=" * 78)
    lines.append(f"  {'class':<10} {'train_recall':>14} {'train_support':>14} {'val_recall':>12} {'delta':>10}")
    lines.append("-" * 78)
    for lab in agg.labels:
        pc = last["per_class"][lab]
        tr = pc["recall"]
        sup = pc["support"]
        vr = val_recall.get(lab, float("nan"))
        delta = (tr - vr) if (tr == tr and vr == vr) else float("nan")
        lines.append(
            f"  {lab:<10} "
            f"{_fmt_pct(tr):>14} {sup:>14} {_fmt_pct(vr):>12} "
            f"{('%+5.1f%%' % (delta * 100)) if delta == delta else '   n/a':>10}"
        )
    lines.append("hint: large positive train-val delta -> possible over-fit / train-set leakage.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_aggregator(
    completions_path: str,
    labels: List[str],
    bucket: int,
    tail_steps: Optional[int] = None,
) -> Aggregator:
    agg = Aggregator(labels=labels, bucket=bucket)
    all_rows = list(iter_jsonl(completions_path))
    # If tail_steps is given, only look at rows whose step >= max_step - tail_steps
    if tail_steps is not None and tail_steps > 0 and all_rows:
        max_step = max((s for s in (_row_step(r) for r in all_rows) if s is not None), default=0)
        cutoff = max_step - tail_steps
        all_rows = [r for r in all_rows if (_row_step(r) or 0) >= cutoff]
    for r in all_rows:
        agg.add_row(r)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir",
        "-r",
        required=True,
        help="GRPO output dir (containing completions.jsonl) OR path to the file itself",
    )
    ap.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help=f"comma-separated class labels (default: {','.join(LABELS_DEFAULT)})",
    )
    ap.add_argument("--bucket", type=int, default=100, help="steps per bucket (default: 100)")
    ap.add_argument(
        "--tail",
        type=int,
        default=0,
        help="only aggregate rows whose step >= max_step - TAIL (0 = whole run)",
    )
    ap.add_argument(
        "--tail-buckets",
        type=int,
        default=0,
        help="only display the last N buckets in the terminal table (0 = show all)",
    )
    ap.add_argument("--csv", default="", help="optional CSV output path (per-bucket timeline)")
    ap.add_argument("--json", dest="json_out", default="", help="optional JSON output path")
    ap.add_argument(
        "--plot",
        default="",
        help="optional PNG output path; per-class recall + accuracy timeline "
             "(requires matplotlib; skipped with a warning if not installed)",
    )
    ap.add_argument(
        "--compare-eval-dir",
        default="",
        help="optional path to a run_eval.sh output directory (or its "
             "eval_summary.json) to print a train-vs-val recall side-by-side",
    )
    # ------------------ Post-mortem: audio-length diagnosis ------------------ #
    # Provide the training jsonl the crashed run was fed with; the analyzer
    # will scan its audio durations and flag over-long samples that are the
    # prime suspect for CUDA 'invalid argument' errors under eager attention.
    ap.add_argument(
        "--train-jsonl",
        default="",
        help="optional path to the training jsonl the run consumed. When set, "
             "the analyzer scans audio durations and flags rows whose "
             "estimated sequence length exceeds --audio-max-length or "
             "--audio-soft-limit; useful as a post-mortem when a GRPO run "
             "crashed early with a CUDA 'invalid argument' error inside "
             "eager attention. Requires scan_audio_lengths.py in the same "
             "directory (already shipped).",
    )
    ap.add_argument(
        "--audio-tokens-per-sec",
        type=float,
        default=25.0,
        help="tokens/sec used to estimate audio-token count (default 25 Hz)",
    )
    ap.add_argument(
        "--audio-prompt-tokens",
        type=int,
        default=96,
        help="fixed prompt-token overhead added to audio-token estimate "
             "(default 96)",
    )
    ap.add_argument(
        "--audio-max-length",
        type=int,
        default=3072,
        help="hard MAX_LENGTH used at training time (default 3072)",
    )
    ap.add_argument(
        "--audio-soft-limit",
        type=int,
        default=2048,
        help="soft warning threshold recommended for eager-attention safety "
             "(default 2048)",
    )
    ap.add_argument(
        "--audio-workers",
        type=int,
        default=8,
        help="parallel probe workers for --train-jsonl scanning (default 8)",
    )
    ap.add_argument(
        "--audio-use-librosa",
        action="store_true",
        help="allow librosa fallback for non-wav formats (mp3/ogg)",
    )
    ap.add_argument(
        "--audio-top",
        type=int,
        default=10,
        help="print top-K longest rows in the audio-length diagnosis "
             "(default 10; 0 = suppress)",
    )
    ap.add_argument("--follow", action="store_true", help="loop, re-reading the file every --interval seconds")
    ap.add_argument("--interval", type=int, default=60, help="follow-mode refresh interval in seconds")
    args = ap.parse_args()

    completions_path = resolve_completions_path(args.run_dir)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    def _one_pass() -> None:
        agg = build_aggregator(
            completions_path=completions_path,
            labels=labels,
            bucket=args.bucket,
            tail_steps=(args.tail if args.tail > 0 else None),
        )
        if args.follow:
            os.system("clear" if os.name != "nt" else "cls")
            print(f"[live @ {time.strftime('%Y-%m-%d %H:%M:%S')}]  path: {completions_path}")
        else:
            print(f"[INFO] path: {completions_path}")
        print(render_terminal(agg, tail_bucket=(args.tail_buckets or None)))
        if args.csv:
            dump_csv(args.csv, agg)
            print(f"[OK  ] wrote CSV: {args.csv}")
        if args.json_out:
            dump_json(args.json_out, agg)
            print(f"[OK  ] wrote JSON: {args.json_out}")
        if args.plot:
            if dump_plot(args.plot, agg):
                print(f"[OK  ] wrote plot: {args.plot}")
        if args.compare_eval_dir:
            try:
                vr = load_val_recall(args.compare_eval_dir, labels)
                print(render_compare(agg, vr))
            except SystemExit as e:
                print(str(e), file=sys.stderr)
        if args.train_jsonl:
            try:
                print(render_audio_length_diagnosis(
                    train_jsonl=args.train_jsonl,
                    tokens_per_sec=args.audio_tokens_per_sec,
                    prompt_tokens=args.audio_prompt_tokens,
                    max_length=args.audio_max_length,
                    soft_limit=args.audio_soft_limit,
                    workers=args.audio_workers,
                    use_librosa=args.audio_use_librosa,
                    top_k=args.audio_top,
                ))
            except Exception as e:
                print(f"[WARN] audio-length diagnosis failed: {e!r}",
                      file=sys.stderr)

    if args.follow:
        try:
            while True:
                _one_pass()
                time.sleep(max(1, args.interval))
        except KeyboardInterrupt:
            print("\n[INFO] follow mode interrupted, exiting.")
    else:
        _one_pass()


if __name__ == "__main__":
    main()
