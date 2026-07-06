#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze an OPSD (On-Policy Self-Distillation) training run.

Why this exists (and how it differs from analyze_grpo_run.py)
-------------------------------------------------------------
GRPO dumps every rollout to ``completions.jsonl`` (prompt + G completions +
per-completion reward + advantages), so we can back out per-class recall
during training. OPSD does NOT: its optimizer step consumes a teacher soft
distribution + student KL, and the trainer only records aggregate scalars
(``loss``, ``learning_rate``, ``eval_loss``, ...) into ``logging.jsonl`` /
``runs/*/events.out.tfevents.*``. Reading a per-batch confusion matrix off
that stream is impossible.

What we CAN do -- and what this script does -- is spot the two OPSD-specific
failure modes early:

1. **eval_loss decoupling from real accuracy.** v14-20260702-112351 showed
   eval_loss falling monotonically 0.0719 -> 0.0666 while real 5-way
   classification accuracy on val collapsed 97.83% -> 87.55%. The signature
   is: eval_loss slope < 0 but per-class recall on val (from run_eval.sh's
   ``eval_summary.json``) is deteriorating, especially on minority classes.
2. **Train loss regime shift.** OPSD's "safe" precision-tuning regime keeps
   train-loss variance small (mean ~0.02, p95 < 0.15). When JSD starts
   pulling the policy off manifold you see a hard step-up in mean loss and
   long-tail spikes -- the earliest tell in the log.

So this script:

* streams ``logging.jsonl`` (or the newest ``trainer_state.json``);
* separates train-loss rows from eval-loss rows;
* buckets by step and reports mean / p50 / p95 / max loss + eval trend +
  learning-rate curve;
* optionally cross-references a directory of per-checkpoint
  ``eval_result_<runtag>_checkpoint-<N>_.../eval_summary.json`` files so
  you get a "eval_loss vs. real accuracy" divergence table + warning;
* keeps the same CLI ergonomics as analyze_grpo_run.py (``--follow``,
  ``--bucket``, ``--tail``, ``--csv``, ``--json``, ``--plot``,
  ``--compare-eval-dir``).

Data schema (as of ms-swift OPSD trainer)
-----------------------------------------
Each line of ``<output_dir>/logging.jsonl`` is one of these shapes::

    # Training log (every logging_steps optimizer steps):
    {"loss": 0.043, "learning_rate": 7e-8, "epoch": 0.00057,
     "global_step/max_steps": "1/500", "elapsed_time": "6s",
     "remaining_time": "47m 29s", "memory(GiB)": 22.46,
     "train_speed(s/it)": 5.71}

    # Eval log (every eval_steps optimizer steps):
    {"eval_loss": 0.0719, "eval_runtime": 49.9, "eval_samples_per_second":
     11.28, "eval_steps_per_second": 2.82, "epoch": 0.028,
     "global_step/max_steps": "50/500", ...}

    # Terminal summary (last line, once):
    {"model_parameter_info": "...", "last_model_checkpoint": "...",
     "best_model_checkpoint": "...", "best_metric": 0.0666,
     "global_step": 246, "log_history": [ ... full replay ... ],
     "memory": 76.85}

Usage
-----
    # One-shot summary of a run
    python analyze_opsd_run.py --run-dir output/stepaudio/opsd/v14-20260702-112351

    # Cross-reference the run's per-checkpoint eval_summary.json outputs
    python analyze_opsd_run.py --run-dir <run> \
        --compare-eval-dir project/stepaudio/infer_results

    # Live mode
    python analyze_opsd_run.py --run-dir <run> --follow --interval 60

    # Emit CSV/JSON/PNG for downstream plotting
    python analyze_opsd_run.py --run-dir <run> \
        --csv opsd_timeline.csv --json opsd_summary.json --plot opsd_curves.png
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
LABELS_DEFAULT = ["speech", "music", "noise", "porn", "song"]

# eval_loss slope thresholds used by the divergence detector below. These are
# tuned against the v14-20260702-112351 post-mortem (eval_loss dropped ~7%
# while real recall fell ~10 pts) so that a single "loss went down, recall
# went down" bucket is reported as a warning rather than dismissed as noise.
EVAL_LOSS_IMPROVE_EPS = 1e-4       # "improved" if new_loss < prev_loss - eps
ACC_DEGRADE_EPS = 5e-4             # "degraded" if new_acc < prev_acc - eps
MACRO_RECALL_ALARM = 0.90          # alarm if macro recall drops below this


# --------------------------------------------------------------------------- #
# Row parsing helpers
# --------------------------------------------------------------------------- #
_STEPMAX_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def _parse_step(row: dict) -> Optional[int]:
    """Recover ``global_step`` from either the flat key or the
    ``"global_step/max_steps": "50/500"`` shape ms-swift writes."""
    for k in ("global_step", "step"):
        v = row.get(k)
        if isinstance(v, int):
            return v
        try:
            if v is not None:
                return int(v)
        except (TypeError, ValueError):
            pass
    combo = row.get("global_step/max_steps")
    if isinstance(combo, str):
        m = _STEPMAX_RE.match(combo)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _parse_max_steps(row: dict) -> Optional[int]:
    combo = row.get("global_step/max_steps")
    if isinstance(combo, str):
        m = _STEPMAX_RE.match(combo)
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                return None
    return None


def _classify_row(row: dict) -> str:
    """Return one of {"train", "eval", "final", "meta"} for a logging.jsonl row."""
    if "eval_loss" in row:
        return "eval"
    if "loss" in row and "learning_rate" in row:
        return "train"
    if "log_history" in row or "best_metric" in row or "model_parameter_info" in row:
        return "final"
    return "meta"


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
class OPSDAggregator:
    """Bucket logging.jsonl into (step-window) statistics.

    Everything is derived off two lists:
      * ``self.train_rows``: (step, loss, lr, mem_gib, speed_s_it)
      * ``self.eval_rows``:  (step, eval_loss, samples_per_sec)
    plus optional ``self.final`` (the terminal row with best_metric etc.).
    """

    def __init__(self, bucket: int):
        self.bucket = max(1, int(bucket))
        self.train_rows: List[Tuple[int, float, Optional[float], Optional[float], Optional[float]]] = []
        self.eval_rows: List[Tuple[int, float, Optional[float]]] = []
        self.final: Optional[dict] = None
        self.max_steps: Optional[int] = None
        self.min_step: Optional[int] = None
        self.max_step: Optional[int] = None

    # ----- ingestion --------------------------------------------------- #
    def add_row(self, row: dict) -> None:
        kind = _classify_row(row)
        step = _parse_step(row)
        ms = _parse_max_steps(row)
        if ms is not None and (self.max_steps is None or ms > self.max_steps):
            self.max_steps = ms

        if step is not None:
            if self.min_step is None or step < self.min_step:
                self.min_step = step
            if self.max_step is None or step > self.max_step:
                self.max_step = step

        if kind == "train":
            try:
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                return
            lr = row.get("learning_rate")
            try:
                lr_f = float(lr) if lr is not None else None
            except (TypeError, ValueError):
                lr_f = None
            mem = row.get("memory(GiB)") or row.get("memory")
            try:
                mem_f = float(mem) if mem is not None else None
            except (TypeError, ValueError):
                mem_f = None
            spd = row.get("train_speed(s/it)")
            try:
                spd_f = float(spd) if spd is not None else None
            except (TypeError, ValueError):
                spd_f = None
            self.train_rows.append((step or 0, loss, lr_f, mem_f, spd_f))
        elif kind == "eval":
            try:
                eloss = float(row["eval_loss"])
            except (KeyError, TypeError, ValueError):
                return
            sps = row.get("eval_samples_per_second")
            try:
                sps_f = float(sps) if sps is not None else None
            except (TypeError, ValueError):
                sps_f = None
            self.eval_rows.append((step or 0, eloss, sps_f))
        elif kind == "final":
            self.final = row
            # log_history in the final row is the ground-truth replay -- if
            # the streaming file was truncated we can rebuild from here.
            hist = row.get("log_history")
            if isinstance(hist, list) and (not self.train_rows or len(hist) > len(self.train_rows) + len(self.eval_rows)):
                self._absorb_log_history(hist)

    def _absorb_log_history(self, hist: List[dict]) -> None:
        """Rebuild train/eval rows off ``trainer_state.log_history`` if the
        streaming logging.jsonl truncated (ms-swift writes the full replay
        into the terminal row)."""
        rebuilt_train, rebuilt_eval = [], []
        for h in hist:
            step = h.get("step")
            try:
                step = int(step) if step is not None else None
            except (TypeError, ValueError):
                step = None
            if step is None:
                continue
            if "eval_loss" in h:
                try:
                    rebuilt_eval.append((step, float(h["eval_loss"]),
                                         float(h["eval_samples_per_second"])
                                         if h.get("eval_samples_per_second") is not None else None))
                except (TypeError, ValueError):
                    pass
            elif "loss" in h:
                try:
                    lr = h.get("learning_rate")
                    rebuilt_train.append((
                        step,
                        float(h["loss"]),
                        float(lr) if lr is not None else None,
                        None,
                        None,
                    ))
                except (TypeError, ValueError):
                    pass
        # Only replace if the replay is strictly larger; otherwise trust the
        # streamed rows (they carry mem/speed the replay drops).
        if len(rebuilt_train) > len(self.train_rows):
            self.train_rows = rebuilt_train
            if rebuilt_train:
                self.min_step = min(self.min_step or rebuilt_train[0][0], rebuilt_train[0][0])
                self.max_step = max(self.max_step or rebuilt_train[-1][0], rebuilt_train[-1][0])
        if len(rebuilt_eval) > len(self.eval_rows):
            self.eval_rows = rebuilt_eval

    # ----- summary primitives ----------------------------------------- #
    @staticmethod
    def _quantiles(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                    "p95": float("nan"), "max": float("nan"), "min": float("nan")}
        s = sorted(xs)
        n = len(s)

        def q(p: float) -> float:
            if n == 1:
                return s[0]
            i = p * (n - 1)
            lo = int(math.floor(i))
            hi = int(math.ceil(i))
            if lo == hi:
                return s[lo]
            return s[lo] + (s[hi] - s[lo]) * (i - lo)

        return {
            "n": n,
            "mean": sum(s) / n,
            "p50": q(0.5),
            "p95": q(0.95),
            "max": s[-1],
            "min": s[0],
        }

    def bucket_stats(self) -> List[dict]:
        """Return per-bucket stats over ``train_rows``. Eval rows are
        attached to the bucket that contains their step."""
        by_bucket: Dict[int, List[Tuple[int, float, Optional[float]]]] = collections.defaultdict(list)
        lr_by_bucket: Dict[int, List[float]] = collections.defaultdict(list)
        for step, loss, lr, _mem, _spd in self.train_rows:
            b = step // self.bucket
            by_bucket[b].append((step, loss, lr))
            if lr is not None:
                lr_by_bucket[b].append(lr)
        eval_by_bucket: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
        for step, eloss, _sps in self.eval_rows:
            eval_by_bucket[step // self.bucket].append((step, eloss))

        out: List[dict] = []
        for b in sorted(by_bucket):
            step_lo = b * self.bucket
            step_hi = step_lo + self.bucket - 1
            rows = by_bucket[b]
            losses = [r[1] for r in rows]
            q = self._quantiles(losses)
            lrs = lr_by_bucket.get(b, [])
            evs = eval_by_bucket.get(b, [])
            out.append({
                "bucket_id": b,
                "step_lo": step_lo,
                "step_hi": step_hi,
                "n_train": q["n"],
                "loss_mean": q["mean"],
                "loss_p50": q["p50"],
                "loss_p95": q["p95"],
                "loss_max": q["max"],
                "loss_min": q["min"],
                "lr_first": lrs[0] if lrs else float("nan"),
                "lr_last": lrs[-1] if lrs else float("nan"),
                "n_eval": len(evs),
                "eval_loss_last": evs[-1][1] if evs else float("nan"),
                "eval_loss_step": evs[-1][0] if evs else None,
            })
        return out


# --------------------------------------------------------------------------- #
# eval_summary.json (per-checkpoint) matching + comparison
# --------------------------------------------------------------------------- #
_EVAL_DIR_RE = re.compile(
    r"^eval_result_(?P<runtag>.+?)_checkpoint-(?P<step>\d+)_(?P<ts>\d+_\d+)$"
)


def find_eval_dirs(root: str, run_tag: Optional[str]) -> List[Tuple[int, str]]:
    """Return ``[(step, dir_path)]`` for every ``eval_result_<runtag>_checkpoint-N_*``
    directory under ``root``. If ``run_tag`` is given, filter to only that
    run's evaluations. Sorted by step ascending; if the same checkpoint has
    multiple eval directories, keep the newest by timestamp suffix."""
    if not os.path.isdir(root):
        return []
    picked: Dict[int, Tuple[str, str]] = {}
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        m = _EVAL_DIR_RE.match(name)
        if not m:
            continue
        if run_tag and m.group("runtag") != run_tag:
            continue
        try:
            step = int(m.group("step"))
        except ValueError:
            continue
        ts = m.group("ts")
        prev = picked.get(step)
        if prev is None or ts > prev[0]:
            picked[step] = (ts, full)
    return sorted([(step, dp) for step, (_ts, dp) in picked.items()], key=lambda x: x[0])


def load_eval_summary(eval_dir_or_json: str, labels: List[str]) -> Optional[Dict[str, object]]:
    """Load a run_eval.sh output. Accepts either a directory containing
    ``eval_summary.json`` or the JSON file itself. Returns a normalized dict
    ``{overall_acc, macro_recall, per_class: {label: {recall, precision, f1, support}}}``
    or None if the file is missing / malformed."""
    p = eval_dir_or_json
    if os.path.isdir(p):
        p = os.path.join(p, "eval_summary.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # run_eval.sh writes the 5-way (multiclass) report under
    # ``multiclass_report`` -- crucially:
    #   multiclass_report.accuracy                     : real 5-way accuracy
    #   multiclass_report.macro.recall / .precision    : macro averages
    #   multiclass_report.per_class[label].{recall,precision,f1,support,tp,fp,fn}
    # The top-level ``per_class`` (if any) is the one-vs-rest binary report
    # driven by ``args.target_class``; tp/support there are NOT a valid basis
    # for computing 5-way accuracy (would double-count the target class).
    # Priority: multiclass_report -> multiclass -> top-level per_class.
    mc_report = data.get("multiclass_report") or data.get("multiclass") or {}
    per_class_raw = mc_report.get("per_class") or data.get("per_class") or {}
    per_class: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        d = per_class_raw.get(lab)
        if not isinstance(d, dict):
            continue
        row: Dict[str, float] = {}
        for k in ("recall", "precision", "f1"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                row[k] = float(v)
        for k in ("support", "tp", "fp", "fn"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                row[k] = float(v)
        per_class[lab] = row

    # macro recall: prefer the one written by run_eval.sh; else average our per_class.
    macro_recall: float = float("nan")
    macro_block = mc_report.get("macro") if isinstance(mc_report, dict) else None
    if isinstance(macro_block, dict) and isinstance(macro_block.get("recall"), (int, float)):
        macro_recall = float(macro_block["recall"])
    elif per_class:
        rs = [v["recall"] for v in per_class.values() if "recall" in v]
        macro_recall = (sum(rs) / len(rs)) if rs else float("nan")

    # overall 5-way accuracy: prefer explicit fields.
    overall_acc: float = float("nan")
    for src in (mc_report, data):
        if not isinstance(src, dict):
            continue
        for k in ("accuracy", "overall_accuracy", "acc"):
            v = src.get(k)
            if isinstance(v, (int, float)):
                overall_acc = float(v)
                break
        if overall_acc == overall_acc:
            break
    # Last-resort recompute: sum of tp / n_total across per_class.
    if overall_acc != overall_acc and per_class:
        n_total = mc_report.get("n_total") if isinstance(mc_report, dict) else None
        if not isinstance(n_total, (int, float)):
            n_total = data.get("n_used") or sum(int(v.get("support", 0)) for v in per_class.values())
        n_correct = mc_report.get("n_correct") if isinstance(mc_report, dict) else None
        if not isinstance(n_correct, (int, float)):
            n_correct = sum(int(v.get("tp", 0)) for v in per_class.values())
        if n_total:
            overall_acc = float(n_correct) / float(n_total)
    return {
        "path": p,
        "per_class": per_class,
        "macro_recall": macro_recall,
        "overall_acc": overall_acc,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{x * 100:5.1f}%"


def _fmt_f(x: float, w: int = 8, prec: int = 4) -> str:
    if x != x:
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.{prec}f}"


def render_terminal(agg: OPSDAggregator, tail_bucket: Optional[int] = None) -> str:
    lines: List[str] = []
    stats = agg.bucket_stats()
    if not stats:
        return "(no bucket data -- logging.jsonl empty or unrecognized schema)"
    if tail_bucket is not None and tail_bucket > 0:
        stats = stats[-tail_bucket:]

    header_top = (
        f"OPSD training timeline  (bucket={agg.bucket} steps, "
        f"steps={agg.min_step}..{agg.max_step}"
        + (f"/{agg.max_steps}" if agg.max_steps else "")
        + f", train_rows={len(agg.train_rows)}, eval_rows={len(agg.eval_rows)})"
    )
    lines.append("=" * 90)
    lines.append(header_top)
    lines.append("=" * 90)

    hdr = (f"{'steps':>13} {'n':>5} "
           f"{'loss_mean':>10} {'loss_p50':>10} {'loss_p95':>10} {'loss_max':>10} "
           f"{'lr_last':>10} {'eval_loss':>10}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in stats:
        eval_cell = _fmt_f(s["eval_loss_last"], w=10, prec=5) if s["n_eval"] else f"{'-':>10}"
        lines.append(
            f"{s['step_lo']:>5}-{s['step_hi']:<5} "
            f"{s['n_train']:>5} "
            f"{_fmt_f(s['loss_mean'], w=10, prec=5)} "
            f"{_fmt_f(s['loss_p50'], w=10, prec=5)} "
            f"{_fmt_f(s['loss_p95'], w=10, prec=5)} "
            f"{_fmt_f(s['loss_max'], w=10, prec=5)} "
            f"{_fmt_f(s['lr_last'], w=10, prec=2)} "
            f"{eval_cell}"
        )
    lines.append("legend: loss_mean/p50/p95/max are computed over train-log rows in the bucket;")
    lines.append("        eval_loss is the last eval-log recorded in the bucket ('-' = no eval this bucket).")

    # ---- Eval-loss trend ---------------------------------------------- #
    if agg.eval_rows:
        lines.append("")
        lines.append("Eval-loss trajectory:")
        lines.append(f"  {'step':>7}  {'eval_loss':>10}  {'delta':>10}  {'samples/s':>10}")
        prev = None
        best_step, best_val = None, float("inf")
        for step, eloss, sps in agg.eval_rows:
            delta = "  n/a" if prev is None else f"{eloss - prev:+.5f}"
            sps_cell = _fmt_f(sps, w=10, prec=2) if sps is not None else f"{'-':>10}"
            marker = ""
            if eloss < best_val:
                best_val, best_step = eloss, step
                marker = "  <- best"
            lines.append(f"  {step:>7}  {eloss:>10.5f}  {delta:>10}  {sps_cell}{marker}")
            prev = eloss
        lines.append(f"[INFO] best eval_loss so far: {best_val:.5f} @ step {best_step}")

    # ---- Warnings ----------------------------------------------------- #
    warnings = _train_regime_warnings(agg, stats)
    lines.append("")
    if warnings:
        lines.append("[!] OPSD training-regime warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    else:
        lines.append("[OK] Train loss and eval_loss look well-behaved (no regime shift detected).")

    if agg.final:
        best_metric = agg.final.get("best_metric")
        best_ckpt = agg.final.get("best_model_checkpoint")
        last_ckpt = agg.final.get("last_model_checkpoint")
        gs = agg.final.get("global_step")
        lines.append("")
        lines.append(
            f"[INFO] final: global_step={gs}, best_metric={best_metric}, "
            f"best={best_ckpt}, last={last_ckpt}"
        )
    return "\n".join(lines)


def _train_regime_warnings(agg: OPSDAggregator, stats: List[dict]) -> List[str]:
    """Emit warnings on the two OPSD-specific failure modes."""
    warnings: List[str] = []
    if not stats:
        return warnings

    # (a) train-loss regime shift: mean loss jumps > 3x between two adjacent
    # buckets -- earliest sign of JSD pulling the policy off-manifold.
    prev = None
    for s in stats:
        if prev is not None and prev["loss_mean"] > 0 and s["loss_mean"] > 0:
            ratio = s["loss_mean"] / prev["loss_mean"]
            if ratio >= 3.0:
                warnings.append(
                    f"train-loss regime shift @ steps {s['step_lo']}-{s['step_hi']}: "
                    f"loss_mean {prev['loss_mean']:.4f} -> {s['loss_mean']:.4f} "
                    f"({ratio:.1f}x) -- possible JSD off-manifold pull."
                )
        prev = s

    # (b) train-loss long-tail spike: p95/mean > 8 in the last bucket
    tail = stats[-1]
    if tail["loss_mean"] > 0 and tail["loss_p95"] / tail["loss_mean"] > 8.0 and tail["n_train"] >= 20:
        warnings.append(
            f"loss long-tail spike @ steps {tail['step_lo']}-{tail['step_hi']}: "
            f"p95/mean = {tail['loss_p95'] / tail['loss_mean']:.1f} "
            f"(p95={tail['loss_p95']:.4f}, mean={tail['loss_mean']:.4f}) -- "
            "check whether outlier batches are a specific class / audio-length bucket."
        )

    # (c) eval_loss stalled while train_loss falls
    if len(agg.eval_rows) >= 3 and len(stats) >= 2:
        first_eval = agg.eval_rows[0][1]
        last_eval = agg.eval_rows[-1][1]
        first_bucket_mean = stats[0]["loss_mean"]
        last_bucket_mean = stats[-1]["loss_mean"]
        train_dropped = (first_bucket_mean - last_bucket_mean) > 0.005
        eval_stalled = abs(last_eval - first_eval) < EVAL_LOSS_IMPROVE_EPS
        if train_dropped and eval_stalled:
            warnings.append(
                f"train_loss decreasing but eval_loss flat "
                f"(first_eval={first_eval:.5f}, last_eval={last_eval:.5f}) "
                "-- possible over-fit to train-set audio prints."
            )
    return warnings


def render_ckpt_compare(
    agg: OPSDAggregator,
    ckpt_evals: List[Tuple[int, dict]],
    labels: List[str],
) -> str:
    """Cross-reference each checkpoint's eval_summary.json with the
    corresponding eval_loss recorded during training. This is the killer
    view for OPSD: it exposes 'eval_loss falling while real recall is
    falling' -- the exact v14 failure signature."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("Checkpoint eval cross-reference (train-time eval_loss vs. run_eval.sh recall)")
    lines.append("=" * 90)
    if not ckpt_evals:
        lines.append("(no eval_summary.json found for this run tag; nothing to compare)")
        return "\n".join(lines)

    step_to_eval_loss: Dict[int, float] = {step: el for step, el, _ in agg.eval_rows}

    hdr = (f"  {'step':>6}  {'eval_loss':>10}  {'overall_acc':>11}  "
           f"{'macro_R':>8}  " + " ".join(f"{lab[:6]:>7}" for lab in labels))
    lines.append(hdr)
    lines.append("-" * len(hdr))

    prev_acc: Optional[float] = None
    prev_eval_loss: Optional[float] = None
    divergence_hits: List[Tuple[int, float, float]] = []
    macro_alarm_steps: List[Tuple[int, float]] = []
    for step, summary in ckpt_evals:
        eloss = step_to_eval_loss.get(step, float("nan"))
        acc = summary.get("overall_acc", float("nan"))
        macro = summary.get("macro_recall", float("nan"))
        pcs = []
        for lab in labels:
            r = summary.get("per_class", {}).get(lab, {}).get("recall", float("nan"))
            pcs.append(_fmt_pct(r))
        eloss_cell = _fmt_f(eloss, w=10, prec=5) if eloss == eloss else f"{'  n/a':>10}"
        lines.append(
            f"  {step:>6}  {eloss_cell}  {_fmt_pct(acc):>11}  "
            f"{_fmt_pct(macro):>8}  " + " ".join(f"{c:>7}" for c in pcs)
        )
        # Divergence detector: eval_loss improved but overall_acc dropped.
        if (prev_eval_loss is not None and prev_acc is not None
                and eloss == eloss and acc == acc
                and prev_eval_loss - eloss > EVAL_LOSS_IMPROVE_EPS
                and prev_acc - acc > ACC_DEGRADE_EPS):
            divergence_hits.append((step, eloss - prev_eval_loss, acc - prev_acc))
        if macro == macro and macro < MACRO_RECALL_ALARM:
            macro_alarm_steps.append((step, macro))
        prev_acc = acc
        prev_eval_loss = eloss

    # ---- Divergence warnings ---------------------------------------- #
    lines.append("")
    if divergence_hits:
        lines.append("[!] eval_loss / real-accuracy DIVERGENCE detected:")
        for step, dloss, dacc in divergence_hits:
            lines.append(
                f"    - step {step}: eval_loss {dloss:+.5f} but overall_acc "
                f"{dacc * 100:+.2f}%  <-- OPSD is optimizing the wrong signal."
            )
        lines.append("    hint: this is v14's failure mode (loss down / recall down). "
                     "Consider softer teacher hint, smaller LR, or larger sft_alpha.")
    else:
        lines.append("[OK] No eval_loss / real-accuracy divergence between checkpoints.")

    if macro_alarm_steps:
        lines.append(
            f"[!] macro recall < {MACRO_RECALL_ALARM * 100:.0f}% at "
            + ", ".join(f"step {st} ({m * 100:.1f}%)" for st, m in macro_alarm_steps)
            + " -- check per-class table for zeroed minority classes."
        )
    return "\n".join(lines)


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


def resolve_run_dir(run_dir_or_file: str) -> Tuple[str, str]:
    """Return (run_dir, logging_jsonl_path). Accepts:
      * a run directory containing ``logging.jsonl``;
      * a direct path to a ``logging.jsonl`` file;
      * a run directory whose logging.jsonl was truncated -- we fall back to
        the newest ``checkpoint-*/trainer_state.json``'s log_history."""
    if os.path.isfile(run_dir_or_file):
        return os.path.dirname(os.path.abspath(run_dir_or_file)), run_dir_or_file
    if os.path.isdir(run_dir_or_file):
        p = os.path.join(run_dir_or_file, "logging.jsonl")
        if os.path.isfile(p):
            return run_dir_or_file, p
        # Fallback: latest trainer_state.json is a full replay.
        cks = sorted(
            glob.glob(os.path.join(run_dir_or_file, "checkpoint-*/trainer_state.json")),
            key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
                if re.search(r"checkpoint-(\d+)", x) else 0,
        )
        if cks:
            return run_dir_or_file, cks[-1]
    raise SystemExit(
        f"[FATAL] cannot locate logging.jsonl or trainer_state.json under: {run_dir_or_file}"
    )


def _run_tag_from_dir(run_dir: str) -> Optional[str]:
    """Given ``.../output/stepaudio/opsd/v14-20260702-112351``, return
    ``v14-20260702-112351``."""
    base = os.path.basename(os.path.abspath(run_dir.rstrip("/")))
    return base or None


def dump_csv(csv_path: str, agg: OPSDAggregator) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_lo", "step_hi", "n_train",
                    "loss_mean", "loss_p50", "loss_p95", "loss_max", "loss_min",
                    "lr_first", "lr_last",
                    "n_eval", "eval_loss_last", "eval_loss_step"])
        for s in agg.bucket_stats():
            w.writerow([s["step_lo"], s["step_hi"], s["n_train"],
                        s["loss_mean"], s["loss_p50"], s["loss_p95"], s["loss_max"], s["loss_min"],
                        s["lr_first"], s["lr_last"],
                        s["n_eval"], s["eval_loss_last"], s["eval_loss_step"]])


def dump_json(json_path: str, agg: OPSDAggregator, ckpt_evals: List[Tuple[int, dict]]) -> None:
    payload = {
        "bucket": agg.bucket,
        "min_step": agg.min_step,
        "max_step": agg.max_step,
        "max_steps": agg.max_steps,
        "n_train_rows": len(agg.train_rows),
        "n_eval_rows": len(agg.eval_rows),
        "buckets": agg.bucket_stats(),
        "eval_trajectory": [
            {"step": s, "eval_loss": el, "samples_per_second": sps}
            for s, el, sps in agg.eval_rows
        ],
        "final": agg.final,
        "checkpoint_evals": [
            {
                "step": step,
                "path": summary.get("path"),
                "overall_acc": summary.get("overall_acc"),
                "macro_recall": summary.get("macro_recall"),
                "per_class": summary.get("per_class"),
            }
            for step, summary in ckpt_evals
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_plot(plot_path: str, agg: OPSDAggregator, ckpt_evals: List[Tuple[int, dict]]) -> bool:
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
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax_loss, ax_eval, ax_acc = axes

    # Top: train loss mean + p95 band
    means = [s["loss_mean"] for s in stats]
    p95s = [s["loss_p95"] for s in stats]
    ax_loss.plot(xs, means, marker="o", linewidth=1.6, label="train loss (mean)", color="C0")
    ax_loss.plot(xs, p95s, marker=".", linewidth=1.0, label="train loss (p95)", color="C0", alpha=0.5, linestyle="--")
    ax_loss.set_ylabel("train loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best", fontsize=9)
    ax_loss.set_title(
        f"OPSD run  (bucket={agg.bucket}, train_rows={len(agg.train_rows)}, "
        f"eval_rows={len(agg.eval_rows)})"
    )

    # Middle: eval_loss trajectory
    if agg.eval_rows:
        ex = [s for s, _, _ in agg.eval_rows]
        ey = [el for _, el, _ in agg.eval_rows]
        ax_eval.plot(ex, ey, marker="s", linewidth=1.6, color="C1", label="eval_loss")
        ax_eval.set_ylabel("eval_loss")
        ax_eval.grid(True, alpha=0.3)
        ax_eval.legend(loc="best", fontsize=9)
    else:
        ax_eval.text(0.5, 0.5, "(no eval rows)", ha="center", va="center",
                     transform=ax_eval.transAxes)
        ax_eval.set_ylabel("eval_loss")

    # Bottom: real per-class recall + overall acc from ckpt eval summaries
    if ckpt_evals:
        steps = [s for s, _ in ckpt_evals]
        accs = [summ.get("overall_acc", float("nan")) for _, summ in ckpt_evals]
        ax_acc.plot(steps, accs, marker="D", linewidth=1.8, color="black",
                    label="overall accuracy")
        # per-class recall
        labels_seen = set()
        for _, summ in ckpt_evals:
            labels_seen.update((summ.get("per_class") or {}).keys())
        for lab in sorted(labels_seen):
            ys = [(summ.get("per_class", {}).get(lab, {}).get("recall", float("nan")))
                  for _, summ in ckpt_evals]
            ax_acc.plot(steps, ys, marker="o", linewidth=1.2, alpha=0.8, label=lab)
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.set_ylabel("val recall")
        ax_acc.set_xlabel("training step")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="lower left", fontsize=8, ncol=3)
    else:
        ax_acc.text(0.5, 0.5, "(no per-checkpoint eval_summary.json found;\n"
                              "pass --compare-eval-dir <infer_results dir>)",
                    ha="center", va="center", transform=ax_acc.transAxes)
        ax_acc.set_xlabel("training step")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_aggregator(logging_path: str, bucket: int, tail_steps: Optional[int] = None) -> OPSDAggregator:
    agg = OPSDAggregator(bucket=bucket)
    # Two supported inputs:
    #   * jsonl streaming file (logging.jsonl)
    #   * trainer_state.json (a single big JSON with log_history)
    if logging_path.endswith(".jsonl"):
        for row in iter_jsonl(logging_path):
            agg.add_row(row)
    else:
        try:
            with open(logging_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"[FATAL] cannot parse {logging_path}: {e}")
        # Wrap it into a single "final"-shaped row so add_row's replay path handles it.
        agg.add_row({"log_history": data.get("log_history") or [],
                     "best_metric": data.get("best_metric"),
                     "best_model_checkpoint": data.get("best_model_checkpoint"),
                     "last_model_checkpoint": data.get("last_model_checkpoint"),
                     "global_step": data.get("global_step")})

    if tail_steps and tail_steps > 0 and agg.max_step is not None:
        cutoff = agg.max_step - tail_steps
        agg.train_rows = [r for r in agg.train_rows if r[0] >= cutoff]
        agg.eval_rows = [r for r in agg.eval_rows if r[0] >= cutoff]
        # recompute step envelope
        all_steps = [r[0] for r in agg.train_rows] + [r[0] for r in agg.eval_rows]
        if all_steps:
            agg.min_step = min(all_steps)
            agg.max_step = max(all_steps)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", "-r", required=True,
                    help="OPSD output dir (containing logging.jsonl) OR path to logging.jsonl / trainer_state.json")
    ap.add_argument("--bucket", type=int, default=50,
                    help="steps per bucket for train-loss aggregation (default: 50)")
    ap.add_argument("--tail", type=int, default=0,
                    help="only aggregate rows whose step >= max_step - TAIL (0 = whole run)")
    ap.add_argument("--tail-buckets", type=int, default=0,
                    help="only display the last N buckets in the terminal table (0 = show all)")
    ap.add_argument("--labels", default=",".join(LABELS_DEFAULT),
                    help=f"comma-separated class labels for the ckpt-eval cross-reference "
                         f"(default: {','.join(LABELS_DEFAULT)})")
    ap.add_argument("--compare-eval-dir", default="",
                    help="root directory holding eval_result_<runtag>_checkpoint-N_* subdirectories "
                         "(typically project/stepaudio/infer_results). If provided, the analyzer will "
                         "auto-match checkpoints by run tag and cross-reference eval_loss vs. real recall.")
    ap.add_argument("--run-tag", default="",
                    help="override the run tag used to filter --compare-eval-dir "
                         "(defaults to the basename of --run-dir).")
    ap.add_argument("--csv", default="", help="optional CSV output path (per-bucket timeline)")
    ap.add_argument("--json", dest="json_out", default="", help="optional JSON output path")
    ap.add_argument("--plot", default="",
                    help="optional PNG output path; train_loss + eval_loss + val recall timeline "
                         "(requires matplotlib)")
    ap.add_argument("--follow", action="store_true",
                    help="loop, re-reading the file every --interval seconds")
    ap.add_argument("--interval", type=int, default=60,
                    help="follow-mode refresh interval in seconds")
    args = ap.parse_args()

    run_dir, logging_path = resolve_run_dir(args.run_dir)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    run_tag = args.run_tag or _run_tag_from_dir(run_dir)

    def _one_pass() -> None:
        agg = build_aggregator(
            logging_path=logging_path,
            bucket=args.bucket,
            tail_steps=(args.tail if args.tail > 0 else None),
        )

        # Match per-checkpoint eval_summary.json.
        ckpt_evals: List[Tuple[int, dict]] = []
        if args.compare_eval_dir:
            eval_dirs = find_eval_dirs(args.compare_eval_dir, run_tag=run_tag)
            for step, dp in eval_dirs:
                s = load_eval_summary(dp, labels)
                if s is not None:
                    ckpt_evals.append((step, s))

        if args.follow:
            os.system("clear" if os.name != "nt" else "cls")
            print(f"[live @ {time.strftime('%Y-%m-%d %H:%M:%S')}]  path: {logging_path}")
        else:
            print(f"[INFO] run_dir : {run_dir}")
            print(f"[INFO] source  : {logging_path}")
            if run_tag:
                print(f"[INFO] run_tag : {run_tag}")

        print(render_terminal(agg, tail_bucket=(args.tail_buckets or None)))

        if args.compare_eval_dir:
            print(render_ckpt_compare(agg, ckpt_evals, labels))
            if not ckpt_evals:
                print(f"[HINT ] no eval dirs matched run_tag='{run_tag}' under {args.compare_eval_dir}. "
                      "Run project/stepaudio/run_eval.sh first, or override with --run-tag.")

        if args.csv:
            dump_csv(args.csv, agg)
            print(f"[OK  ] wrote CSV: {args.csv}")
        if args.json_out:
            dump_json(args.json_out, agg, ckpt_evals)
            print(f"[OK  ] wrote JSON: {args.json_out}")
        if args.plot:
            if dump_plot(args.plot, agg, ckpt_evals):
                print(f"[OK  ] wrote plot: {args.plot}")

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
