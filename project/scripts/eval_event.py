#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate event-classification inference output produced by `infer_event_qwen3.sh`.

This evaluator reuses `EventAccuracy` / `EventFormat` ORMs defined in
`examples/train/grpo/plugin/event/event_plugin.py`, so that the
offline evaluation logic stays consistent with the GRPO training reward.

Usage:
    python eval_event.py --infer_file <path>/infer_event.jsonl \
        [--gt_jsonl /path/to/test.jsonl] \
        [--report /path/to/report.json] [--show_errors 5]

Notes:
    - The infer file is expected to be the swift-infer jsonl output
      (records contain `messages` and `response`/`choices`).
    - When swift drops the `label` field during inference, the script
      falls back to looking up ground-truth from `--gt_jsonl` keyed by the
      user-side text (instruction + input).
"""
import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Make event_plugin importable regardless of where this script is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_PLUGIN_DIR = os.path.join(_PROJECT_ROOT, "examples", "train", "grpo", "plugin", "event")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from event_plugin import (  # type: ignore  # noqa: E402
    EventAccuracy,
    EventFormat,
    _extract_categories_from_text,
    _find_category_in_completion,
    _get_prompt_text,
    _normalize_gt_label,
)


# Qwen3 chat template still emits an empty `<think>\n\n</think>\n\n` block at
# the head of every reply even when `enable_thinking=false`. We strip it from
# both the `response` text (so ORMs see a clean answer body) and from the
# logprob token sequence (so the "first answer token" used as a confidence
# proxy is the actual category/`<explanation>` token, not `<think>`).
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
# Tokens that belong to the head of the reply but are NOT meaningful answer
# content. Anything that is exactly one of these (after surrounding whitespace
# is removed) is skipped when locating the first "real" answer token.
_THINK_SKIP_TOKENS = {"<think>", "</think>", "think", "/think", "<", ">", "</"}


def _strip_think_prefix(text: str) -> str:
    """Remove a leading ``<think>...</think>`` block from a model response."""
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text, count=1)


def get_user_text(item: dict) -> str:
    """Concatenate user/system text from a swift-infer record's `messages`."""
    msgs = item.get("messages") or []
    parts = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in ("system", "user"):
            parts.append(str(m.get("content", "")))
    return "\n".join(parts).strip()


def build_gt_lookup(gt_jsonl: str) -> Dict[str, str]:
    """Build {user_text -> label} dict from the original alpaca-style jsonl.

    The key is `instruction + "\\n" + input` (matching how swift would build
    the user-side prompt), so we can look up the GT label after swift drops it.
    """
    if not gt_jsonl:
        return {}
    if not os.path.exists(gt_jsonl):
        print(f"[warn] gt_jsonl not found: {gt_jsonl}", file=sys.stderr)
        return {}
    lookup: Dict[str, str] = {}
    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = obj.get("label") or obj.get("solution") or ""
            if not label:
                continue
            instr = str(obj.get("instruction") or "").strip()
            inp = str(obj.get("input") or "").strip()
            # Index by several plausible keys so lookup is robust.
            for key in {
                (instr + "\n" + inp).strip(),
                inp,
                instr,
            }:
                if key and key not in lookup:
                    lookup[key] = str(label)
    print(f"[gt] loaded {len(lookup)} ground-truth labels from {gt_jsonl}")
    return lookup


def load_response(item: dict, gt_lookup: Dict[str, str]) -> Tuple[str, str]:
    """Return (response_text, gt_label) for a swift-infer record."""
    gt = item.get("label") or item.get("solution") or ""
    if not gt and gt_lookup:
        user_text = get_user_text(item)
        # Try multiple candidate keys.
        for key in (user_text, *user_text.split("\n", 1)):
            if key in gt_lookup:
                gt = gt_lookup[key]
                break
    resp = item.get("response")
    if resp is None:
        choices = item.get("choices") or []
        if choices and isinstance(choices[0], dict):
            resp = choices[0].get("message", {}).get("content", "")
    if resp is None:
        resp = item.get("generated_text", "") or ""
    # Qwen3 always prepends an empty `<think>\n\n</think>\n\n` block; remove
    # it so the downstream ORMs (EventAccuracy/EventFormat) see only the
    # actual answer body ("<category>\n<explanation>...</explanation>").
    resp = _strip_think_prefix(str(resp))
    return resp, str(gt)


def _extract_first_answer_token_logprob(item: dict) -> Optional[float]:
    """Logprob of the first *answer* token in a swift-infer record.

    Qwen3 output starts with an empty think block:
        <think>\\n\\n</think>\\n\\n<actual answer ...>
    The bare "first generated token" is therefore always ``<think>`` and
    carries no information about the model's confidence on the predicted
    label. Instead, walk the per-token logprob sequence and return the
    logprob of the first token that is not part of the empty think block,
    nor pure whitespace -- i.e. the first token that belongs to the actual
    answer (typically ``<`` of ``<explanation>``, or the first character of
    the predicted category name).

    swift writes per-token logprobs in two possible places depending on
    backend / version:
      1. Top-level: ``item['logprobs'] = {'content': [{'token','logprob',...}, ...]}``
         (see swift/pipelines/infer/infer.py: ``'logprobs': resp.choices[0].logprobs``).
      2. Inside choices: ``item['choices'][0]['logprobs'] = {...}`` (OpenAI-style).

    Returns the answer-token logprob (a non-positive float), or None when
    logprobs were not requested / not available.
    """
    candidates = []
    lp = item.get("logprobs")
    if lp:
        candidates.append(lp)
    choices = item.get("choices") or []
    if choices and isinstance(choices[0], dict):
        lp2 = choices[0].get("logprobs")
        if lp2:
            candidates.append(lp2)

    for lp in candidates:
        if not isinstance(lp, dict):
            continue
        content = lp.get("content") or []
        if not content:
            continue

        # Walk the token stream and skip the leading think block.
        # Strategy:
        #   - Skip any token whose stripped text is empty (pure whitespace /
        #     newline) or is a structural piece of the think block (the tags
        #     and their fragments after BPE splitting).
        #   - Once we have seen the closing ``</think>`` tag, every subsequent
        #     non-whitespace token is fair game; before seeing it we still
        #     skip whitespace but allow the loop to fall through to the
        #     fallback-first-non-whitespace branch in case logprobs were
        #     produced without a think block at all.
        seen_close_think = False
        first_non_ws_idx: Optional[int] = None
        for idx, tok in enumerate(content):
            if not isinstance(tok, dict):
                continue
            raw = tok.get("token")
            if raw is None:
                continue
            stripped = str(raw).strip()
            if not stripped:
                continue
            if first_non_ws_idx is None:
                first_non_ws_idx = idx
            if not seen_close_think and stripped in _THINK_SKIP_TOKENS:
                if stripped == "</think>" or stripped == "/think":
                    seen_close_think = True
                continue
            # First "real" answer token.
            try:
                return float(tok["logprob"])
            except (TypeError, ValueError, KeyError):
                continue

        # Fallback: if we never identified a clean answer token (e.g. the
        # output had no think block, or used unusual tokenisation), use the
        # first non-whitespace token's logprob -- still better than the
        # naive content[0].
        if first_non_ws_idx is not None:
            tok = content[first_non_ws_idx]
            try:
                return float(tok["logprob"])
            except (TypeError, ValueError, KeyError):
                pass
    return None


def _logprob_to_prob(lp: Optional[float]) -> Optional[float]:
    """Map a logprob into a probability in [0, 1] via ``exp``. None passthrough."""
    if lp is None:
        return None
    try:
        p = math.exp(lp)
    except OverflowError:
        return 1.0
    # Clamp to guard against tiny numerical overshoots.
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _build_threshold_grid() -> List[float]:
    """Default threshold grid for the P/R/F1 sweep.

    Dense near 0 / 1 (where the curves typically bend), uniform in between.
    """
    grid = set()
    for x in (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
              0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
              0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999):
        grid.add(round(x, 4))
    return sorted(grid)


def compute_threshold_curve(
    correctness: List[int],
    confidences: List[Optional[float]],
    thresholds: List[float],
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    """Sweep ``thresholds`` and compute reject-option P/R/F1 metrics.

    Semantics (single-threshold reject-option for multi-class classification):
      * accept sample iff confidence >= threshold
      * TP = #accepted & correct
      * FP = #accepted & wrong
      * FN = #rejected & correct + #rejected & wrong  (everything not accepted
              cannot recover a TP, so rejected-correct also counts as FN here)
      * precision = TP / (TP + FP)              (accuracy on accepted)
      * recall    = TP / N                       (== coverage * precision)
      * F1        = 2PR / (P + R)
      * coverage  = (TP + FP) / N

    Samples missing a confidence score are treated as confidence=NaN, i.e.
    they are rejected for any threshold > 0 but accepted at threshold 0.

    Returns (curve, best_f1_point). ``best_f1_point`` is None when curve empty.
    """
    n = len(correctness)
    assert n == len(confidences)
    curve: List[Dict[str, float]] = []
    for tau in thresholds:
        tp = fp = 0
        accepted = 0
        for ok, c in zip(correctness, confidences):
            if c is None:
                # No confidence available -> reject for any tau > 0,
                # accept only at tau == 0 (so the tau=0 row reproduces the
                # raw accuracy from the existing report).
                accept = (tau <= 0.0)
            else:
                accept = (c >= tau)
            if accept:
                accepted += 1
                if ok:
                    tp += 1
                else:
                    fp += 1
        precision = (tp / accepted) if accepted > 0 else 0.0
        recall = tp / n if n > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        coverage = accepted / n if n > 0 else 0.0
        curve.append({
            "threshold": float(tau),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "coverage": float(coverage),
            "tp": int(tp),
            "fp": int(fp),
            "accepted": int(accepted),
        })
    if not curve:
        return curve, None
    best = max(curve, key=lambda r: (r["f1"], r["threshold"]))
    return curve, dict(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_file", required=True, help="Path to infer_event.jsonl")
    ap.add_argument("--gt_jsonl", default="",
                    help="Original alpaca jsonl (with 'label' field), used to look "
                         "up ground truth when the infer file lacks 'label'.")
    ap.add_argument("--report", default="", help="Optional path to dump JSON report")
    ap.add_argument("--show_errors", type=int, default=5, help="Print N error samples")
    args = ap.parse_args()

    if not os.path.exists(args.infer_file):
        print(f"[error] file not found: {args.infer_file}", file=sys.stderr)
        sys.exit(1)

    gt_lookup = build_gt_lookup(args.gt_jsonl)

    # Buffers fed into the ORMs in a single batch call.
    completions: List[str] = []
    labels: List[str] = []
    messages: List[list] = []
    raw_records: List[dict] = []
    first_token_logprobs: List[Optional[float]] = []

    n_no_gt = 0
    with open(args.infer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp, gt = load_response(item, gt_lookup)
            completions.append(resp)
            labels.append(_normalize_gt_label(gt))
            messages.append(item.get("messages") or [])
            raw_records.append(item)
            first_token_logprobs.append(_extract_first_answer_token_logprob(item))
            if not gt:
                n_no_gt += 1

    n_total = len(completions)
    if n_total == 0:
        print("[error] no valid samples in file", file=sys.stderr)
        sys.exit(2)

    acc_orm = EventAccuracy()
    fmt_orm = EventFormat()
    acc_rewards = acc_orm(completions=completions, label=labels, messages=messages)
    fmt_rewards = fmt_orm(completions=completions, messages=messages)

    n_correct = sum(1 for r in acc_rewards if r >= 0.5)
    n_format = sum(1 for r in fmt_rewards if r >= 0.5)
    acc = n_correct / n_total
    fmt_rate = n_format / n_total

    # ---- Threshold sweep on the first answer-token confidence --------------
    # confidence = exp(logprob_of_first_answer_token), in [0, 1].
    # "first answer token" = first generated token after the empty
    # `<think></think>` block (Qwen3 always emits one even with
    # enable_thinking=false), see _extract_first_answer_token_logprob.
    confidences: List[Optional[float]] = [_logprob_to_prob(lp) for lp in first_token_logprobs]
    n_with_conf = sum(1 for c in confidences if c is not None)
    correctness = [1 if r >= 0.5 else 0 for r in acc_rewards]
    threshold_curve: List[Dict[str, float]] = []
    best_point: Optional[Dict[str, float]] = None
    if n_with_conf > 0:
        thresholds = _build_threshold_grid()
        threshold_curve, best_point = compute_threshold_curve(
            correctness, confidences, thresholds)

    # Collect error samples for display.
    errors = []
    for i, (r_acc, gt) in enumerate(zip(acc_rewards, labels)):
        if r_acc >= 0.5:
            continue
        if len(errors) >= args.show_errors:
            break
        prompt_text = _get_prompt_text(messages[i], None)
        cats = _extract_categories_from_text(prompt_text)
        if gt and gt not in cats:
            cats.append(gt)
        pred = _find_category_in_completion(completions[i], cats)
        errors.append({
            "gt": gt,
            "pred": pred,
            "resp_tail": completions[i][-200:],
        })

    print("=" * 60)
    print("Event Classification Evaluation Report")
    print(f"  file       : {args.infer_file}")
    print(f"  total      : {n_total}")
    print(f"  correct    : {n_correct}")
    print(f"  accuracy   : {acc:.4f}")
    print(f"  format_rate: {fmt_rate:.4f}")
    if n_no_gt:
        print(f"  WARN missing-gt samples: {n_no_gt} (counted as wrong)")
    print("=" * 60)

    # Print the threshold sweep table.
    if threshold_curve:
        print("\n[Threshold sweep] (confidence = exp(first_answer_token_logprob), think block skipped)")
        print(f"  samples with logprob: {n_with_conf}/{n_total}")
        print(f"  {'threshold':>10s}  {'precision':>9s}  {'recall':>7s}  {'f1':>6s}  "
              f"{'coverage':>8s}  {'accepted':>8s}  {'tp':>5s}  {'fp':>5s}")
        for row in threshold_curve:
            print(f"  {row['threshold']:>10.4f}  {row['precision']:>9.4f}  "
                  f"{row['recall']:>7.4f}  {row['f1']:>6.4f}  "
                  f"{row['coverage']:>8.4f}  {row['accepted']:>8d}  "
                  f"{row['tp']:>5d}  {row['fp']:>5d}")
        if best_point is not None:
            print(f"\n  >> best-F1 @ threshold={best_point['threshold']:.4f}: "
                  f"P={best_point['precision']:.4f}  R={best_point['recall']:.4f}  "
                  f"F1={best_point['f1']:.4f}  coverage={best_point['coverage']:.4f}")
    else:
        print("\n[Threshold sweep] skipped: no first-token logprob found in infer file.")
        print("  -> rerun infer with `--logprobs true --top_logprobs 3` (see infer_event_qwen3.sh)")
    if errors:
        print("\n[Sample errors]")
        for i, e in enumerate(errors, 1):
            print(f"  #{i}  gt={e['gt']!r}  pred={e['pred']!r}")
            print(f"       resp_tail: {e['resp_tail']!r}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "infer_file": args.infer_file,
                    "total": n_total,
                    "correct": n_correct,
                    "accuracy": acc,
                    "format_rate": fmt_rate,
                    "missing_gt": n_no_gt,
                    "samples_with_logprob": n_with_conf,
                    "threshold_curve": threshold_curve,
                    "best_f1": best_point,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[report] written -> {args.report}")


if __name__ == "__main__":
    main()
