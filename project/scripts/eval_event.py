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
import os
import sys
from typing import Dict, List, Tuple

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
    return str(resp), str(gt)


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
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[report] written -> {args.report}")


if __name__ == "__main__":
    main()
