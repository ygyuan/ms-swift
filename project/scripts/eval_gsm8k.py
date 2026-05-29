#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate GSM8K inference output produced by `infer_gsm8k_qwen3.5.sh`.

Usage:
    python eval_gsm8k.py --infer_file <path>/infer_gsm8k.jsonl
    python eval_gsm8k.py --infer_file <path>/infer_gsm8k.jsonl --report <path>/report.json
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple


def extract_answer(text: str) -> str:
    """Extract the last \\boxed{...} or #### number, mirroring gsm8k_plugin."""
    if not text:
        return ""
    tail = text[-500:] if len(text) > 500 else text
    boxed = re.findall(r"\\boxed\{([^}]+)\}", tail)
    if boxed:
        return boxed[-1].replace(",", "").replace(" ", "").strip()
    matches = re.findall(r"####\s*([\-\d,\.\s]+)", tail)
    if matches:
        return matches[-1].replace(",", "").replace(" ", "").strip()
    return ""


def is_correct(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (ValueError, OverflowError):
        return pred == gt


def has_format(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\\boxed\{[^}]+\}", text) or re.search(r"####\s*[\-\d,\.]+", text)
    )


def load_response(item: dict) -> Tuple[str, str]:
    """Return (response_text, gt_solution) from a swift infer jsonl record."""
    # gt
    gt = item.get("solution") or item.get("labels") or ""
    # response: try common fields
    resp = item.get("response")
    if resp is None:
        choices = item.get("choices") or []
        if choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            resp = msg.get("content", "")
    if resp is None:
        resp = item.get("generated_text", "") or ""
    return str(resp), str(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_file", required=True, help="Path to infer_gsm8k.jsonl")
    ap.add_argument("--report", default="", help="Optional path to dump JSON report")
    ap.add_argument("--show_errors", type=int, default=5, help="Print N error samples")
    args = ap.parse_args()

    if not os.path.exists(args.infer_file):
        print(f"[error] file not found: {args.infer_file}", file=sys.stderr)
        sys.exit(1)

    n_total = 0
    n_correct = 0
    n_format = 0
    errors = []

    with open(args.infer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp, gt = load_response(item)
            gt_num = extract_answer(gt)
            pred_num = extract_answer(resp)
            ok = is_correct(pred_num, gt_num)
            fmt = has_format(resp)
            n_total += 1
            n_correct += int(ok)
            n_format += int(fmt)
            if not ok and len(errors) < args.show_errors:
                errors.append({"gt": gt_num, "pred": pred_num, "resp_tail": resp[-200:]})

    if n_total == 0:
        print("[error] no valid samples in file", file=sys.stderr)
        sys.exit(2)

    acc = n_correct / n_total
    fmt_rate = n_format / n_total
    print("=" * 60)
    print(f"GSM8K Evaluation Report")
    print(f"  file       : {args.infer_file}")
    print(f"  total      : {n_total}")
    print(f"  correct    : {n_correct}")
    print(f"  accuracy   : {acc:.4f}")
    print(f"  format_rate: {fmt_rate:.4f}")
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
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[report] written -> {args.report}")


if __name__ == "__main__":
    main()
