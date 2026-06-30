#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""StepAudio2-mini audio scene classification evaluator.

Reads the JSONL produced by run_inference.sh and reports:
  A. Multi-class precision / recall / F1 (per-class + macro + weighted) and a
     confusion matrix, using argmax (model textual response).
  B. Threshold sweep on a target class (default: porn). Score = softmax over
     candidate classes' first-token logprobs (requires --logprobs in inference).

Usage:
    python eval_classification.py \
        --result_path infer_results/result_xxx.jsonl \
        --val_jsonl   data/val.jsonl \
        --target_class porn \
        --output_dir  infer_results/eval_xxx
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


CANDIDATE_CLASSES_DEFAULT = ["speech", "music", "noise", "porn", "song"]

# 识别推理输出中的 "无效模板泄漏" token：TTS / audio token / EOT 等
# 这些出现说明模型根本没按分类指令生成，应该单独统计。
_INVALID_TOKEN_PATTERNS = [
    re.compile(r"<tts_(?:start|end|pad)>", re.IGNORECASE),
    re.compile(r"<audio_\d+>", re.IGNORECASE),
    re.compile(r"<\|EOT\|>", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
]


def is_invalid_response(text: Optional[str]) -> bool:
    """判断 response 是否是“无效输出”：空 / 包含任何模板泄漏 token。"""
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    for p in _INVALID_TOKEN_PATTERNS:
        if p.search(s):
            return True
    return False


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label(text, classes):
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    for c in classes:
        if s == c:
            return c
    best_pos, best_cls = None, None
    for c in classes:
        idx = s.find(c)
        if idx >= 0 and (best_pos is None or idx < best_pos):
            best_pos, best_cls = idx, c
    return best_cls


def extract_gt_label(item, classes):
    if item.get("label"):
        cand = normalize_label(item["label"], classes)
        if cand:
            return cand
    if item.get("labels"):
        cand = normalize_label(item["labels"], classes)
        if cand:
            return cand
    for m in item.get("messages", []) or []:
        if m.get("role") == "assistant":
            cand = normalize_label(m.get("content", ""), classes)
            if cand:
                return cand
    return None


def extract_pred_label(item, classes):
    return normalize_label(item.get("response", ""), classes)


def _safe_lower(x):
    return "" if x is None else str(x).strip().lower()


def extract_first_token_topk(item):
    lp = item.get("logprobs")
    if not lp:
        return None
    content = lp.get("content") if isinstance(lp, dict) else None
    if not content:
        return None
    first = content[0]
    topk = first.get("top_logprobs") or []
    out = []
    for t in topk:
        if not isinstance(t, dict):
            continue
        tok = t.get("token", "")
        lpv = t.get("logprob")
        if lpv is None:
            continue
        try:
            out.append((tok, float(lpv)))
        except (TypeError, ValueError):
            continue
    self_tok = first.get("token", "")
    self_lp = first.get("logprob")
    if self_tok and self_lp is not None:
        try:
            self_lp = float(self_lp)
            if not any(_safe_lower(t) == _safe_lower(self_tok) for t, _ in out):
                out.append((self_tok, self_lp))
        except (TypeError, ValueError):
            pass
    return out or None


def class_score_from_topk(topk, classes):
    raw_logp = {c: -math.inf for c in classes}
    for tok, lp in topk:
        t = _safe_lower(tok)
        if not t:
            continue
        for c in classes:
            cl = c.lower()
            if t == cl or t.lstrip().startswith(cl) or cl.startswith(t.lstrip()):
                if lp > raw_logp[c]:
                    raw_logp[c] = lp
                break
    finite = [v for v in raw_logp.values() if math.isfinite(v)]
    if not finite:
        return {c: 0.0 for c in classes}
    m = max(finite)
    exps = {c: (math.exp(v - m) if math.isfinite(v) else 0.0) for c, v in raw_logp.items()}
    s = sum(exps.values())
    if s <= 0:
        return {c: 0.0 for c in classes}
    return {c: v / s for c, v in exps.items()}


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def per_class_report(y_true, y_pred, classes):
    rows = {}
    n_total = len(y_true)
    n_correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    label_set = list(classes)
    if any(p not in classes and p is not None for p in y_pred):
        label_set = label_set + ["<other>"]
    confusion = {gt: Counter() for gt in label_set}
    for gt, pr in zip(y_true, y_pred):
        if gt is None:
            continue
        pr_eff = pr if pr in classes else ("<none>" if pr is None else "<other>")
        confusion[gt][pr_eff] = confusion[gt].get(pr_eff, 0) + 1

    macro_p = macro_r = macro_f = 0.0
    weighted_p = weighted_r = weighted_f = 0.0
    total_support = 0
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        support = sum(1 for a in y_true if a == c)
        p, r, f = precision_recall_f1(tp, fp, fn)
        rows[c] = {"precision": p, "recall": r, "f1": f, "support": support,
                   "tp": tp, "fp": fp, "fn": fn}
        macro_p += p; macro_r += r; macro_f += f
        weighted_p += p * support; weighted_r += r * support; weighted_f += f * support
        total_support += support
    n_classes = max(len(classes), 1)
    macro_p /= n_classes; macro_r /= n_classes; macro_f /= n_classes
    if total_support > 0:
        weighted_p /= total_support; weighted_r /= total_support; weighted_f /= total_support
    return {
        "accuracy": accuracy, "n_total": n_total, "n_correct": n_correct,
        "per_class": rows,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f},
        "confusion": {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


def threshold_sweep(scores, is_pos, thresholds):
    rows = []
    n_pos_total = sum(is_pos)
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, is_pos):
            pred = 1 if s >= thr else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 1: fn += 1
            else: tn += 1
        p, r, f = precision_recall_f1(tp, fp, fn)
        rows.append({"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": p, "recall": r, "f1": f,
                     "predicted_positive": tp + fp, "actual_positive": n_pos_total})
    return rows


def fmt_pct(x):
    return f"{x * 100:7.3f}%"


def print_class_report(report, classes):
    print("=" * 72)
    print(f"Multi-class report  (N={report['n_total']}, "
          f"accuracy={fmt_pct(report['accuracy'])})")
    print("-" * 72)
    print(f"{'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} "
          f"{'support':>8}  (tp/fp/fn)")
    for c in classes:
        row = report["per_class"][c]
        print(f"{c:<10} {fmt_pct(row['precision'])} {fmt_pct(row['recall'])} "
              f"{fmt_pct(row['f1'])} {row['support']:>8}  "
              f"({row['tp']}/{row['fp']}/{row['fn']})")
    m, w = report["macro"], report["weighted"]
    print("-" * 72)
    print(f"{'macro':<10} {fmt_pct(m['precision'])} {fmt_pct(m['recall'])} {fmt_pct(m['f1'])}")
    print(f"{'weighted':<10} {fmt_pct(w['precision'])} {fmt_pct(w['recall'])} {fmt_pct(w['f1'])}")


def print_confusion(report, classes):
    print("=" * 72)
    print("Confusion matrix (rows=GT, cols=Pred):")
    cols = list(classes)
    extras = sorted({p for row in report["confusion"].values() for p in row.keys() if p not in cols})
    cols = cols + extras
    print(" " * 10 + "".join(f"{c:>10}" for c in cols))
    for gt in classes:
        row = report["confusion"].get(gt, {})
        print(f"{gt:<10}" + "".join(f"{row.get(c, 0):>10}" for c in cols))


def print_threshold_table(rows, target):
    print("=" * 72)
    print(f"Threshold sweep on target = '{target}'")
    print("-" * 72)
    print(f"{'thr':>6} {'precision':>10} {'recall':>10} {'f1':>10} "
          f"{'tp':>6} {'fp':>6} {'fn':>6} {'tn':>6}")
    for r in rows:
        print(f"{r['threshold']:>6.2f} {fmt_pct(r['precision'])} "
              f"{fmt_pct(r['recall'])} {fmt_pct(r['f1'])} "
              f"{r['tp']:>6} {r['fp']:>6} {r['fn']:>6} {r['tn']:>6}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", required=True)
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--classes", default=",".join(CANDIDATE_CLASSES_DEFAULT))
    ap.add_argument("--target_class", default="porn")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_print_rows", type=int, default=21)
    return ap.parse_args()


def build_val_index(val_path):
    idx = {}
    if not val_path or not os.path.isfile(val_path):
        return idx
    for i, item in enumerate(load_jsonl(val_path)):
        key = item.get("key") or f"__row_{i}__"
        idx[str(key)] = item
        idx[f"__row_{i}__"] = item
    return idx


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.target_class not in classes:
        print(f"[WARN] target_class={args.target_class} not in classes={classes}, append.",
              file=sys.stderr)
        classes.append(args.target_class)

    if not args.thresholds:
        thresholds = [round(x * 0.05, 2) for x in range(1, 20)]
    else:
        thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.result_path)),
        "eval_" + os.path.splitext(os.path.basename(args.result_path))[0])
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] result_path = {args.result_path}")
    print(f"[INFO] val_jsonl   = {args.val_jsonl}")
    print(f"[INFO] classes     = {classes}")
    print(f"[INFO] target      = {args.target_class}")
    print(f"[INFO] output_dir  = {out_dir}")

    val_index = build_val_index(args.val_jsonl) if args.val_jsonl else {}
    items = load_jsonl(args.result_path)
    if not items:
        print(f"[ERROR] empty result file: {args.result_path}", file=sys.stderr)
        sys.exit(1)

    y_true, y_pred = [], []
    scores_target, is_pos_target = [], []
    n_unknown_gt = n_unknown_pred = n_with_logprob = 0
    n_invalid_response = 0
    invalid_by_gt = Counter()        # 按真值类别统计无效输出数
    invalid_samples: List[dict] = []  # 采样几个无效输出示例，用于 debug

    for i, it in enumerate(items):
        gt = None
        if val_index:
            ref = None
            key = it.get("key")
            if key and str(key) in val_index:
                ref = val_index[str(key)]
            elif f"__row_{i}__" in val_index:
                ref = val_index[f"__row_{i}__"]
            if ref is not None:
                gt = normalize_label(ref.get("label") or ref.get("labels"), classes)
        if gt is None:
            gt = extract_gt_label(it, classes)
        if gt is None:
            n_unknown_gt += 1
            continue

        raw_resp = it.get("response", "")
        invalid = is_invalid_response(raw_resp)
        if invalid:
            n_invalid_response += 1
            invalid_by_gt[gt] += 1
            if len(invalid_samples) < 10:
                invalid_samples.append({
                    "row": i,
                    "gt": gt,
                    "response": (raw_resp or "")[:120],
                })

        pred = extract_pred_label(it, classes)
        if pred is None:
            n_unknown_pred += 1
        y_true.append(gt)
        y_pred.append(pred if pred is not None else "<none>")

        topk = extract_first_token_topk(it)
        if topk is not None:
            n_with_logprob += 1
            probs = class_score_from_topk(topk, classes)
            score = probs.get(args.target_class, 0.0)
        else:
            score = 1.0 if pred == args.target_class else 0.0
        scores_target.append(score)
        is_pos_target.append(1 if gt == args.target_class else 0)

    print(f"[INFO] usable={len(y_true)}/{len(items)}, "
          f"unknown_gt={n_unknown_gt}, unknown_pred={n_unknown_pred}, "
          f"with_logprob={n_with_logprob}")
    if n_invalid_response > 0:
        ratio = n_invalid_response / max(len(items), 1)
        print(f"[WARN] invalid_response (空/含<tts_*>/<audio_*>/<|EOT|>) = "
              f"{n_invalid_response}/{len(items)} ({ratio*100:.2f}%)")
        per_gt = ", ".join(f"{k}={v}" for k, v in invalid_by_gt.most_common())
        if per_gt:
            print(f"[WARN] invalid_response 按 GT 分布: {per_gt}")
        for ex in invalid_samples[:5]:
            print(f"       e.g. row={ex['row']} gt={ex['gt']} resp={ex['response']!r}")

    report = per_class_report(y_true,
                              [p if p in classes else "<other>" for p in y_pred],
                              classes)
    print_class_report(report, classes)
    print_confusion(report, classes)

    sweep_rows = threshold_sweep(scores_target, is_pos_target, thresholds)
    fine_thrs = sorted({round(s, 4) for s in scores_target} | set(thresholds))
    fine_rows = threshold_sweep(scores_target, is_pos_target, fine_thrs)
    best = max(fine_rows, key=lambda r: r["f1"]) if fine_rows else None

    print_threshold_table(sweep_rows[: args.max_print_rows], args.target_class)
    if best is not None:
        print("=" * 72)
        print(f"Best F1 on '{args.target_class}': "
              f"threshold={best['threshold']:.4f}, "
              f"precision={fmt_pct(best['precision'])}, "
              f"recall={fmt_pct(best['recall'])}, "
              f"f1={fmt_pct(best['f1'])}, "
              f"tp/fp/fn/tn={best['tp']}/{best['fp']}/{best['fn']}/{best['tn']}")

    save_json({
        "args": vars(args),
        "classes": classes,
        "target_class": args.target_class,
        "n_total_records": len(items),
        "n_used": len(y_true),
        "n_unknown_gt": n_unknown_gt,
        "n_unknown_pred": n_unknown_pred,
        "n_with_logprob": n_with_logprob,
        "n_invalid_response": n_invalid_response,
        "invalid_response_by_gt": dict(invalid_by_gt),
        "invalid_response_samples": invalid_samples,
        "multiclass_report": report,
        "threshold_sweep": sweep_rows,
        "best_threshold": best,
    }, os.path.join(out_dir, "eval_summary.json"))

    csv_path = os.path.join(out_dir, "threshold_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1,tp,fp,fn,tn,predicted_positive,actual_positive\n")
        for r in fine_rows:
            f.write(f"{r['threshold']},{r['precision']:.6f},{r['recall']:.6f},"
                    f"{r['f1']:.6f},{r['tp']},{r['fp']},{r['fn']},{r['tn']},"
                    f"{r['predicted_positive']},{r['actual_positive']}\n")

    print(f"[INFO] saved: {os.path.join(out_dir, 'eval_summary.json')}")
    print(f"[INFO] saved: {csv_path}")


if __name__ == "__main__":
    main()
