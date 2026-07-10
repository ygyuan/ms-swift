#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prior-corrected re-scoring for MELD 7-way emotion classification.

【背景】
    v7-ckpt150 (SFT baseline) 已经把 macro-F1 顶到 41.17%. 混淆矩阵显示所有 rare-class
    的错误几乎全部塌到 neutral (anger→neutral 47%, joy→neutral 64%, sadness→neutral 61%,
    fear→neutral 58%). 但训练集 (train.balanced_letter) neutral 只占 34.7%, test 集
    neutral 却占 48.1% —— 这是典型的"类别先验偏斜"造成的判别阈值倾斜.
    (DPO 六次实验、Path-B 数据修复也全部失败, 说明问题不在 DPO 或 pair 数据本身,
     而在 SFT 端 rare-class 的决策边界被 neutral 先验淹没.)

【本工具做什么】
    读入 swift infer 产出的 result_xxx.jsonl (要求 --logprobs true --top_logprobs 20),
    在 first-token top-k logprobs 上做:
        logit'_c = logit_c - log P_train(c) + log P_target(c)
    然后重新做受限 argmax, 输出:
        - 新的 macro-F1 / weighted-F1 / accuracy / per-class P/R/F1;
        - 新的 confusion matrix;
        - 与"未校正"版的对比 HEADLINE;
        - 若给了 --grid_temperature 或 --grid_alpha, 还会 grid-search 校正强度.
    完全不训练, 是路 C 的最低成本诊断实验:
        - 若 macro-F1 从 41.2% 一跃到 45%+, 说明模型已具备 rare-class 判别能力,
          只是被 neutral 先验掩盖 —— 后续走 C1 (class-weighted / rare oversample SFT)
          就有信心;
        - 若 prior correction 也拉不动, 说明 rare-class 的判别能力本身没被学到,
          必须走 C2 (hard example mining) 甚至 audio encoder 侧的改造.

【用法】
    python project/stepaudio/tools/apply_prior_correction.py \
        --result-jsonl project/stepaudio/infer_results/result_v7-...checkpoint-150_test_...jsonl \
        --val-jsonl    project/stepaudio/data_meld/test.jsonl \
        --train-jsonl  project/stepaudio/data_meld/train.balanced_letter.jsonl \
        --classes S,A,N,J,D,F,G \
        --label-style auto \
        --target-prior uniform \
        --output-dir   project/stepaudio/infer_results/eval_prior_v7_ckpt150
    # 或做 alpha (校正强度) grid:
    #   --grid-alpha 0.0,0.25,0.5,0.75,1.0,1.25,1.5
    # 或做温度 grid (对 logit 做除法, 温度 T>1 让分布更平, T<1 更尖):
    #   --grid-temperature 0.5,0.75,1.0,1.25,1.5,2.0
    # 二者可以同时给, 会做二维 grid 找 best macro-F1.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


LETTER_CLASSES = ["S", "A", "N", "J", "D", "F", "G"]
WORD_CLASSES = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]

# letter -> word 映射, 仅用于打印 / 归一化 GT / normalize response
LETTER_TO_WORD = {
    "S": "surprise", "A": "anger", "N": "neutral", "J": "joy",
    "D": "sadness", "F": "fear", "G": "disgust",
}
WORD_TO_LETTER = {v: k for k, v in LETTER_TO_WORD.items()}


# ---------- io ----------

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- label normalization ----------

def _detect_label_style(classes: List[str]) -> str:
    if classes and all(len(c) == 1 for c in classes):
        return "letter"
    return "word"


def normalize_label(text, classes, label_style: str = "word"):
    """把任意文本归一化到 classes (letter 严格区分大小写, word 用 lower + 子串包含)."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if label_style == "letter":
        if s in classes:
            return s
        first = s[0]
        if first in classes:
            return first
        return None
    s_low = s.lower()
    for c in classes:
        if s_low == c.lower():
            return c
    best_pos, best_cls = None, None
    for c in classes:
        idx = s_low.find(c.lower())
        if idx >= 0 and (best_pos is None or idx < best_pos):
            best_pos, best_cls = idx, c
    return best_cls


def extract_gt_label(item, classes, label_style):
    if item.get("label"):
        cand = normalize_label(item["label"], classes, label_style)
        if cand:
            return cand
    if item.get("labels"):
        cand = normalize_label(item["labels"], classes, label_style)
        if cand:
            return cand
    for m in item.get("messages", []) or []:
        if m.get("role") == "assistant":
            cand = normalize_label(m.get("content", ""), classes, label_style)
            if cand:
                return cand
    return None


# ---------- logprob extraction (same as eval_classification_meld) ----------

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
            if not any(str(t).strip().lower() == str(self_tok).strip().lower()
                       for t, _ in out):
                out.append((self_tok, self_lp))
        except (TypeError, ValueError):
            pass
    return out or None


def raw_class_logits_from_topk(topk, classes, label_style: str) -> Dict[str, float]:
    """从 first-token top-k logprobs 里聚合出 per-class raw logit (log-sum-exp 合并).

    letter 模式: 严格区分大小写, token 去空白后 == 类字母才计.
    word   模式: lower 后前缀匹配 (与 eval_classification_meld 保持一致).

    未命中的类返回 -inf.
    """
    logits = {c: -math.inf for c in classes}
    if label_style == "letter":
        for tok, lp in topk:
            if tok is None:
                continue
            t = str(tok).strip()
            if not t or t not in classes:
                continue
            cur = logits[t]
            if not math.isfinite(cur):
                logits[t] = lp
            else:
                mx = max(cur, lp); mn = min(cur, lp)
                logits[t] = mx + math.log1p(math.exp(mn - mx))
    else:
        for tok, lp in topk:
            if tok is None:
                continue
            t = str(tok).strip().lower()
            if not t:
                continue
            for c in classes:
                cl = c.lower()
                if t == cl or t.startswith(cl) or cl.startswith(t):
                    cur = logits[c]
                    if not math.isfinite(cur):
                        logits[c] = lp
                    else:
                        mx = max(cur, lp); mn = min(cur, lp)
                        logits[c] = mx + math.log1p(math.exp(mn - mx))
                    break  # 一个 token 只贡献给最先匹配到的类, 避免重复
    return logits


# ---------- report utilities ----------

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def per_class_report(y_true, y_pred, classes):
    rows = {}
    n_total = len(y_true)
    n_correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = n_correct / n_total if n_total else 0.0
    confusion = {gt: Counter() for gt in classes}
    for gt, pr in zip(y_true, y_pred):
        if gt is None:
            continue
        pr_eff = pr if pr in classes else ("<none>" if pr is None else "<other>")
        confusion[gt][pr_eff] += 1
    macro_p = macro_r = macro_f = 0.0
    w_p = w_r = w_f = 0.0
    total_sup = 0
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        sup = sum(1 for a in y_true if a == c)
        p, r, f = _prf(tp, fp, fn)
        rows[c] = {"precision": p, "recall": r, "f1": f, "support": sup,
                   "tp": tp, "fp": fp, "fn": fn}
        macro_p += p; macro_r += r; macro_f += f
        w_p += p * sup; w_r += r * sup; w_f += f * sup
        total_sup += sup
    n = max(len(classes), 1)
    macro_p /= n; macro_r /= n; macro_f /= n
    if total_sup:
        w_p /= total_sup; w_r /= total_sup; w_f /= total_sup
    return {
        "accuracy": acc, "n_total": n_total, "n_correct": n_correct,
        "per_class": rows,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "weighted": {"precision": w_p, "recall": w_r, "f1": w_f},
        "confusion": {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


def fmt_pct(x): return f"{x*100:6.2f}%"


def print_report(tag, report, classes):
    m = report["macro"]; w = report["weighted"]
    print(f"[{tag}] acc={fmt_pct(report['accuracy'])} "
          f"weighted-F1={fmt_pct(w['f1'])} macro-F1={fmt_pct(m['f1'])}")
    print(f"  {'cls':<8} {'P':>7} {'R':>7} {'F1':>7} {'sup':>6}  (tp/fp/fn)")
    for c in classes:
        r = report["per_class"][c]
        print(f"  {c:<8} {fmt_pct(r['precision'])} {fmt_pct(r['recall'])} "
              f"{fmt_pct(r['f1'])} {r['support']:>6}  "
              f"({r['tp']}/{r['fp']}/{r['fn']})")


def print_confusion(tag, report, classes):
    print(f"[{tag}] confusion (rows=GT, cols=Pred):")
    cols = list(classes)
    extras = sorted({p for row in report["confusion"].values() for p in row
                     if p not in cols})
    cols += extras
    print(" " * 10 + "".join(f"{c:>8}" for c in cols))
    for gt in classes:
        row = report["confusion"].get(gt, {})
        print(f"  {gt:<8}" + "".join(f"{row.get(c, 0):>8}" for c in cols))


# ---------- prior estimation ----------

def read_train_prior(train_jsonl: str, classes: List[str], label_style: str) -> Dict[str, float]:
    """从训练集 jsonl 里数出各类样本量, 归一化成 prior 概率.

    支持 label / labels / messages[-1](assistant).content 三种字段.
    未命中的类给一个 tiny smoothing 值 (1/len) 避免 log(0).

    【重要】训练集 label 可能是 letter (S/A/...) 也可能是 word (surprise/...) —— 与 result
    侧的 label_style 独立. 因此本函数**同时尝试** letter 和 word 两种归一化,
    并把命中类映射回目标 classes (通过 LETTER_TO_WORD / WORD_TO_LETTER).
    """
    c = Counter()
    total = 0
    if train_jsonl and os.path.isfile(train_jsonl):
        for it in load_jsonl(train_jsonl):
            lab = it.get("label") or it.get("labels")
            if not lab:
                for m in reversed(it.get("messages", []) or []):
                    if m.get("role") == "assistant":
                        lab = m.get("content", "").strip()
                        break
            if not lab:
                continue
            s = str(lab).strip()
            # 依次尝试: 直接匹配 classes; letter->word 映射; word->letter 映射
            cand = normalize_label(s, classes, label_style)
            if cand is None:
                # 训练集是 letter, 但 classes 是 word: 通过 LETTER_TO_WORD 映射
                if len(s) == 1 and s in LETTER_TO_WORD:
                    w = LETTER_TO_WORD[s]
                    if w in classes:
                        cand = w
                    elif s in classes:
                        cand = s
            if cand is None:
                # 训练集是 word, 但 classes 是 letter: 通过 WORD_TO_LETTER 映射
                sl = s.lower()
                for w, l in WORD_TO_LETTER.items():
                    if sl == w or sl.startswith(w) or w.startswith(sl):
                        if l in classes:
                            cand = l
                            break
                        elif w in classes:
                            cand = w
                            break
            if cand is None:
                continue
            c[cand] += 1
            total += 1
    if total == 0:
        # 无训练数据 fallback: 均匀先验
        return {k: 1.0 / len(classes) for k in classes}
    # laplace smoothing
    smoothed = {k: (c.get(k, 0) + 1) / (total + len(classes)) for k in classes}
    return smoothed


def parse_target_prior(spec: str, classes: List[str],
                       train_prior: Dict[str, float]) -> Dict[str, float]:
    """--target-prior 有 3 种取值:
       - "uniform"  : P_target(c) = 1/K
       - "train"    : P_target(c) = P_train(c)  (等于不校正, 用来做 no-op 基线)
       - "c1:v1,c2:v2,..." : 显式指定, 会归一化.
    """
    if spec == "uniform" or spec is None:
        return {k: 1.0 / len(classes) for k in classes}
    if spec == "train":
        return dict(train_prior)
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    tgt = {k: 0.0 for k in classes}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        if k in tgt:
            tgt[k] = float(v)
    s = sum(tgt.values())
    if s <= 0:
        return {k: 1.0 / len(classes) for k in classes}
    return {k: v / s for k, v in tgt.items()}


# ---------- correction core ----------

def apply_correction(raw_logits: Dict[str, float],
                     train_prior: Dict[str, float],
                     target_prior: Dict[str, float],
                     alpha: float,
                     temperature: float) -> Dict[str, float]:
    """logit'_c = (logit_c - alpha * log P_train(c) + alpha * log P_target(c)) / T

    alpha=0     : 不校正 (等价于原始 logit).
    alpha=1     : 完整 Bayes 校正 (log-likelihood ratio).
    alpha>1     : 过校正, rare-class 更被抬起来 (适合 recall-oriented).
    temperature : 对最终 logit 做除法, T<1 更尖, T>1 更平.

    对 -inf logit (即某类根本没在 top-k 里出现) 直接保留 -inf, 避免 log-prior 项
    人为把它抬起来 (否则会出现"完全没在候选里的类"因先验低反而被选中的 pathology).
    """
    new = {}
    for c, lg in raw_logits.items():
        if not math.isfinite(lg):
            new[c] = -math.inf
            continue
        adj = lg
        if alpha != 0.0:
            lp_tr = math.log(max(train_prior.get(c, 1e-6), 1e-12))
            lp_tg = math.log(max(target_prior.get(c, 1e-6), 1e-12))
            adj = adj - alpha * lp_tr + alpha * lp_tg
        if temperature and temperature != 1.0:
            adj = adj / temperature
        new[c] = adj
    return new


def logits_to_softmax(logits: Dict[str, float]) -> Dict[str, float]:
    finite = [v for v in logits.values() if math.isfinite(v)]
    if not finite:
        return {k: 0.0 for k in logits}
    m = max(finite)
    exps = {k: (math.exp(v - m) if math.isfinite(v) else 0.0) for k, v in logits.items()}
    s = sum(exps.values())
    if s <= 0:
        return {k: 0.0 for k in logits}
    return {k: v / s for k, v in exps.items()}


def argmax_class(probs: Dict[str, float]) -> Optional[str]:
    if not probs:
        return None
    top = max(probs.items(), key=lambda kv: kv[1])
    if top[1] <= 0:
        return None
    return top[0]


# ---------- main pipeline ----------

def build_predictions(items, classes, label_style, train_prior, target_prior,
                      alpha, temperature):
    """跑一遍所有样本, 返回 (y_true, y_pred_raw, y_pred_corrected, n_with_lp, n_no_lp)."""
    y_true, y_raw, y_cor = [], [], []
    n_lp = n_no_lp = 0
    for it in items:
        gt = normalize_label(it.get("label") or it.get("labels"), classes, label_style)
        if gt is None:
            gt = extract_gt_label(it, classes, label_style)
        if gt is None:
            continue
        topk = extract_first_token_topk(it)
        if topk is None:
            n_no_lp += 1
            # 退化到 response 字面匹配
            pr = normalize_label(it.get("response", ""), classes, label_style)
            y_true.append(gt); y_raw.append(pr or "<none>"); y_cor.append(pr or "<none>")
            continue
        n_lp += 1
        raw = raw_class_logits_from_topk(topk, classes, label_style)
        # raw pred: alpha=0, T=1
        raw_probs = logits_to_softmax(raw)
        pr_raw = argmax_class(raw_probs)
        if pr_raw is None:
            pr_raw = normalize_label(it.get("response", ""), classes, label_style)
        # corrected
        cor = apply_correction(raw, train_prior, target_prior, alpha, temperature)
        cor_probs = logits_to_softmax(cor)
        pr_cor = argmax_class(cor_probs)
        if pr_cor is None:
            pr_cor = pr_raw
        y_true.append(gt)
        y_raw.append(pr_raw or "<none>")
        y_cor.append(pr_cor or "<none>")
    return y_true, y_raw, y_cor, n_lp, n_no_lp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-jsonl", required=True,
                    help="swift infer 产出的 result_xxx.jsonl (需含 top_logprobs)")
    ap.add_argument("--val-jsonl", default=None,
                    help="对应的测试集 jsonl (可选, 用来兜底 GT)")
    ap.add_argument("--train-jsonl", default=None,
                    help="训练集 jsonl, 用来估 P_train. 未给则用 --train-prior 或均匀.")
    ap.add_argument("--train-prior", default=None,
                    help="显式指定训练先验, 覆盖 --train-jsonl. 格式 'S:0.1,A:0.1,N:0.35,...'")
    ap.add_argument("--target-prior", default="uniform",
                    help="目标先验: 'uniform' (默认) / 'train' (no-op) / 'c1:v1,c2:v2,...'")
    ap.add_argument("--classes", default=",".join(LETTER_CLASSES),
                    help="逗号分隔. 若每项 1 字符则 letter 模式, 否则 word.")
    ap.add_argument("--label-style", default="auto",
                    choices=["letter", "word", "auto"])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="校正强度. 0=不校正, 1=完整 Bayes, >1 过校正.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax 温度. <1 尖, >1 平.")
    ap.add_argument("--grid-alpha", default=None,
                    help="逗号分隔, 例 '0,0.25,0.5,0.75,1.0,1.25,1.5'. 会做 grid.")
    ap.add_argument("--grid-temperature", default=None,
                    help="逗号分隔, 例 '0.5,0.75,1.0,1.25,1.5,2.0'. 会做 grid.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default="prior_corr",
                    help="打印和文件名前缀")
    return ap.parse_args()


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    label_style = (_detect_label_style(classes) if args.label_style == "auto"
                   else args.label_style)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.result_jsonl)),
        f"eval_{args.tag}_" + os.path.splitext(os.path.basename(args.result_jsonl))[0])
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] result_jsonl = {args.result_jsonl}")
    print(f"[INFO] train_jsonl  = {args.train_jsonl}")
    print(f"[INFO] classes      = {classes}  label_style={label_style}")
    print(f"[INFO] output_dir   = {out_dir}")

    # ---- 载入 GT/结果 ----
    items = load_jsonl(args.result_jsonl)
    if not items:
        print(f"[ERROR] empty result: {args.result_jsonl}", file=sys.stderr)
        sys.exit(1)

    # 若 result 里没有 label 字段, 用 val_jsonl 按顺序对齐
    if args.val_jsonl and os.path.isfile(args.val_jsonl):
        val = load_jsonl(args.val_jsonl)
        if len(val) == len(items):
            for it, ref in zip(items, val):
                if not (it.get("label") or it.get("labels")):
                    it["labels"] = ref.get("label") or ref.get("labels")

    # ---- prior ----
    if args.train_prior:
        parts = [x.strip() for x in args.train_prior.split(",") if x.strip()]
        train_prior = {k: 0.0 for k in classes}
        for p in parts:
            k, v = p.split(":", 1)
            k = k.strip()
            if k in train_prior:
                train_prior[k] = float(v)
        s = sum(train_prior.values()) or 1.0
        train_prior = {k: v / s for k, v in train_prior.items()}
    else:
        train_prior = read_train_prior(args.train_jsonl, classes, label_style)
    target_prior = parse_target_prior(args.target_prior, classes, train_prior)

    print(f"[INFO] train_prior  = {{{', '.join(f'{k}:{v:.4f}' for k,v in train_prior.items())}}}")
    print(f"[INFO] target_prior = {{{', '.join(f'{k}:{v:.4f}' for k,v in target_prior.items())}}}")

    # ---- baseline (alpha=0, T=1) ----
    y_true, y_raw, _, n_lp, n_no_lp = build_predictions(
        items, classes, label_style, train_prior, target_prior, 0.0, 1.0)
    report_raw = per_class_report(y_true, y_raw, classes)
    print()
    print("=" * 72)
    print(f"[BASELINE] n_used={len(y_true)}, n_with_logprob={n_lp}, n_no_logprob={n_no_lp}")
    print_report("BASELINE (alpha=0, T=1)", report_raw, classes)
    print()

    # ---- grid search ----
    alphas = ([float(x) for x in args.grid_alpha.split(",") if x.strip()]
              if args.grid_alpha else [args.alpha])
    temps  = ([float(x) for x in args.grid_temperature.split(",") if x.strip()]
              if args.grid_temperature else [args.temperature])

    all_rows = []
    best = None
    for a in alphas:
        for t in temps:
            _, _, y_cor, _, _ = build_predictions(
                items, classes, label_style, train_prior, target_prior, a, t)
            rep = per_class_report(y_true, y_cor, classes)
            row = {
                "alpha": a, "temperature": t,
                "accuracy": rep["accuracy"],
                "macro_f1": rep["macro"]["f1"],
                "weighted_f1": rep["weighted"]["f1"],
                "per_class_recall": {c: rep["per_class"][c]["recall"] for c in classes},
                "per_class_f1":     {c: rep["per_class"][c]["f1"]     for c in classes},
            }
            all_rows.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = {**row, "_report": rep}

    # ---- grid table ----
    print("=" * 72)
    print(f"[GRID] rows={len(all_rows)} (alpha × temperature) — sorted by macro-F1:")
    print(f"  {'alpha':>6} {'T':>5} {'acc':>7} {'wF1':>7} {'mF1':>7}   "
          f"{'  '.join(f'R_{c}' for c in classes)}")
    for row in sorted(all_rows, key=lambda r: -r["macro_f1"]):
        rec_str = " ".join(f"{row['per_class_recall'][c]:.2f}" for c in classes)
        print(f"  {row['alpha']:>6.2f} {row['temperature']:>5.2f} "
              f"{row['accuracy']*100:>6.2f}% {row['weighted_f1']*100:>6.2f}% "
              f"{row['macro_f1']*100:>6.2f}%   {rec_str}")

    # ---- best detailed report ----
    print()
    print("=" * 72)
    print(f"[BEST]   alpha={best['alpha']} T={best['temperature']}   "
          f"macro-F1={fmt_pct(best['macro_f1'])} (baseline {fmt_pct(report_raw['macro']['f1'])})  "
          f"delta={(best['macro_f1']-report_raw['macro']['f1'])*100:+.2f}pp")
    print_report(f"BEST  alpha={best['alpha']} T={best['temperature']}",
                 best["_report"], classes)
    print()
    print_confusion("BASELINE", report_raw, classes)
    print()
    print_confusion(f"BEST alpha={best['alpha']} T={best['temperature']}",
                    best["_report"], classes)

    # ---- save ----
    save_json({
        "args": vars(args),
        "classes": classes, "label_style": label_style,
        "train_prior": train_prior, "target_prior": target_prior,
        "n_used": len(y_true), "n_with_logprob": n_lp, "n_no_logprob": n_no_lp,
        "baseline_report": report_raw,
        "grid": all_rows,
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "best_report": best["_report"],
    }, os.path.join(out_dir, "prior_correction_summary.json"))
    print(f"\n[INFO] saved: {os.path.join(out_dir, 'prior_correction_summary.json')}")


if __name__ == "__main__":
    main()
