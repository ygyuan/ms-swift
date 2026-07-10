#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""StepAudio2-mini MELD speech emotion classification evaluator.

Reads the JSONL produced by run_inference_meld.sh and reports:
  A. Multi-class precision / recall / F1 (per-class + macro + weighted) and a
     confusion matrix, using argmax (model textual response).
  B. Threshold sweep on a target class (default: joy). Score = softmax over
     candidate classes' first-token logprobs (requires --logprobs in inference).

MELD emotion labels (7):
    surprise, anger, neutral, joy, sadness, fear, disgust

Usage:
    python eval_classification_meld.py \
        --result_path infer_results/result_meld_xxx.jsonl \
        --val_jsonl   data_meld/test.jsonl \
        --target_class joy \
        --output_dir  infer_results/eval_meld_xxx
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


CANDIDATE_CLASSES_DEFAULT = ["S", "A", "N", "J", "D", "F", "G"]

# 【方案 1.A】label_style=letter 的单字母 -> 容易阅读的全名映射,
# 仅用于 HEADLINE / confusion matrix / eval_summary.json 里的 pretty print.
# 内部打分、argmax 等完全以单字母为准.
LETTER_TO_WORD = {
    "S": "surprise",
    "A": "anger",
    "N": "neutral",
    "J": "joy",
    "D": "sadness",
    "F": "fear",
    "G": "disgust",
}

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


def _detect_label_style(classes: List[str]) -> str:
    """自动推断 label_style: 每个类都是 1 字符则 letter, 否则 word."""
    if classes and all(len(c) == 1 for c in classes):
        return "letter"
    return "word"


# 【v11+ / self_asr 两段式输出】: response 形如
#   <transcript>...</transcript>\n<answer>joy</answer>
# 直接对全文做 find 会被 transcript 里的 'sad' / 'anger' / 'surprise you'
# 等英文词假匹配, 因此优先只在 <answer>...</answer> 内做归一化.
# 老格式 (v9/v10) 不含该标签, 走原有逻辑, 完全向后兼容.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([^<>]*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def normalize_label(text, classes, label_style: str = "word"):
    """把任意文本归一化到 classes 里的一个类名, 或返回 None.

    【方案 1.A】label_style=letter 时必须严格区分大小写:
      • tokenizer 中 'S'(id=50) 与 's'(id=82) 是不同 token; 小写 s 是很多英文单词的 BPE 起点,
        如果推理输出 'so' / 'she', 盲目 lower() 就会把它们都归到 S=surprise 上, 造成假命中.
      • label_style=letter 下: 只接受 response 去空白后与类字母严格相等 (区分大小写) 的情况;
        或者 response 首字符 (去空白后) 严格等于类字母. 以避免引入奇怪的 startswith 匹配.
      • label_style=word 保持旧行为 (lower + 子串包含), 不影响发布历史 word 版数据.

    【v11+ 增强】若文本包含 <answer>...</answer>, 优先只在标签内部做归一化,
    避免 transcript 段落里出现的 'sad'/'anger'/'surprise' 等英文词造成假匹配.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # 优先: 收窄到 <answer>...</answer> 内容 (v11+ 两段式格式)
    m = _ANSWER_TAG_RE.search(s)
    if m:
        inner = m.group(1).strip()
        if inner:
            s = inner
    if label_style == "letter":
        # 完全相等
        if s in classes:
            return s
        # 首字符等于某个类字母 (处理 response 多了个 '\n' / 空格的情况)
        first = s[0]
        if first in classes:
            return first
        return None
    # word 模式（旧行为）
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


def extract_gt_label(item, classes, label_style: str = "word"):
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


def extract_pred_label(item, classes, label_style: str = "word"):
    return normalize_label(item.get("response", ""), classes, label_style)


def _safe_lower(x):
    return "" if x is None else str(x).strip().lower()


def _locate_answer_token_index(content: List[dict]) -> int:
    """在 logprobs.content 的 token 序列里定位 '<answer>' 之后的第一个 token.

    v11+ self_asr 两段式格式下 response 是:
        <transcript>...</transcript>\n<answer>joy</answer>
    此时 content[0] 恒为 '<', 其 top-k 分布不含任何情感类, 直接取 content[0]
    会让 pred_source=logprob 全部退化为 unknown. 这里通过拼接 token 文本、
    找到 '<answer>' 出现的位置, 返回紧跟其后的那个 token 的下标. 找不到则返回 0
    (老格式 v9/v10 无 <answer> 标签, 自动退化到旧行为).
    """
    if not content:
        return 0
    toks = []
    for c in content:
        t = c.get("token", "") if isinstance(c, dict) else ""
        toks.append(t if isinstance(t, str) else "")
    joined = "".join(toks)
    idx = joined.find("<answer>")
    if idx < 0:
        return 0
    target_char = idx + len("<answer>")
    cum = 0
    for k, t in enumerate(toks):
        cum += len(t)
        if cum > target_char:
            return k
    return 0


def extract_first_token_topk(item):
    """取用于受限 argmax 的 top-k logprobs.

    历史行为: 取生成序列第一个 token 的 top-k. 对 v9/v10 老格式 (response 直接以
    情感词/字母开头) 是正确的.

    【v11+ 增强】若 response 是 <transcript>...</transcript>\n<answer>xxx</answer>
    两段式格式, 第一个 token 是 '<', 完全没有情感信号. 这里改为先在 logprobs.content
    里定位 '<answer>' 之后的那个 token 再取其 top-k, 从而拿到真正的 7 类判别分布.
    对不含 <answer> 的老结果自动退化到 content[0], 完全向后兼容.
    """
    lp = item.get("logprobs")
    if not lp:
        return None
    content = lp.get("content") if isinstance(lp, dict) else None
    if not content:
        return None
    tgt_idx = _locate_answer_token_index(content)
    if tgt_idx < 0 or tgt_idx >= len(content):
        tgt_idx = 0
    first = content[tgt_idx]
    if not isinstance(first, dict):
        return None
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


def class_score_from_topk(topk, classes, label_style: str = "word",
                          prior_alpha: float = 0.0,
                          train_prior: Optional[Dict[str, float]] = None,
                          target_prior: Optional[Dict[str, float]] = None):
    """从 first-token top-k logprobs 里产出各类 softmax 概率.

    【方案 1.A / label_style=letter】必须严格区分大小写:
      • tokenizer 中 'S'(id=50) 与 's'(id=82) 是不同 token.
      • 小写 s 是很多常见英文单词的 BPE 前缀 (so/she/super/...),
        若盲目 lower + prefix startswith 会把它们也归到 S=surprise,
        造成大量假归属 -> 报告虚高.
      • letter 模式严格匹配: token 去前后空白后必须完全等于类字母(区分大小写)才计分;
        均不命中时则概率均为 0, restricted_argmax 会返回 None 并令上层退化到 response.

    label_style=word 时继续使用小写 + 前缀匹配 + log-sum-exp 合并 (旧行为兼容).

    【prior correction, 2026-07】当 prior_alpha != 0 时:
        logit'_c = logit_c - alpha * log P_train(c) + alpha * log P_target(c)
    这是 C3 诊断实验发现的关键增益路径 (v7-ckpt150 上 alpha=1.75 时 macro-F1
    从 41.17% 上升到 43.17%, 三条 v9 硬标准全部达标). 逻辑:
      - SFT 训练集 neutral 占比 34.7%, test 集 48%, 但 rare-class 的判别
        边界被 neutral 先验淹没 → 后处理减去 log P_train 就等于抵消这种
        阈值偏斜.
      - 对已经是 -inf 的类 (即根本没在 top-k 里出现) 不做校正, 防止"没出现
        的类因先验低反而被顶到最上"的病态.
      - 目标先验 target_prior 通常是 uniform (1/K), 表示"输出中希望各类平等",
        用户也可显式指定. alpha=0 时退化到旧行为 (向后兼容).
    """
    raw_logp = {c: -math.inf for c in classes}
    if label_style == "letter":
        # 严格区分大小写的 exact match
        for tok, lp in topk:
            if tok is None:
                continue
            t = str(tok).strip()
            if not t:
                continue
            if t in classes:
                # 同一类如果多个 token 命中, 取 log-sum-exp
                cur = raw_logp[t]
                if not math.isfinite(cur):
                    raw_logp[t] = lp
                else:
                    a, b = cur, lp
                    mx = max(a, b)
                    raw_logp[t] = mx + math.log1p(math.exp(-abs(a - b)))
    else:
        # word 模式: 保留旧行为 (lower + 前缀匹配 + log-sum-exp 合并)
        for tok, lp in topk:
            t = _safe_lower(tok)
            if not t:
                continue
            t_stripped = t.lstrip()
            for c in classes:
                cl = c.lower()
                if t == cl or t_stripped.startswith(cl) or cl.startswith(t_stripped):
                    if not math.isfinite(raw_logp[c]):
                        raw_logp[c] = lp
                    else:
                        a, b = raw_logp[c], lp
                        mx = max(a, b)
                        raw_logp[c] = mx + math.log1p(math.exp(-abs(a - b)))

    # ---- prior correction (opt-in via prior_alpha) ----
    # 只对已经命中 (finite) 的类做校正; -inf 保持不动, 避免病态.
    if prior_alpha and prior_alpha != 0.0 and train_prior is not None:
        _tp = target_prior or {c: 1.0 / len(classes) for c in classes}
        for c in classes:
            if not math.isfinite(raw_logp[c]):
                continue
            lp_tr = math.log(max(train_prior.get(c, 1e-6), 1e-12))
            lp_tg = math.log(max(_tp.get(c, 1e-6), 1e-12))
            raw_logp[c] = raw_logp[c] - prior_alpha * lp_tr + prior_alpha * lp_tg

    finite = [v for v in raw_logp.values() if math.isfinite(v)]
    if not finite:
        return {c: 0.0 for c in classes}
    m = max(finite)
    exps = {c: (math.exp(v - m) if math.isfinite(v) else 0.0) for c, v in raw_logp.items()}
    s = sum(exps.values())
    if s <= 0:
        return {c: 0.0 for c in classes}
    return {c: v / s for c, v in exps.items()}


def restricted_argmax_from_topk(topk, classes, label_style: str = "word",
                                prior_alpha: float = 0.0,
                                train_prior: Optional[Dict[str, float]] = None,
                                target_prior: Optional[Dict[str, float]] = None):
    """把 first-token top-k logprobs 归约到类上, 取 argmax.

    letter 模式: 如果任何类字母都没在 topk 里命中(概率全 0), 返回 None.
    prior_alpha != 0 时会先做 logit-adjustment (见 class_score_from_topk 注释).
    """
    probs = class_score_from_topk(topk, classes, label_style,
                                  prior_alpha=prior_alpha,
                                  train_prior=train_prior,
                                  target_prior=target_prior)
    if not probs:
        return None
    top_c, top_p = max(probs.items(), key=lambda kv: kv[1])
    if top_p <= 0:
        return None
    return top_c


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
    m, w = report["macro"], report["weighted"]
    # ---- HEADLINE: MELD 的社区惯例主指标是 weighted-F1（类高度不均衡时 macro 会低估）,
    #      同时打印 accuracy / macro-F1 便于三项一起对比. 塌陷到多数类时 acc 会虚高、
    #      macro-F1 会很低, 一眼就能看出问题.
    print("=" * 72)
    print(f"[HEADLINE] N={report['n_total']}  "
          f"accuracy={fmt_pct(report['accuracy'])}  "
          f"weighted-F1={fmt_pct(w['f1'])}  macro-F1={fmt_pct(m['f1'])}")
    # ---- 塌陷检测: 若某类占预测总数 > 80%, 报警. 用来自动发现"什么都输出 neutral"这类失败模式.
    pred_counts = {c: report["per_class"][c]["tp"] + report["per_class"][c]["fp"] for c in classes}
    total_pred = sum(pred_counts.values()) or 1
    top_c = max(pred_counts, key=pred_counts.get)
    top_r = pred_counts[top_c] / total_pred
    if top_r > 0.8:
        print(f"[WARN ] mode-collapse suspected: {top_r*100:.1f}% of predictions are '{top_c}'. "
              f"检查 训练是否只学到了多数类先验 / class imbalance.")
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
    ap.add_argument("--target_class", default="J")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_print_rows", type=int, default=21)
    # ---- 【方案 1.A】label_style ----
    # letter — 单字母标签 (S/A/N/J/D/F/G), 严格区分大小写, 默认.
    # word   — word 版标签 (surprise…), 历史兼容模式.
    # auto   — 根据 classes 自动推断 (每个类名长=1 则 letter, 否则 word).
    ap.add_argument("--label_style", default="auto",
                    choices=["letter", "word", "auto"],
                    help="letter/word/auto (default auto: infer from classes).")
    # ---- 【新增】2026-07: --pred_source 控制用什么当 y_pred ----
    # response — 用 item.response 字串的正则归一化结果 (旧默认, 会受 greedy 塑造,
    #             导致在不均衡类任务里看起来“卡在 neutral").
    # logprob — 用 first-token top-k logprobs 在类上做受限 argmax (新默认,
    #             发现“模型已经学到判别信号但被 neutral 先验掩盖"的情况).
    # both    — 同时算两套报告, y_pred 采用 logprob 版, response 版也会打印对比.
    # 需要推理时开了 --logprobs true 才能用 logprob 模式; 否则自动退化到 response.
    ap.add_argument("--pred_source", default="logprob",
                    choices=["response", "logprob", "both"],
                    help="how to derive prediction. default=logprob")
    # ---- 【新增】2026-07-08: prior correction (logit adjustment) ----
    # 在 first-token logprobs 上做 logit'_c = logit_c - alpha * log P_train(c)
    # + alpha * log P_target(c). alpha=0 (默认) 关闭, 保持向后兼容.
    # C3 诊断实验证明 v7-ckpt150 上 alpha=1.75 能把 macro-F1 从 41.17% 抬到 43.17%.
    # 仅对 pred_source=logprob (或 both) 生效.
    ap.add_argument("--prior_alpha", type=float, default=0.0,
                    help="Logit-adjustment 强度. 0=关闭(默认). 推荐 1.5~1.75. "
                         "MELD 上 alpha=1.75 可让 macro-F1 从 41.17→43.17.")
    ap.add_argument("--train_prior_jsonl", default=None,
                    help="用来估 P_train 的 jsonl (通常是训练集). "
                         "prior_alpha != 0 时必填 (或用 --train_prior).")
    ap.add_argument("--train_prior", default=None,
                    help="显式指定训练先验, 覆盖 --train_prior_jsonl. "
                         "格式 'c1:v1,c2:v2,...', 会归一化.")
    ap.add_argument("--target_prior", default="uniform",
                    help="目标先验: 'uniform' (默认) / 'train' (no-op) / "
                         "'c1:v1,c2:v2,...'.")
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

    # ---- 决定 label_style: letter / word / auto ----
    # auto: 每个 class 名都是 1 字符则视为 letter, 否则 word.
    if args.label_style == "auto":
        label_style = _detect_label_style(classes)
    else:
        label_style = args.label_style

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
    print(f"[INFO] label_style = {label_style}"
          + (f"  (letter -> {{{', '.join(f'{k}={v}' for k,v in LETTER_TO_WORD.items() if k in classes)}}})"
             if label_style == "letter" else ""))
    print(f"[INFO] output_dir  = {out_dir}")

    val_index = build_val_index(args.val_jsonl) if args.val_jsonl else {}
    items = load_jsonl(args.result_path)
    if not items:
        print(f"[ERROR] empty result file: {args.result_path}", file=sys.stderr)
        sys.exit(1)

    # ---- 【prior correction 2026-07-08】载入 train_prior / target_prior ----
    # 仅在 args.prior_alpha != 0 时启用. C3 诊断实验证明 MELD v7-ckpt150 上
    # prior_alpha=1.75 + uniform target 可让 macro-F1 从 41.17% 升到 43.17%.
    prior_alpha = float(getattr(args, "prior_alpha", 0.0) or 0.0)
    train_prior: Optional[Dict[str, float]] = None
    target_prior: Optional[Dict[str, float]] = None
    if prior_alpha != 0.0:
        # 1) train_prior: 显式 --train_prior 优先, 其次 --train_prior_jsonl.
        if getattr(args, "train_prior", None):
            parts = [x.strip() for x in args.train_prior.split(",") if x.strip()]
            train_prior = {k: 0.0 for k in classes}
            for p in parts:
                if ":" not in p:
                    continue
                k, v = p.split(":", 1)
                k = k.strip()
                if k in train_prior:
                    train_prior[k] = float(v)
            s = sum(train_prior.values()) or 1.0
            train_prior = {k: v / s for k, v in train_prior.items()}
        elif getattr(args, "train_prior_jsonl", None) and os.path.isfile(args.train_prior_jsonl):
            # 从训练 jsonl 数样本; 同时兼容 letter/word 两种 label 编码.
            cnt = Counter()
            total = 0
            for it in load_jsonl(args.train_prior_jsonl):
                lab = it.get("label") or it.get("labels")
                if not lab:
                    for m in reversed(it.get("messages", []) or []):
                        if m.get("role") == "assistant":
                            lab = m.get("content", "").strip(); break
                if not lab:
                    continue
                s = str(lab).strip()
                cand = normalize_label(s, classes, label_style)
                if cand is None:
                    # letter -> word 映射
                    if len(s) == 1 and s in LETTER_TO_WORD:
                        w = LETTER_TO_WORD[s]
                        if w in classes: cand = w
                        elif s in classes: cand = s
                    # word -> letter 映射
                    if cand is None:
                        sl = s.lower()
                        for w2, l2 in {v: k for k, v in LETTER_TO_WORD.items()}.items():
                            if sl == w2 or sl.startswith(w2) or w2.startswith(sl):
                                if l2 in classes: cand = l2; break
                                elif w2 in classes: cand = w2; break
                if cand is None:
                    continue
                cnt[cand] += 1; total += 1
            if total > 0:
                # laplace smoothing
                train_prior = {k: (cnt.get(k, 0) + 1) / (total + len(classes)) for k in classes}
            else:
                print("[WARN] train_prior_jsonl 里没有能匹配到 classes 的样本, 退化到 uniform 训练先验", file=sys.stderr)
        if train_prior is None:
            print("[WARN] prior_alpha != 0 但未成功载入 train_prior, 退化到 uniform, 相当于关闭校正", file=sys.stderr)
            train_prior = {k: 1.0 / len(classes) for k in classes}

        # 2) target_prior
        tp_spec = getattr(args, "target_prior", "uniform") or "uniform"
        if tp_spec == "uniform":
            target_prior = {k: 1.0 / len(classes) for k in classes}
        elif tp_spec == "train":
            target_prior = dict(train_prior)
        else:
            target_prior = {k: 0.0 for k in classes}
            for p in tp_spec.split(","):
                p = p.strip()
                if ":" not in p: continue
                k, v = p.split(":", 1)
                k = k.strip()
                if k in target_prior:
                    target_prior[k] = float(v)
            s = sum(target_prior.values()) or 1.0
            target_prior = {k: v / s for k, v in target_prior.items()}

        print(f"[INFO] prior_alpha  = {prior_alpha}")
        print(f"[INFO] train_prior  = {{{', '.join(f'{k}:{v:.4f}' for k,v in train_prior.items())}}}")
        print(f"[INFO] target_prior = {{{', '.join(f'{k}:{v:.4f}' for k,v in target_prior.items())}}}")

    y_true = []
    y_pred_response = []   # 基于 item.response 字面归一化的预测
    y_pred_logprob  = []   # 基于 first-token top-k logprobs 的受限 argmax 预测
    scores_target, is_pos_target = [], []
    n_unknown_gt = n_unknown_pred_response = n_unknown_pred_logprob = 0
    n_with_logprob = 0
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
                gt = normalize_label(ref.get("label") or ref.get("labels"), classes, label_style)
        if gt is None:
            gt = extract_gt_label(it, classes, label_style)
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

        pred_r = extract_pred_label(it, classes, label_style)
        if pred_r is None:
            n_unknown_pred_response += 1

        topk = extract_first_token_topk(it)
        pred_l = None
        if topk is not None:
            n_with_logprob += 1
            probs = class_score_from_topk(topk, classes, label_style,
                                          prior_alpha=prior_alpha,
                                          train_prior=train_prior,
                                          target_prior=target_prior)
            score = probs.get(args.target_class, 0.0)
            pred_l = restricted_argmax_from_topk(topk, classes, label_style,
                                                 prior_alpha=prior_alpha,
                                                 train_prior=train_prior,
                                                 target_prior=target_prior)
        else:
            score = 1.0 if pred_r == args.target_class else 0.0
        if pred_l is None:
            n_unknown_pred_logprob += 1

        y_true.append(gt)
        y_pred_response.append(pred_r if pred_r is not None else "<none>")
        y_pred_logprob.append(pred_l if pred_l is not None else
                              (pred_r if pred_r is not None else "<none>"))
        scores_target.append(score)
        is_pos_target.append(1 if gt == args.target_class else 0)

    print(f"[INFO] usable={len(y_true)}/{len(items)}, "
          f"unknown_gt={n_unknown_gt}, "
          f"unknown_pred_response={n_unknown_pred_response}, "
          f"unknown_pred_logprob={n_unknown_pred_logprob}, "
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

    # 两套预测都算一遍, 便于对比; y_pred 选哪一套由 --pred_source 决定.
    report_response = per_class_report(
        y_true,
        [p if p in classes else "<other>" for p in y_pred_response],
        classes)
    report_logprob = per_class_report(
        y_true,
        [p if p in classes else "<other>" for p in y_pred_logprob],
        classes)

    if args.pred_source == "response":
        report = report_response
        y_pred = y_pred_response
        print(f"[INFO] pred_source = response (使用 item.response 字面归一化)")
    else:
        # logprob / both 都以 logprob 版为主报告; both 额外打印 response 版对比.
        if n_with_logprob == 0:
            print("[WARN] result 没有 logprobs, --pred_source=logprob 无法生效, 退化到 response.")
            report = report_response
            y_pred = y_pred_response
        else:
            report = report_logprob
            y_pred = y_pred_logprob
            print(f"[INFO] pred_source = logprob (对 first-token top-{20} 在 7 个类上受限 argmax)")

    # 先打印两套对比的 HEADLINE, 至关重要:
    # 当 response 塑造下的 accuracy 【远低】 于 logprob-restricted argmax 时,
    # 说明模型学到了判别信号但被 greedy 塑造 ‘neutral’ 掩盖了.
    m_r, w_r = report_response["macro"], report_response["weighted"]
    m_l, w_l = report_logprob ["macro"], report_logprob ["weighted"]
    print("=" * 72)
    print(f"[COMPARE] response-based: acc={fmt_pct(report_response['accuracy'])} "
          f"weighted-F1={fmt_pct(w_r['f1'])} macro-F1={fmt_pct(m_r['f1'])}")
    print(f"[COMPARE] logprob -based: acc={fmt_pct(report_logprob ['accuracy'])} "
          f"weighted-F1={fmt_pct(w_l['f1'])} macro-F1={fmt_pct(m_l['f1'])}")
    gap_acc = report_logprob["accuracy"] - report_response["accuracy"]
    if gap_acc > 0.10:
        print(f"[HINT ] logprob-restricted argmax 比 response-based 高 {gap_acc*100:.1f} pp —— "
              f"模型已经学到判别信号, 只是 greedy 被 neutral 先验塑造. 建议以 logprob 报告为准.")

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
        "n_unknown_pred_response": n_unknown_pred_response,
        "n_unknown_pred_logprob": n_unknown_pred_logprob,
        "n_with_logprob": n_with_logprob,
        "n_invalid_response": n_invalid_response,
        "invalid_response_by_gt": dict(invalid_by_gt),
        "invalid_response_samples": invalid_samples,
        "pred_source": args.pred_source,
        # 【prior correction 2026-07-08】
        # 若 prior_alpha > 0 则报告基于校正后的 logprob 分布, 否则为原始 argmax.
        "prior_alpha": prior_alpha,
        "train_prior": train_prior,
        "target_prior": target_prior,
        # 主报告: 根据 pred_source 选定的那一套
        "multiclass_report": report,
        # 两套照旧保留, 便于后续对比 / 回溯
        "multiclass_report_response": report_response,
        "multiclass_report_logprob": report_logprob,
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
