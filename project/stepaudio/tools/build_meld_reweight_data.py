#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a re-weighted MELD training jsonl for Path-C1 SFT fine-tune.

【背景】
    v7-ckpt150 SFT baseline 上 macro-F1 = 41.17%. C3 诊断实验证明模型已经学到 rare-class
    判别方向, 只是决策阈值被 neutral 先验淹没:
        prior_alpha=1.75 后处理让 macro-F1 提到 43.33%, rare-class recall 全线抬升
        (surprise 46%→63%, joy 25%→39%, sadness 24%→34%, disgust 46%→53%).
    C1 (本工具的目的) 是把 prior correction 的效果内化到权重里, 不再依赖后处理:
        通过 class-conditional oversampling 让每个类 (尤其 rare-class) 在训练 loss 里
        的**样本数量权重**贴近 uniform, 等价于隐式的 balanced softmax / class-weighted CE.

【本工具做什么】
    读入 train.balanced_letter.jsonl (或任意有 label / labels / messages 的 jsonl),
    按目标 per-class 数量做 oversample-with-replacement 或 downsample-without-replacement:
        - 每类都到 --target-per-class (默认 5000, 与 neutral 数量对齐)
        - 或按 --uniform-target 强制均匀分布
        - 或按 --factor-per-class 显式指定每类倍率
    输出一份新的 jsonl 供 --dataset 直接消费, 不需要改 ms-swift 训练循环. 由于 SFT loss
    是"每个样本一次 forward + CE"独立计算, oversample 样本 = 增加该类在 loss 里的权重,
    数学上完全等价于 class-weighted CE (weight_c ∝ new_count_c / old_count_c).

【典型用法】
    # 让每类都到 5000 条 (与 neutral 对齐, rare-class 从 1427 → 5000 = 3.5× oversample)
    python project/stepaudio/tools/build_meld_reweight_data.py \
        --input  project/stepaudio/data_meld/train.balanced_letter.jsonl \
        --output project/stepaudio/data_meld/train.reweight_uniform_letter.jsonl \
        --target-per-class 5000

    # 或直接均匀化 (以最大类为基线, 其他类 oversample 到与它相等):
    python project/stepaudio/tools/build_meld_reweight_data.py \
        --input  project/stepaudio/data_meld/train.balanced_letter.jsonl \
        --output project/stepaudio/data_meld/train.reweight_uniform_letter.jsonl \
        --uniform-target

    # 或对 rare-class 更激进 (rare 6000, joy 4000, neutral 保持):
    python project/stepaudio/tools/build_meld_reweight_data.py \
        --input  project/stepaudio/data_meld/train.balanced_letter.jsonl \
        --output project/stepaudio/data_meld/train.reweight_rare6k_letter.jsonl \
        --target-per-class 6000 --cap-per-class N:4709 --cap-per-class J:4000

【下游训练如何使用】
    在 run_train_sft_lora_meld_reweight.sh (或 patch 现有的 run_train_sft_lora_meld.sh)
    里把 TRAIN_JSONL 指向这个新 jsonl, MODEL_PATH 指向 v7 SFT ckpt (让 lora 继续训, 而非
    从 base 冷启), 用小 lr (5e-7 ~ 1e-6) 短 max_steps (<=300) 精修.

    目标: 让 macro-F1 从 v7 的 41.17% 提到 45%+, 且推理端**不再需要 prior_alpha=1.75 后处理**,
    权重内化 rare-class 判别边界.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# letter <-> word 双向, 兼容两种 label 编码
LETTER_TO_WORD = {
    "S": "surprise", "A": "anger", "N": "neutral", "J": "joy",
    "D": "sadness", "F": "fear", "G": "disgust",
}
WORD_TO_LETTER = {v: k for k, v in LETTER_TO_WORD.items()}


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def extract_label(rec: dict) -> Optional[str]:
    """Try label / labels / assistant-content, 归一化为原字符串 (不做 letter↔word 转换)."""
    lab = rec.get("label") or rec.get("labels")
    if not lab:
        for m in reversed(rec.get("messages", []) or []):
            if m.get("role") == "assistant":
                lab = (m.get("content") or "").strip()
                if lab:
                    break
    if not lab:
        return None
    return str(lab).strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="输入训练 jsonl (通常 train.balanced_letter.jsonl 或 train_letter.jsonl)")
    ap.add_argument("--output", type=Path, required=True,
                    help="输出 jsonl 路径")
    ap.add_argument("--target-per-class", type=int, default=0,
                    help="每类目标样本数. 0 = 不启用 (需 --uniform-target 或 --factor-per-class)")
    ap.add_argument("--uniform-target", action="store_true",
                    help="以最大类为基线, oversample 其他类到与它相等 (常用一键均衡).")
    ap.add_argument("--factor-per-class", action="append", default=[],
                    help="显式倍率, 格式 'CLS:F', 可重复. 例 --factor-per-class J:2.0. "
                         "会覆盖 --target-per-class 对该类的效果.")
    ap.add_argument("--cap-per-class", action="append", default=[],
                    help="每类上限 (硬截断), 格式 'CLS:N', 可重复. 覆盖 --target-per-class.")
    ap.add_argument("--min-per-class", action="append", default=[],
                    help="每类下限 (至少要有 N 条, 不足则 oversample), 格式 'CLS:N', 可重复.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印统计, 不写出.")
    return ap.parse_args()


def parse_per_class_kv(items: List[str], name: str) -> Dict[str, float]:
    """Parse ['J:2.0', 'N:4709'] into dict."""
    out: Dict[str, float] = {}
    for it in items:
        if ":" not in it:
            print(f"[WARN] --{name} 忽略非法项 (需要 CLS:VAL 格式): {it!r}", file=sys.stderr)
            continue
        k, v = it.split(":", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            print(f"[WARN] --{name} 忽略非法项 (VAL 不是数字): {it!r}", file=sys.stderr)
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.input.is_file():
        print(f"[ERROR] 输入不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(args.input)
    print(f"[INFO] 已载入 {len(records)} 条记录 <- {args.input}")

    # ---- 按 label 原样分桶 (不做 letter/word 归一化, 让下游 SFT 保持相同 label 编码) ----
    buckets: Dict[str, List[dict]] = defaultdict(list)
    n_skip = 0
    for r in records:
        lab = extract_label(r)
        if not lab:
            n_skip += 1
            continue
        buckets[lab].append(r)
    if n_skip:
        print(f"[WARN] 跳过 {n_skip} 条无 label 的记录")

    orig_counts = {k: len(v) for k, v in buckets.items()}
    print("[INFO] 原始各类样本数:")
    for k in sorted(orig_counts, key=lambda x: -orig_counts[x]):
        print(f"    {k:<12}  {orig_counts[k]}")

    # ---- 计算每类目标数 ----
    factor = parse_per_class_kv(args.factor_per_class, "factor-per-class")
    cap    = parse_per_class_kv(args.cap_per_class,    "cap-per-class")
    minv   = parse_per_class_kv(args.min_per_class,    "min-per-class")

    if not args.target_per_class and not args.uniform_target and not factor:
        print("[ERROR] 需要至少给一个: --target-per-class / --uniform-target / --factor-per-class",
              file=sys.stderr)
        sys.exit(2)

    if args.uniform_target:
        base = max(orig_counts.values()) if orig_counts else 0
        target = {k: base for k in orig_counts}
        print(f"[INFO] --uniform-target: base={base} (最大类样本数), 全部对齐到该值")
    elif args.target_per_class > 0:
        target = {k: args.target_per_class for k in orig_counts}
        print(f"[INFO] --target-per-class={args.target_per_class}: 全部对齐到该值")
    else:
        # 只有 factor
        target = dict(orig_counts)

    # factor 覆盖
    for k, f in factor.items():
        if k in orig_counts:
            target[k] = int(round(orig_counts[k] * f))
            print(f"[INFO] --factor-per-class {k}:{f} → target[{k}]={target[k]}")
        else:
            print(f"[WARN] factor 里的类 {k!r} 不在数据里, 忽略")

    # cap 覆盖 (硬上限)
    for k, c in cap.items():
        if k in target:
            new = min(int(c), target[k])
            if new != target[k]:
                print(f"[INFO] --cap-per-class {k}:{int(c)} → target[{k}] {target[k]} → {new}")
            target[k] = new

    # min 覆盖 (硬下限)
    for k, m in minv.items():
        if k in target:
            new = max(int(m), target[k])
            if new != target[k]:
                print(f"[INFO] --min-per-class {k}:{int(m)} → target[{k}] {target[k]} → {new}")
            target[k] = new

    print("[INFO] 目标各类样本数:")
    for k in sorted(target, key=lambda x: -target[x]):
        delta = target[k] - orig_counts.get(k, 0)
        sign = "+" if delta > 0 else ("" if delta == 0 else "")
        print(f"    {k:<12}  {orig_counts.get(k, 0)} → {target[k]}  ({sign}{delta:+d})")

    # ---- 生成最终样本池 ----
    out_records: List[dict] = []
    for k, tgt in target.items():
        pool = buckets[k]
        if tgt <= 0 or not pool:
            continue
        if tgt <= len(pool):
            # downsample without replacement (保持随机, 但样本互不重复)
            sampled = rng.sample(pool, tgt)
        else:
            # 先全部保留 (每条 >=1 次), 再对差额 with replacement
            need = tgt - len(pool)
            extra = rng.choices(pool, k=need)
            sampled = list(pool) + extra
        out_records.extend(sampled)

    rng.shuffle(out_records)

    final_counts = Counter()
    for r in out_records:
        lab = extract_label(r)
        if lab:
            final_counts[lab] += 1
    print(f"[INFO] 最终样本总数: {len(out_records)}")
    print("[INFO] 最终各类占比:")
    for k in sorted(final_counts, key=lambda x: -final_counts[x]):
        p = final_counts[k] / max(1, len(out_records))
        print(f"    {k:<12}  {final_counts[k]:>6}  ({p*100:.2f}%)")

    if args.dry_run:
        print("[INFO] --dry-run 生效, 不写出.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 已写出 {len(out_records)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
