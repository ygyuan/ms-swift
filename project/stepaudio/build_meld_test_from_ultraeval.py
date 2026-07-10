#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 UltraEval-Audio 的 meld-emo 评测记录 jsonl 严格重建 project/stepaudio/data_meld/test.jsonl。

背景
────
- UltraEval-Audio 在 base Step-Audio-2-mini 上用 word-form prompt 直接推理即可到达
  acc=55.47% (2610 条 test 集), 是本任务的 baseline 上限.
- 我们自己构造的 test.jsonl (基于 MELD 原始 dev/test.csv + dia*_utt*.flac 音频) 走 swift infer
  只能拿到 48.08%, 差距 ~7pp. 差距来源:
    1) 音频文件差异: UltraEval 用的是 HF 版 `TwinkStart/MELD` 的 wav (16-bit 16kHz mono),
       我们用的是从原始视频抽出的 flac (可能采样率不同, 或原视频有多说话人叠加,
       HF 上传时可能做了裁切/归一化);
    2) audio index 与 label 的映射: UltraEval 的 raw/TwinkStart/MELD/test/<id>.wav 里的
       <id> 是 HF dataset row_index (0..2609), 与 MELD 原生 dia*_utt* 命名毫无对应关系;
    3) prompt 排版可能有细微差异 (标点/空格).
- 只要严格用 UltraEval 的音频文件 + prompt + ref, 我们的推理管线就能复现 55.47%,
  这样后续 SFT 训练/评测才能有一致的对照基准.

产物
────
project/stepaudio/data_meld/test.jsonl  (2610 条, 每条如下 schema, 与 UltraEval 严格一致):
{
  "messages": [
    {"role": "user", "content": "<audio><text as UltraEval prompt verbatim>"},
    {"role": "assistant", "content": "<ref label word>"}
  ],
  "audios": ["<absolute path to raw/TwinkStart/MELD/test/<id>.wav>"],
  "label": "<ref label word>",
  "label_str_origin": "<ref label word>",   # 兼容旧字段
  "key": "ultra_<id>",                       # 唯一 key, 便于 sweep 目录命名
  "ultra_id": <int id>                       # 与 UltraEval 完全对齐的 index
}

用法
────
python build_meld_test_from_ultraeval.py \
    --ultra_jsonl /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/UltraEval-Audio/res/Step-Audio-2-mini/meld-emo/2026-04-08_21-57-38.jsonl \
    --ultra_raw_dir /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/UltraEval-Audio/raw \
    --out_jsonl /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/project/stepaudio/data_meld/test.jsonl

如果 --ultra_jsonl 不指定, 会自动在 UltraEval-Audio/res/Step-Audio-2-mini/meld-emo/ 目录下
挑选最新的 *.jsonl.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter


def _find_latest_ultra_jsonl(ultra_res_dir: str) -> str:
    """在 res/Step-Audio-2-mini/meld-emo/ 目录下挑最新的 *.jsonl (排除 overall.json)"""
    candidates = [p for p in glob.glob(os.path.join(ultra_res_dir, "*.jsonl"))]
    if not candidates:
        raise FileNotFoundError(f"no *.jsonl found under {ultra_res_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_ultra_jsonl(path: str):
    """
    UltraEval-Audio 的评测记录 jsonl 每条形如:
        {"type": "prompt",     "id": 0, "data": {"content": [{"role":"user", "contents":[{"type":"audio","value":"raw/.../0.wav"},{"type":"text","value":"listen ..."}]}]}}
        {"type": "inference",  "id": 0, "data": {"content": "neutral"}}
        {"type": "post_process","id": 0, "data": {"content": "neutral"}}
        {"type": "eval",       "id": 0, "data": {"pred": "neutr", "ref": "surprise", "match": 0}}
    我们按 id 收集 (audio_path, prompt_text, ref_label).
    """
    prompt_map = {}   # id -> {"audio": relative_wav_path, "text": prompt_text}
    ref_map = {}      # id -> ref_label (from eval record)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {lineno}: json decode failed: {e}", file=sys.stderr)
                continue
            typ = d.get("type")
            idx = d.get("id")
            if typ == "prompt":
                # content 结构: [{"role":"user", "contents":[...]}]
                content_list = d.get("data", {}).get("content", [])
                if not content_list:
                    continue
                contents = content_list[0].get("contents", [])
                audio_path, text = None, None
                for c in contents:
                    if c.get("type") == "audio":
                        audio_path = c.get("value")
                    elif c.get("type") == "text":
                        text = c.get("value")
                if audio_path is not None and text is not None:
                    prompt_map[idx] = {"audio": audio_path, "text": text}
            elif typ == "eval":
                ref = d.get("data", {}).get("ref")
                if ref is not None:
                    ref_map[idx] = ref
    return prompt_map, ref_map


def build_test_jsonl(
    ultra_jsonl: str,
    ultra_raw_dir: str,
    out_jsonl: str,
    dry_run: bool = False,
) -> None:
    print(f"[INFO] ultra_jsonl   = {ultra_jsonl}")
    print(f"[INFO] ultra_raw_dir = {ultra_raw_dir}")
    print(f"[INFO] out_jsonl     = {out_jsonl}")

    prompt_map, ref_map = parse_ultra_jsonl(ultra_jsonl)
    print(f"[INFO] parsed: prompt records = {len(prompt_map)}, ref records = {len(ref_map)}")

    # 检查一致性
    ids_common = sorted(set(prompt_map.keys()) & set(ref_map.keys()))
    ids_missing_prompt = set(ref_map.keys()) - set(prompt_map.keys())
    ids_missing_ref = set(prompt_map.keys()) - set(ref_map.keys())
    if ids_missing_prompt:
        print(f"[WARN] {len(ids_missing_prompt)} ids have ref but no prompt (skip)")
    if ids_missing_ref:
        print(f"[WARN] {len(ids_missing_ref)} ids have prompt but no ref (skip)")
    print(f"[INFO] common ids = {len(ids_common)}")

    # 检查 prompt 是否所有条目都一致 (UltraEval 一般是一段 prompt 复用到所有条)
    prompt_texts = {p["text"] for p in prompt_map.values()}
    if len(prompt_texts) == 1:
        print(f"[INFO] prompt text is uniform across all rows ✓")
        print(f"[INFO] prompt: {list(prompt_texts)[0][:120]!r}")
    else:
        print(f"[WARN] found {len(prompt_texts)} distinct prompt texts, will keep each row's own text")

    # 检查音频文件是否存在
    n_audio_ok, n_audio_missing = 0, 0
    label_counter = Counter()
    out_lines = []
    for idx in ids_common:
        rel_path = prompt_map[idx]["audio"]
        text_ = prompt_map[idx]["text"]
        ref = ref_map[idx]
        # UltraEval 里 audio value 通常是 "raw/TwinkStart/MELD/test/0.wav"
        # (相对于 UltraEval-Audio 项目根). 拼绝对路径.
        # 用户环境里 UltraEval-Audio/raw 是符号链接, 所以 ultra_raw_dir 可以指向 UltraEval-Audio 根.
        if rel_path.startswith("raw/"):
            abs_audio = os.path.join(os.path.dirname(ultra_raw_dir.rstrip("/")), rel_path)
            # 更稳妥: 直接用 raw dir + 剥离前缀
            abs_audio = os.path.join(ultra_raw_dir.rstrip("/"), rel_path[len("raw/"):])
        elif os.path.isabs(rel_path):
            abs_audio = rel_path
        else:
            abs_audio = os.path.join(ultra_raw_dir.rstrip("/"), rel_path)

        if not os.path.isfile(abs_audio):
            n_audio_missing += 1
            if n_audio_missing <= 5:
                print(f"[WARN] audio missing (id={idx}): {abs_audio}")
            continue
        n_audio_ok += 1
        label_counter[ref] += 1

        # 生成 swift 训练/推理格式的一条
        record = {
            "messages": [
                {"role": "user", "content": f"<audio>{text_}"},
                {"role": "assistant", "content": ref},
            ],
            "audios": [abs_audio],
            "label": ref,
            "label_str_origin": ref,
            "key": f"ultra_{idx}",
            "ultra_id": int(idx),
        }
        out_lines.append(json.dumps(record, ensure_ascii=False))

    print(f"[INFO] wav found = {n_audio_ok}, wav missing = {n_audio_missing}")
    print(f"[INFO] label distribution:")
    for k, v in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"       {k}: {v} ({v / max(1, len(out_lines)) * 100:.1f}%)")

    if dry_run:
        print(f"[DRY-RUN] would write {len(out_lines)} lines to {out_jsonl}, skipped.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fo:
        for l in out_lines:
            fo.write(l + "\n")
    print(f"[OK] wrote {len(out_lines)} records -> {out_jsonl}")


def main():
    default_ultra_res_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "UltraEval-Audio", "res", "Step-Audio-2-mini", "meld-emo",
    ))
    default_ultra_raw_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "UltraEval-Audio", "raw",
    ))
    default_out_jsonl = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "data_meld", "test.jsonl",
    ))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ultra_jsonl", default=None,
                    help=f"UltraEval meld-emo 评测记录 jsonl. 未指定则自动挑 {default_ultra_res_dir} 下最新的.")
    ap.add_argument("--ultra_raw_dir", default=default_ultra_raw_dir,
                    help=f"UltraEval-Audio raw 目录 (含 TwinkStart/MELD/test/*.wav). 默认: {default_ultra_raw_dir}")
    ap.add_argument("--out_jsonl", default=default_out_jsonl,
                    help=f"输出 jsonl. 默认: {default_out_jsonl}")
    ap.add_argument("--dry_run", action="store_true", help="只统计, 不写入")
    args = ap.parse_args()

    ultra_jsonl = args.ultra_jsonl
    if ultra_jsonl is None:
        ultra_jsonl = _find_latest_ultra_jsonl(default_ultra_res_dir)
        print(f"[INFO] auto-picked latest ultra jsonl: {ultra_jsonl}")

    build_test_jsonl(
        ultra_jsonl=ultra_jsonl,
        ultra_raw_dir=args.ultra_raw_dir,
        out_jsonl=args.out_jsonl,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
