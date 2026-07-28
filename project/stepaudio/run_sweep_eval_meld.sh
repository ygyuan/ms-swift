#!/usr/bin/env bash
# StepAudio2-mini MELD GRPO —— 一键"挑 3 个 ckpt 做评测"脚本
#
# 用途:
#   给定一个 GRPO run 目录 (例如 output/meld/grpo/v1-YYYYMMDD-HHMMSS),
#   从中自动选出 3 个代表性 ckpt (默认: early / mid / late),
#   依次调用 run_inference_meld.sh + run_eval_meld.sh,
#   最后打印一张 markdown 汇总表 (accuracy / macro-F1 / weighted-F1 / 每类 recall),
#   并把汇总落盘到 <RUN_DIR>/sweep_eval_summary_<TS>.md 便于事后回溯.
#
# 环境变量 (均可用 KEY=VALUE 命令行覆盖):
#   RUN_DIR         [必填] GRPO 输出目录 (含 checkpoint-XXX 子目录)
#                   例:    output/meld/grpo/v1-20260708-120000
#   CKPT_STEPS      [可选] 要评测的 checkpoint step 列表, 逗号分隔.
#                   例:    "50,150,300"
#                   若不设置, 脚本会自动扫描 RUN_DIR 下所有 checkpoint-*,
#                   按 step 排序后挑 3 个: 第 1 个 / 中间 / 最后 1 个 (early/mid/late).
#                   若 ckpt 少于 3 个则全部评测.
#   PICK_STRATEGY   [可选] "even" (默认, early/mid/late 三等分)
#                          "last3" (只取最后 3 个, 通常最好也最像收敛后表现)
#   VAL_JSONL       [可选] 评测数据集, 默认 project/stepaudio/data_meld/test.jsonl
#   NPROC_PER_NODE  [可选] 推理并行度, 默认继承 run_inference_meld.sh 默认值 (1);
#                          有 4 卡时推荐 NPROC_PER_NODE=4 加速.
#   EVAL_BATCH_SIZE [可选] 推理 batch, 默认 1
#   SKIP_INFER      [可选] "1" 时跳过推理直接跑 eval (要求 RESULT_PATH_* 已存在),
#                          用于"只重跑评测阶段"的场景.
#   FORCE_REINFER   [可选] "1" 时即使 RESULT_PATH 已存在也重新推理, 覆盖旧结果.
#                          默认 "0" (智能复用已存在的推理结果).
#   PRIOR_ALPHA     [可选] 推理端 prior correction (logit-adjustment) 强度.
#                          默认 0.0=关闭 (向后兼容). 推荐 1.5~1.75.
#                          C3 诊断实验证明 alpha=1.75 + uniform target 可让 v7-ckpt150
#                          的 macro-F1 从 41.17% 抬到 43.17% (三条 v9 硬标准全部达标),
#                          无需重训, 是 A 方案的即时增益路径.
#   TRAIN_PRIOR_JSONL [可选] 用来估 P_train 的训练集 jsonl. 默认继承 run_eval_meld.sh
#                          里的 data_meld/train.balanced_letter.jsonl. 仅在 PRIOR_ALPHA!=0
#                          时生效.
#   TARGET_PRIOR    [可选] 目标先验, 'uniform' (默认) / 'train' (no-op) / 'c1:v1,...'.
#                          仅在 PRIOR_ALPHA!=0 时生效.
#
# 用法示例:
#   # 最常见: 训完之后一键跑
#   bash run_sweep_eval_meld.sh RUN_DIR=output/meld/grpo/v1-20260708-120000
#
#   # 显式指定 3 个 step
#   bash run_sweep_eval_meld.sh RUN_DIR=output/meld/grpo/v1-xxx CKPT_STEPS=100,200,300
#
#   # 4 卡并行加速推理
#   bash run_sweep_eval_meld.sh RUN_DIR=output/meld/grpo/v1-xxx NPROC_PER_NODE=4
#
#   # 只重跑评测 (推理结果已在)
#   bash run_sweep_eval_meld.sh RUN_DIR=output/meld/grpo/v1-xxx SKIP_INFER=1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 允许 `bash $0 KEY=VALUE ...` 方式覆盖环境变量 ----
for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
    esac
done

# ---- 参数校验 ----
if [ -z "${RUN_DIR:-}" ]; then
    echo "[ERROR] 请通过 RUN_DIR 指定 GRPO 训练输出目录。" >&2
    echo "        示例: bash $0 RUN_DIR=output/meld/grpo/v1-20260708-120000" >&2
    exit 1
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "[ERROR] RUN_DIR 不存在或不是目录: $RUN_DIR" >&2
    exit 1
fi
RUN_DIR=$(cd "$RUN_DIR" && pwd)   # 转绝对路径

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data_meld/test.r1omni_self_asr.jsonl"}
PICK_STRATEGY=${PICK_STRATEGY:-even}
SKIP_INFER=${SKIP_INFER:-0}
FORCE_REINFER=${FORCE_REINFER:-0}

INFER_SH="$SCRIPT_DIR/run_inference_meld.sh"
EVAL_SH="$SCRIPT_DIR/run_eval_meld.sh"
if [ ! -f "$INFER_SH" ]; then
    echo "[ERROR] 未找到推理脚本: $INFER_SH" >&2
    exit 1
fi
if [ ! -f "$EVAL_SH" ]; then
    echo "[ERROR] 未找到评估脚本: $EVAL_SH" >&2
    exit 1
fi

# ---- 扫描 checkpoint 列表 ----
mapfile -t _ALL_CKPTS < <(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F'checkpoint-' '{print $2 "\t" $0}' \
        | sort -k1 -n \
        | cut -f2
)
_N_CKPTS=${#_ALL_CKPTS[@]}
if [ "$_N_CKPTS" -eq 0 ]; then
    echo "[ERROR] 在 $RUN_DIR 下未发现任何 checkpoint-* 目录。" >&2
    exit 1
fi
echo "[INFO] 在 $RUN_DIR 下共发现 $_N_CKPTS 个 checkpoint"

# ---- 决定要评测哪几个 step ----
SELECTED_CKPTS=()
if [ -n "${CKPT_STEPS:-}" ]; then
    # 显式指定 step 列表: 逐个查找匹配的目录
    IFS=',' read -ra _STEPS <<< "$CKPT_STEPS"
    for s in "${_STEPS[@]}"; do
        s=$(echo "$s" | xargs)   # trim
        [ -z "$s" ] && continue
        _cand="$RUN_DIR/checkpoint-$s"
        if [ -d "$_cand" ]; then
            SELECTED_CKPTS+=("$_cand")
        else
            echo "[WARN] 忽略不存在的 checkpoint: $_cand"
        fi
    done
    if [ ${#SELECTED_CKPTS[@]} -eq 0 ]; then
        echo "[ERROR] CKPT_STEPS 中的所有 step 都在 $RUN_DIR 下找不到对应目录。" >&2
        exit 1
    fi
else
    # 自动挑 3 个
    case "$PICK_STRATEGY" in
        even)
            # 三等分: 索引 0 / (N-1)/2 / N-1
            if [ "$_N_CKPTS" -le 3 ]; then
                SELECTED_CKPTS=("${_ALL_CKPTS[@]}")
            else
                _i_first=0
                _i_mid=$(( (_N_CKPTS - 1) / 2 ))
                _i_last=$(( _N_CKPTS - 1 ))
                SELECTED_CKPTS=("${_ALL_CKPTS[$_i_first]}" "${_ALL_CKPTS[$_i_mid]}" "${_ALL_CKPTS[$_i_last]}")
            fi
            ;;
        last3)
            if [ "$_N_CKPTS" -le 3 ]; then
                SELECTED_CKPTS=("${_ALL_CKPTS[@]}")
            else
                SELECTED_CKPTS=(
                    "${_ALL_CKPTS[$((_N_CKPTS - 3))]}"
                    "${_ALL_CKPTS[$((_N_CKPTS - 2))]}"
                    "${_ALL_CKPTS[$((_N_CKPTS - 1))]}"
                )
            fi
            ;;
        *)
            echo "[ERROR] 未知 PICK_STRATEGY: $PICK_STRATEGY (仅支持 even / last3)" >&2
            exit 1
            ;;
    esac
fi

echo "[INFO] 本轮将评测以下 ${#SELECTED_CKPTS[@]} 个 checkpoint:"
for c in "${SELECTED_CKPTS[@]}"; do
    echo "         $c"
done

# ---- 生成汇总 markdown 输出路径 ----
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_MD="$RUN_DIR/sweep_eval_summary_${TS}.md"
echo "[INFO] 汇总 markdown 将写入: $SUMMARY_MD"

# ---- 选择 python 解释器 (循环内的预热步骤也会用到) ----
# 优先 conda env-3.12.11, 其次 python3
CONDA_PY="/data/miniconda3/envs/env-3.12.11/bin/python"
if [ -x "$CONDA_PY" ]; then
    PYTHON_BIN="$CONDA_PY"
else
    PYTHON_BIN=$(command -v python3 || command -v python)
fi

# ---- HF 动态模块缓存根目录 (trust_remote_code 缓存)
# 多卡 DDP 首次加载同一个 ckpt 时, transformers 会把 configuration_*.py / modeling_*.py
# 复制到该目录; 4 个 rank 并发写入没有互斥锁, 曾出现 rank 加载到"半写"文件 → AttributeError:
# 'module ... has no attribute StepAudio2Config'. 因此每个 ckpt 开跑前:
#   1) 清理旧缓存目录 (方案 B), 避免残留混合状态;
#   2) 单进程 AutoConfig 预热 (方案 A), 把缓存冷启动写盘一次, 后续 4 rank 只读不写.
HF_MODULES_ROOT="${HF_MODULES_CACHE:-$HOME/.cache/huggingface/modules}/transformers_modules"

# ---- 依次跑推理 + 评估 ----
# 记录每个 ckpt 的 eval_summary.json 路径, 供最后汇总用
EVAL_JSONS=()
CKPT_TAGS=()
for CKPT_DIR in "${SELECTED_CKPTS[@]}"; do
    CKPT_TAG=$(basename "$CKPT_DIR")            # e.g. checkpoint-150
    PARENT_TAG=$(basename "$RUN_DIR")           # e.g. v1-20260708-120000
    _INFER_TS=$(date +%Y%m%d_%H%M%S)
    RESULT_PATH="$SCRIPT_DIR/infer_results/result_${PARENT_TAG}_${CKPT_TAG}_test_${_INFER_TS}.jsonl"
    EVAL_OUTPUT_DIR="$SCRIPT_DIR/infer_results/eval_result_${PARENT_TAG}_${CKPT_TAG}_test_${_INFER_TS}"

    echo ""
    echo "================================================================"
    echo "[SWEEP] ==> $CKPT_TAG"
    echo "[SWEEP] CKPT_DIR   = $CKPT_DIR"
    echo "[SWEEP] RESULT     = $RESULT_PATH"
    echo "[SWEEP] EVAL_OUT   = $EVAL_OUTPUT_DIR"
    echo "================================================================"

    # ---- HF dynamic module cache: 清理 + 预热 (仅当本轮真的会推理时才做) ----
    if [ "$SKIP_INFER" != "1" ]; then
        # transformers 会把 ckpt 目录名做 slug 化后作为缓存子目录名.
        # 新版本 (transformers>=5): '-' -> '_hyphen_'   (例: checkpoint-200 -> checkpoint_hyphen_200)
        # 旧版本:                    '-' -> '_'         (例: checkpoint-200 -> checkpoint_200)
        # 两种可能命名一并清理, 保证干净的冷启动.
        _CKPT_BASENAME=$(basename "$CKPT_DIR")                                # e.g. checkpoint-200
        _CKPT_SLUG_NEW=$(echo "$_CKPT_BASENAME" | sed 's/-/_hyphen_/g')        # checkpoint_hyphen_200
        _CKPT_SLUG_OLD=$(echo "$_CKPT_BASENAME" | tr '-' '_')                  # checkpoint_200
        for _slug in "$_CKPT_SLUG_NEW" "$_CKPT_SLUG_OLD"; do
            _cache_dir="$HF_MODULES_ROOT/$_slug"
            if [ -d "$_cache_dir" ]; then
                echo "[SWEEP] 清理旧的 HF 动态模块缓存: $_cache_dir"
                rm -rf "$_cache_dir"
            fi
        done

        # 单进程预热: 触发 transformers 把 configuration_*.py / modeling_*.py 复制到缓存目录,
        # 由于是单进程串行执行, 不会有并发写导致的"半文件"问题.
        # 之后 torchrun 4 rank 再进来时会命中缓存 (跳过 copy), 从而避免 race.
        echo "[SWEEP] 预热 HF 动态模块缓存 (单进程 AutoConfig.from_pretrained)..."
        env CKPT_DIR_FOR_PREWARM="$CKPT_DIR" "$PYTHON_BIN" - <<'PY'
import os, sys, traceback
ckpt = os.environ["CKPT_DIR_FOR_PREWARM"]
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    print(f"[PREWARM] OK: {type(cfg).__name__} loaded from {ckpt}")
except Exception as e:
    print(f"[PREWARM] FAILED: {e!r}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
PY
        if [ $? -ne 0 ]; then
            echo "[ERROR] 预热 HF 动态模块缓存失败, 中止评测。" >&2
            exit 1
        fi
    fi

    # ---- 推理阶段 ----
    if [ "$SKIP_INFER" != "1" ]; then
        # 智能复用: 若已存在同 ckpt 的旧推理结果, 且 FORCE_REINFER=0, 则复用它.
        # 命名匹配 result_{parent}_{ckpt_tag}_test_*.jsonl 的最新一份.
        _REUSED=""
        if [ "$FORCE_REINFER" != "1" ]; then
            _REUSED=$(ls -1t "$SCRIPT_DIR/infer_results/result_${PARENT_TAG}_${CKPT_TAG}_test_"*.jsonl 2>/dev/null | head -n 1 || true)
        fi
        if [ -n "$_REUSED" ] && [ -s "$_REUSED" ]; then
            RESULT_PATH="$_REUSED"
            echo "[SWEEP] 复用已有推理结果 (FORCE_REINFER=0): $RESULT_PATH"
            # 同步重算 EVAL_OUTPUT_DIR, 保持与实际 RESULT_PATH 的时间戳一致
            _reused_base=$(basename "$RESULT_PATH" .jsonl)
            _reused_base_noprefix=${_reused_base#result_}
            EVAL_OUTPUT_DIR="$SCRIPT_DIR/infer_results/eval_result_${_reused_base_noprefix}"
        else
            _INFER_ENV=(
                MODEL_PATH="$CKPT_DIR"
                RESULT_PATH="$RESULT_PATH"
                VAL_JSONL="$VAL_JSONL"
            )
            [ -n "${NPROC_PER_NODE:-}" ] && _INFER_ENV+=(NPROC_PER_NODE="$NPROC_PER_NODE")
            [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && _INFER_ENV+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
            [ -n "${EVAL_BATCH_SIZE:-}" ] && _INFER_ENV+=(EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE")
            [ -n "${BASE_MODEL:-}" ] && _INFER_ENV+=(BASE_MODEL="$BASE_MODEL")

            echo "[SWEEP] 启动推理: env ${_INFER_ENV[*]} bash $INFER_SH"
            env "${_INFER_ENV[@]}" bash "$INFER_SH"
        fi
    else
        # SKIP_INFER=1: 要求已存在同 ckpt 的旧推理结果
        _REUSED=$(ls -1t "$SCRIPT_DIR/infer_results/result_${PARENT_TAG}_${CKPT_TAG}_test_"*.jsonl 2>/dev/null | head -n 1 || true)
        if [ -z "$_REUSED" ] || [ ! -s "$_REUSED" ]; then
            echo "[ERROR] SKIP_INFER=1 但未找到 $CKPT_TAG 的历史推理结果, 请先跑一次推理。" >&2
            exit 1
        fi
        RESULT_PATH="$_REUSED"
        _reused_base=$(basename "$RESULT_PATH" .jsonl)
        _reused_base_noprefix=${_reused_base#result_}
        EVAL_OUTPUT_DIR="$SCRIPT_DIR/infer_results/eval_result_${_reused_base_noprefix}"
        echo "[SWEEP] 复用已有推理结果 (SKIP_INFER=1): $RESULT_PATH"
    fi

    # ---- 评估阶段 ----
    echo "[SWEEP] 启动评估..."
    RESULT_PATH="$RESULT_PATH" \
        VAL_JSONL="$VAL_JSONL" \
        OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
        PRIOR_ALPHA="${PRIOR_ALPHA:-0.0}" \
        TRAIN_PRIOR_JSONL="${TRAIN_PRIOR_JSONL:-}" \
        TRAIN_PRIOR="${TRAIN_PRIOR:-}" \
        TARGET_PRIOR="${TARGET_PRIOR:-uniform}" \
        bash "$EVAL_SH"

    _eval_json="$EVAL_OUTPUT_DIR/eval_summary.json"
    if [ ! -f "$_eval_json" ]; then
        echo "[ERROR] 评估未生成 $_eval_json" >&2
        exit 1
    fi
    EVAL_JSONS+=("$_eval_json")
    CKPT_TAGS+=("$CKPT_TAG")
done

# ---- 汇总 markdown ----
# PYTHON_BIN 已在循环开始前定义, 此处直接复用即可.

echo ""
echo "================================================================"
echo "[SWEEP] 生成汇总表 -> $SUMMARY_MD"
echo "================================================================"

# 把 tag list 和 json path list 用换行分隔通过 stdin 传给 python, 避免 bash 参数解析歧义
_TAGS_STR=$(printf '%s\n' "${CKPT_TAGS[@]}")
_JSONS_STR=$(printf '%s\n' "${EVAL_JSONS[@]}")

env RUN_DIR="$RUN_DIR" \
    SUMMARY_MD="$SUMMARY_MD" \
    SWEEP_TAGS="$_TAGS_STR" \
    SWEEP_JSONS="$_JSONS_STR" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

run_dir     = os.environ["RUN_DIR"]
summary_md  = os.environ["SUMMARY_MD"]
tags        = [t for t in os.environ["SWEEP_TAGS"].splitlines() if t.strip()]
json_paths  = [p for p in os.environ["SWEEP_JSONS"].splitlines() if p.strip()]

assert len(tags) == len(json_paths), f"tag/json 数量不一致: {len(tags)} vs {len(json_paths)}"

# 加载每个 ckpt 的 eval_summary.json
rows = []
per_class_labels = None
for tag, jp in zip(tags, json_paths):
    with open(jp, "r", encoding="utf-8") as f:
        summary = json.load(f)
    # 我们统一读 multiclass_report (来自 pred_source=logprob 的默认口径)
    rep = summary.get("multiclass_report", {})
    acc = rep.get("accuracy", float("nan"))
    m   = rep.get("macro", {}) or {}
    w   = rep.get("weighted", {}) or {}
    pc  = rep.get("per_class", {}) or {}
    # 类别顺序沿用 summary["classes"] (保持列稳定)
    if per_class_labels is None:
        per_class_labels = summary.get("classes") or list(pc.keys())
    rows.append({
        "tag":        tag,
        "json":       jp,
        "acc":        acc,
        "macro_f1":   m.get("f1", float("nan")),
        "macro_p":    m.get("precision", float("nan")),
        "macro_r":    m.get("recall", float("nan")),
        "weighted_f1":w.get("f1", float("nan")),
        "per_class":  pc,
    })

def _fmt(v):
    try:
        return f"{float(v)*100:.2f}"
    except Exception:
        return "n/a"

lines = []
lines.append(f"# GRPO Sweep Eval Summary")
lines.append("")
lines.append(f"- **RUN_DIR**: `{run_dir}`")
lines.append(f"- **# ckpts evaluated**: {len(rows)}")
lines.append(f"- **pred_source**: `logprob` (top-k restricted argmax over 7 classes)")
lines.append("")

# 主表: accuracy / macro-F1 / weighted-F1
lines.append("## Overall metrics")
lines.append("")
lines.append("| Checkpoint | Accuracy | Macro-F1 | Macro-P | Macro-R | Weighted-F1 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| `{r['tag']}` | {_fmt(r['acc'])} | **{_fmt(r['macro_f1'])}** | "
        f"{_fmt(r['macro_p'])} | {_fmt(r['macro_r'])} | {_fmt(r['weighted_f1'])} |"
    )
lines.append("")

# 每类 recall
if per_class_labels:
    lines.append("## Per-class recall (%)")
    lines.append("")
    header = "| Checkpoint | " + " | ".join(per_class_labels) + " |"
    sep    = "|---|" + "|".join(["---:"] * len(per_class_labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for lab in per_class_labels:
            pc = r["per_class"].get(lab, {}) or {}
            cells.append(_fmt(pc.get("recall")))
        lines.append(f"| `{r['tag']}` | " + " | ".join(cells) + " |")
    lines.append("")

    # 每类 F1 也顺便打一份
    lines.append("## Per-class F1 (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for lab in per_class_labels:
            pc = r["per_class"].get(lab, {}) or {}
            cells.append(_fmt(pc.get("f1")))
        lines.append(f"| `{r['tag']}` | " + " | ".join(cells) + " |")
    lines.append("")

# 最佳 ckpt (按 macro-F1)
best = max(rows, key=lambda r: (float(r["macro_f1"]) if r["macro_f1"] == r["macro_f1"] else -1.0))
lines.append("## Best checkpoint (by macro-F1)")
lines.append("")
lines.append(f"- **{best['tag']}** — accuracy={_fmt(best['acc'])}%, "
             f"macro-F1={_fmt(best['macro_f1'])}%, weighted-F1={_fmt(best['weighted_f1'])}%")
lines.append(f"- eval json: `{best['json']}`")
lines.append("")

# 与 SFT baseline 的对比提示 (硬编码, 便于快速判断是否已超越)
lines.append("## Reference baselines (from history)")
lines.append("")
lines.append("| Baseline | Accuracy | Macro-F1 | Weighted-F1 |")
lines.append("|---|---:|---:|---:|")
lines.append("| SFT LoRA v7 ckpt-150 | 58.47 | 41.17 | 54.27 |")
lines.append("")
lines.append("**目标**: macro-F1 需 >= 41.17 才算超越 SFT baseline; ")
lines.append("同时观察 disgust / surprise / anger 三类 recall 是否恢复到 SFT 水平之上.")
lines.append("")

content = "\n".join(lines) + "\n"
os.makedirs(os.path.dirname(summary_md), exist_ok=True)
with open(summary_md, "w", encoding="utf-8") as f:
    f.write(content)
# 同时把 markdown 内容打印到 stdout, 方便直接看
print(content)
PY

echo ""
echo "[SWEEP] ==============================================================="
echo "[SWEEP]  全部完成. 汇总落盘: $SUMMARY_MD"
echo "[SWEEP] ==============================================================="
