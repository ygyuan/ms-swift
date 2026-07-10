#!/usr/bin/env bash
# StepAudio2-mini 音频场景分类评估脚本
# 读取 run_inference.sh 的输出 (含 logprobs), 计算:
#   - 多类 precision / recall / F1 + confusion
#   - 目标类 (默认 porn) 的 threshold 扫描 + 最佳 F1
#
# 环境变量 (可覆盖):
#   RESULT_PATH    [必填] 推理结果 jsonl (run_inference.sh 生成的那个)
#   VAL_JSONL      原始 val.jsonl, 用于按 key 取更可信的 GT label (默认 data/val.jsonl)
#   TARGET_CLASS   threshold 评估的目标类 (默认 porn)
#   CLASSES        候选类列表, 逗号分隔 (默认 speech,music,noise,porn,song)
#   THRESHOLDS     threshold 列表, 逗号分隔; 空则使用 0.05~0.95 (步长 0.05)
#   OUTPUT_DIR     评估输出目录, 默认 infer_results/eval_<result_name>
#
# 也支持 KEY=VALUE 直接通过命令行覆盖, 例如:
#   bash run_eval.sh RESULT_PATH=infer_results/result_xxx.jsonl TARGET_CLASS=porn
#
# 用法示例:
#   RESULT_PATH=project/stepaudio/infer_results/result_v15_xxx.jsonl bash run_eval.sh
#   bash run_eval.sh RESULT_PATH=project/stepaudio/infer_results/result_v15_xxx.jsonl TARGET_CLASS=song

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 允许 `bash run_eval.sh KEY=VALUE ...` 这种方式覆盖环境变量
for kv in "$@"; do
    case "$kv" in
        *=*)
            export "$kv"
            ;;
    esac
done

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data/val.jsonl"}
TARGET_CLASS=${TARGET_CLASS:-porn}
CLASSES=${CLASSES:-speech,music,noise,porn,song}
THRESHOLDS=${THRESHOLDS:-}

# RESULT_PATH 必须显式传入 (不再自动定位最新推理结果)
if [ -z "${RESULT_PATH:-}" ]; then
    echo "[ERROR] 请通过参数 RESULT_PATH 指定推理结果文件。" >&2
    echo "        示例: bash $0 RESULT_PATH=project/stepaudio/infer_results/result_xxx.jsonl" >&2
    echo "        或:   RESULT_PATH=project/stepaudio/infer_results/result_xxx.jsonl bash $0" >&2
    exit 1
fi
if [ ! -f "$RESULT_PATH" ]; then
    echo "[ERROR] RESULT_PATH 不存在或不是文件: $RESULT_PATH" >&2
    exit 1
fi

if [ -z "${OUTPUT_DIR+x}" ] || [ -z "$OUTPUT_DIR" ]; then
    base=$(basename "$RESULT_PATH" .jsonl)
    OUTPUT_DIR="$SCRIPT_DIR/infer_results/eval_${base}"
fi

echo "[INFO] RESULT_PATH  = $RESULT_PATH"
echo "[INFO] VAL_JSONL    = $VAL_JSONL"
echo "[INFO] TARGET_CLASS = $TARGET_CLASS"
echo "[INFO] CLASSES      = $CLASSES"
echo "[INFO] OUTPUT_DIR   = $OUTPUT_DIR"

# 选择 python 解释器：优先 conda env，其次默认 python
CONDA_PY="/data/miniconda3/envs/env-3.12.11/bin/python"
if [ -x "$CONDA_PY" ]; then
    PYTHON_BIN="$CONDA_PY"
else
    PYTHON_BIN=$(command -v python3 || command -v python)
fi
echo "[INFO] PYTHON_BIN   = $PYTHON_BIN"

EVAL_ARGS=(
    "$SCRIPT_DIR/eval_classification.py"
    --result_path "$RESULT_PATH"
    --target_class "$TARGET_CLASS"
    --classes "$CLASSES"
    --output_dir "$OUTPUT_DIR"
)
if [ -n "$VAL_JSONL" ] && [ -f "$VAL_JSONL" ]; then
    EVAL_ARGS+=(--val_jsonl "$VAL_JSONL")
fi
if [ -n "$THRESHOLDS" ]; then
    EVAL_ARGS+=(--thresholds "$THRESHOLDS")
fi

"$PYTHON_BIN" "${EVAL_ARGS[@]}"

echo "[INFO] 评估完成，结果保存在: $OUTPUT_DIR"
