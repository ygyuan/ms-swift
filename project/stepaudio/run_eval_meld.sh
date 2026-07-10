#!/usr/bin/env bash
# StepAudio2-mini MELD 语音情感 7 分类评估脚本
# 读取 run_inference_meld.sh 的输出 (含 logprobs), 计算:
#   - 多类 precision / recall / F1 + confusion
#   - 目标类 (默认 joy) 的 threshold 扫描 + 最佳 F1
#
# 环境变量 (可覆盖):
#   RESULT_PATH    [必填] 推理结果 jsonl (run_inference_meld.sh 生成的那个)
#   VAL_JSONL      原始 test.jsonl, 用于按 key 取更可信的 GT label (默认 data_meld/test.jsonl)
#   TARGET_CLASS   threshold 评估的目标类 (默认 joy)
#   CLASSES        候选类列表, 逗号分隔 (默认 surprise,anger,neutral,joy,sadness,fear,disgust)
#   THRESHOLDS     threshold 列表, 逗号分隔; 空则使用 0.05~0.95 (步长 0.05)
#   OUTPUT_DIR     评估输出目录, 默认 infer_results/eval_<result_name>
#
# 也支持 KEY=VALUE 直接通过命令行覆盖, 例如:
#   bash run_eval_meld.sh RESULT_PATH=infer_results/result_xxx.jsonl TARGET_CLASS=anger
#
# 用法示例:
#   RESULT_PATH=project/stepaudio/infer_results/result_meld_xxx.jsonl bash run_eval_meld.sh
#   bash run_eval_meld.sh RESULT_PATH=project/stepaudio/infer_results/result_meld_xxx.jsonl TARGET_CLASS=sadness

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

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data_meld/test.jsonl"}
# 【v2 · Word 版】默认类名列表使用完整英文单词, 与 UltraEval-Audio 评测流程完全对齐:
#   surprise / anger / neutral / joy / sadness / fear / disgust
# 好处:
#   (1) 与 Step-Audio-2-mini 的 base 输出格式一致 (base 用 word prompt 拿到 55.47% acc),
#   (2) eval_classification_meld.py 的 normalize_label 支持大小写 + 常见简写 (surp/joy/neu 等)
#       所以即使模型输出 "surp"/"neu" 也能被识别.
# 如需 letter 对照实验, 显式传:
#   CLASSES=S,A,N,J,D,F,G  TARGET_CLASS=J  VAL_JSONL=.../test_letter.jsonl
TARGET_CLASS=${TARGET_CLASS:-joy}
CLASSES=${CLASSES:-surprise,anger,neutral,joy,sadness,fear,disgust}
THRESHOLDS=${THRESHOLDS:-}
# ---- 【重要】PRED_SOURCE: 决定评估用什么当预测 ----
#   response  \u2014 用 item.response 字面 (会被 greedy 塑造成 neutral 主导, 不推荐)
#   logprob   \u2014 用 first-token top-k logprobs 在 7 个类上做受限 argmax (默认, 推荐)
#   both      \u2014 同时算并对比打印, y_pred 采用 logprob 版
# 详情见 eval_classification_meld.py 里 --pred_source 参数注释.
PRED_SOURCE=${PRED_SOURCE:-logprob}
# ---- 【prior correction 2026-07-08】opt-in logit-adjustment 后处理 ----
# PRIOR_ALPHA: 校正强度. 0 (默认) = 关闭, 保持向后兼容; 推荐 1.5~1.75.
#   C3 诊断实验证明 v7-ckpt150 上 alpha=1.75 + uniform target 可让
#   macro-F1 从 41.17% 抬到 43.17%, 三条 v9 硬标准 (macro-F1 >= 39%,
#   disgust recall >= 35%, anger recall >= 20%) 全部达标。
# TRAIN_PRIOR_JSONL: 用来估 P_train 的训练集 jsonl (推荐: train.balanced_letter.jsonl).
#   仅在 PRIOR_ALPHA != 0 时必需. 与 TRAIN_PRIOR (显式) 二选一.
# TRAIN_PRIOR: 直接指定训练先验, 格式 'c1:v1,c2:v2,...' (覆盖 TRAIN_PRIOR_JSONL).
# TARGET_PRIOR: 目标先验, 'uniform' (默认) / 'train' (no-op) / 'c1:v1,...'.
PRIOR_ALPHA=${PRIOR_ALPHA:-0.0}
TRAIN_PRIOR_JSONL=${TRAIN_PRIOR_JSONL:-"$SCRIPT_DIR/data_meld/train.balanced_letter.jsonl"}
TRAIN_PRIOR=${TRAIN_PRIOR:-}
TARGET_PRIOR=${TARGET_PRIOR:-uniform}

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
echo "[INFO] PRED_SOURCE  = $PRED_SOURCE  [logprob 用 top-k 受限 argmax; response 用字面; both 对比]"
echo "[INFO] PRIOR_ALPHA  = $PRIOR_ALPHA  [0=关闭, 推荐 1.5~1.75, 见 C3 诊断实验]"
if [ -n "$PRIOR_ALPHA" ] && [ "$PRIOR_ALPHA" != "0" ] && [ "$PRIOR_ALPHA" != "0.0" ]; then
    if [ -n "$TRAIN_PRIOR" ]; then
        echo "[INFO] TRAIN_PRIOR  = $TRAIN_PRIOR  [显式指定]"
    else
        echo "[INFO] TRAIN_PRIOR_JSONL = $TRAIN_PRIOR_JSONL"
    fi
    echo "[INFO] TARGET_PRIOR = $TARGET_PRIOR"
fi
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
    "$SCRIPT_DIR/eval_classification_meld.py"
    --result_path "$RESULT_PATH"
    --target_class "$TARGET_CLASS"
    --classes "$CLASSES"
    --output_dir "$OUTPUT_DIR"
    --pred_source "$PRED_SOURCE"
)
if [ -n "$VAL_JSONL" ] && [ -f "$VAL_JSONL" ]; then
    EVAL_ARGS+=(--val_jsonl "$VAL_JSONL")
fi
if [ -n "$THRESHOLDS" ]; then
    EVAL_ARGS+=(--thresholds "$THRESHOLDS")
fi
# ---- prior correction (opt-in) ----
# 只要 PRIOR_ALPHA 不是字面上的 "0"/"0.0" 就传下去 (让 py 端自己判断 float 是否为 0)
if [ -n "$PRIOR_ALPHA" ] && [ "$PRIOR_ALPHA" != "0" ] && [ "$PRIOR_ALPHA" != "0.0" ]; then
    EVAL_ARGS+=(--prior_alpha "$PRIOR_ALPHA")
    if [ -n "$TRAIN_PRIOR" ]; then
        EVAL_ARGS+=(--train_prior "$TRAIN_PRIOR")
    elif [ -n "$TRAIN_PRIOR_JSONL" ] && [ -f "$TRAIN_PRIOR_JSONL" ]; then
        EVAL_ARGS+=(--train_prior_jsonl "$TRAIN_PRIOR_JSONL")
    fi
    EVAL_ARGS+=(--target_prior "$TARGET_PRIOR")
fi

"$PYTHON_BIN" "${EVAL_ARGS[@]}"

echo "[INFO] 评估完成，结果保存在: $OUTPUT_DIR"
