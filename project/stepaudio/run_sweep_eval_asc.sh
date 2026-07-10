#!/usr/bin/env bash
# StepAudio2-mini 多 checkpoint 批量推理 + 评估脚本
#
# 功能:
#   遍历 OUTPUT_DIR 下所有 checkpoint-* 子目录, 依次执行:
#     1) run_inference.sh  (生成 jsonl 推理结果, 含 logprobs)
#     2) run_eval.sh       (基于该 jsonl 计算 P/R/F1 + threshold sweep)
#
# 环境变量 (均可覆盖默认值, 也支持 `bash run_sweep_eval.sh KEY=VALUE ...` 形式):
#   OUTPUT_DIR    [必填] 训练输出目录, 例如:
#                 /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/output/v16-20260629-162422
#   CKPT_PATTERN  匹配 checkpoint 子目录的 glob, 默认 "checkpoint-*"
#   CKPT_LIST     可选: 用逗号或空格分隔的 step 列表, 例如 "200,400,800"
#                 设置后只跑这些 step 的 checkpoint, 忽略 CKPT_PATTERN
#   SKIP_EXISTING 1 表示已存在评估结果目录则跳过 (默认 1; 设 0 强制重跑)
#   CONTINUE_ON_ERROR  1 表示某个 checkpoint 失败时继续跑下一个 (默认 1)
#   INFER_EXTRA_ARGS   透传给 run_inference.sh 的额外 KEY=VALUE 参数串
#                      例如: INFER_EXTRA_ARGS="NPROC_PER_NODE=2 MAX_SAMPLES=100"
#   EVAL_EXTRA_ARGS    透传给 run_eval.sh 的额外 KEY=VALUE 参数串
#                      例如: EVAL_EXTRA_ARGS="TARGET_CLASS=porn"
#   NPROC_PER_NODE     便捷开关, 等价于 INFER_EXTRA_ARGS 中追加 NPROC_PER_NODE=...
#                      (默认不设置, 走 run_inference.sh 的默认值 1)
#
# 用法示例:
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/output/v16-20260629-162422 \
#       NPROC_PER_NODE=2
#
#   # 只跑指定的几个 step:
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/path/to/output/v16-xxx \
#       CKPT_LIST="800,1600,2000" \
#       NPROC_PER_NODE=2
#
#   # 强制重跑 (忽略已有评估目录):
#   bash run_sweep_eval.sh OUTPUT_DIR=/path/to/output/v16-xxx SKIP_EXISTING=0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 允许 `bash run_sweep_eval.sh KEY=VALUE ...` 这种方式覆盖环境变量
for kv in "$@"; do
    case "$kv" in
        *=*)
            export "$kv"
            ;;
    esac
done

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "[ERROR] 请通过参数 OUTPUT_DIR 指定训练输出目录。" >&2
    echo "        示例: bash $0 OUTPUT_DIR=/path/to/output/v16-xxx NPROC_PER_NODE=2" >&2
    exit 1
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[ERROR] OUTPUT_DIR 不存在或不是目录: $OUTPUT_DIR" >&2
    exit 1
fi

CKPT_PATTERN=${CKPT_PATTERN:-"checkpoint-*"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
INFER_EXTRA_ARGS=${INFER_EXTRA_ARGS:-""}
EVAL_EXTRA_ARGS=${EVAL_EXTRA_ARGS:-""}

# 把便捷开关 NPROC_PER_NODE 合并进 INFER_EXTRA_ARGS
if [ -n "${NPROC_PER_NODE:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS NPROC_PER_NODE=$NPROC_PER_NODE"
fi

# 收集 checkpoint 列表
declare -a CKPT_DIRS=()
if [ -n "${CKPT_LIST:-}" ]; then
    # 逗号或空格分隔, 转成空格分隔
    LIST_NORM=$(echo "$CKPT_LIST" | tr ',' ' ')
    for step in $LIST_NORM; do
        d="$OUTPUT_DIR/checkpoint-$step"
        if [ -d "$d" ]; then
            CKPT_DIRS+=("$d")
        else
            echo "[WARN] CKPT_LIST 指定的 checkpoint 不存在, 跳过: $d" >&2
        fi
    done
else
    # glob 出所有 checkpoint-*, 按 step 数字排序
    while IFS= read -r d; do
        [ -d "$d" ] && CKPT_DIRS+=("$d")
    done < <(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d -name "$CKPT_PATTERN" \
                | awk -F'checkpoint-' '{print $0"\t"$2}' \
                | sort -k2 -n \
                | cut -f1)
fi

if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    echo "[ERROR] 在 $OUTPUT_DIR 下没有找到匹配 '$CKPT_PATTERN' 的 checkpoint 目录" >&2
    exit 1
fi

PARENT_TAG=$(basename "$OUTPUT_DIR")
INFER_RESULTS_DIR="$SCRIPT_DIR/infer_results"
mkdir -p "$INFER_RESULTS_DIR"

SUMMARY_LOG="$INFER_RESULTS_DIR/sweep_${PARENT_TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] OUTPUT_DIR        = $OUTPUT_DIR"
echo "[INFO] PARENT_TAG        = $PARENT_TAG"
echo "[INFO] 共发现 ${#CKPT_DIRS[@]} 个 checkpoint:"
for d in "${CKPT_DIRS[@]}"; do echo "        - $d"; done
echo "[INFO] INFER_EXTRA_ARGS  = $INFER_EXTRA_ARGS"
echo "[INFO] EVAL_EXTRA_ARGS   = $EVAL_EXTRA_ARGS"
echo "[INFO] SKIP_EXISTING     = $SKIP_EXISTING"
echo "[INFO] CONTINUE_ON_ERROR = $CONTINUE_ON_ERROR"
echo "[INFO] SUMMARY_LOG       = $SUMMARY_LOG"
echo

TS_RUN=$(date +%Y%m%d_%H%M%S)

NUM_OK=0
NUM_FAIL=0
NUM_SKIP=0

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    CKPT_TAG=$(basename "$ckpt_dir")
    RESULT_PATH="$INFER_RESULTS_DIR/result_${PARENT_TAG}_${CKPT_TAG}_${TS_RUN}.jsonl"
    EVAL_DIR="$INFER_RESULTS_DIR/eval_result_${PARENT_TAG}_${CKPT_TAG}_${TS_RUN}"

    # 已经存在 (历史) 评估结果时, 按需跳过
    if [ "$SKIP_EXISTING" = "1" ]; then
        existing_eval=$(find "$INFER_RESULTS_DIR" -maxdepth 1 -type d \
            -name "eval_result_${PARENT_TAG}_${CKPT_TAG}_*" 2>/dev/null | head -n1 || true)
        if [ -n "$existing_eval" ] && [ -f "$existing_eval/eval_summary.json" ]; then
            echo "================================================================"
            echo "[SKIP] $CKPT_TAG 已有评估结果: $existing_eval (设置 SKIP_EXISTING=0 可强制重跑)"
            echo "================================================================"
            NUM_SKIP=$((NUM_SKIP + 1))
            echo "[SKIP] $CKPT_TAG -> $existing_eval" >> "$SUMMARY_LOG"
            continue
        fi
    fi

    echo "================================================================"
    echo "[RUN ] $CKPT_TAG"
    echo "       MODEL_PATH  = $ckpt_dir"
    echo "       RESULT_PATH = $RESULT_PATH"
    echo "       EVAL_DIR    = $EVAL_DIR"
    echo "================================================================"

    # 1) 推理
    set +e
    bash "$SCRIPT_DIR/run_inference.sh" \
        MODEL_PATH="$ckpt_dir" \
        RESULT_PATH="$RESULT_PATH" \
        $INFER_EXTRA_ARGS
    rc_infer=$?
    set -e

    # 判定"推理是否成功":
    #   历史上我们只看 rc_infer, 但 run_inference.sh 里跑签名 / 健康检查这两个
    #   附带脚本时, 如果它们 rc!=0 (曾出现 heredoc 与 DDP stdin 竞态导致
    #   rc=127 的诡异失败) 就会把整个 run_inference.sh 的返回码带偏, 让我们
    #   把一次已经成功的推理误判为失败并跳过评估.
    #   现在的判定放宽为: 只要 RESULT_PATH 存在且行数 > 0, 就认为推理产物可用,
    #   评估阶段可以继续; rc_infer 只作为附加告警.
    result_lines=0
    if [ -f "$RESULT_PATH" ]; then
        result_lines=$(wc -l < "$RESULT_PATH" 2>/dev/null || echo 0)
    fi

    if [ ! -f "$RESULT_PATH" ] || [ "$result_lines" -eq 0 ]; then
        echo "[FAIL] 推理失败 (rc=$rc_infer, 未生成 RESULT_PATH 或结果为空): $ckpt_dir" \
            | tee -a "$SUMMARY_LOG"
        NUM_FAIL=$((NUM_FAIL + 1))
        if [ "$CONTINUE_ON_ERROR" = "1" ]; then
            continue
        else
            exit $rc_infer
        fi
    fi

    if [ "$rc_infer" -ne 0 ]; then
        echo "[WARN] run_inference.sh 返回非零 (rc=$rc_infer), 但结果文件已生成 ($result_lines 行), 继续评估: $RESULT_PATH" \
            | tee -a "$SUMMARY_LOG"
    fi

    # 2) 评估
    set +e
    bash "$SCRIPT_DIR/run_eval.sh" \
        RESULT_PATH="$RESULT_PATH" \
        OUTPUT_DIR="$EVAL_DIR" \
        $EVAL_EXTRA_ARGS
    rc_eval=$?
    set -e

    if [ $rc_eval -ne 0 ]; then
        echo "[FAIL] 评估失败 (rc=$rc_eval): $RESULT_PATH" | tee -a "$SUMMARY_LOG"
        NUM_FAIL=$((NUM_FAIL + 1))
        if [ "$CONTINUE_ON_ERROR" = "1" ]; then
            continue
        else
            exit $rc_eval
        fi
    fi

    echo "[OK  ] $CKPT_TAG -> $EVAL_DIR" | tee -a "$SUMMARY_LOG"
    NUM_OK=$((NUM_OK + 1))
done

echo
echo "================================================================"
echo "[DONE] 全部 checkpoint 处理完毕"
echo "       成功: $NUM_OK | 失败: $NUM_FAIL | 跳过: $NUM_SKIP"
echo "       明细日志: $SUMMARY_LOG"
echo "================================================================"
