#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B 多 checkpoint 批量推理 + 评估脚本
# 训练环境 transformers==4.57.6
# 功能:
#   遍历 OUTPUT_DIR 下所有 checkpoint-* 子目录, 依次执行:
#     1) run_inference.sh  (基于 val.jsonl 生成含 logprobs 的推理 jsonl)
#     2) run_eval.sh       (基于该 jsonl 计算 P/R/F1 + threshold sweep)
#
# 与 stepaudio 版 sweep 的差异 (Qwen3-Omni 专属):
#   - 30B MoE thinker+talker 结构, 单卡吃力, 默认多卡 DDP, 便捷开关 NPROC_PER_NODE
#   - 基座权重 ~132GB, 从 CephFS/NFS 读极慢, 提供 USE_SHM_CACHE 便捷开关将
#     基座缓存到 /dev/shm; sweep 多个 ckpt 时只有第一次触发拷贝, 后续目录已存在直接命中
#     (run_inference.sh 侧默认已开启 USE_SHM_CACHE=1, 这里保持一致, 也允许覆盖为 0)
#   - LoRA / 全参自动区分 (由 run_inference.sh 通过 adapter_config.json 判断),
#     LoRA 场景可显式指定 BASE_MODEL
#   - 分类任务默认 5 类 [speech,music,noise,porn,song], target_class=porn
#
# 重要实现说明 (与 stepaudio 版的关键差异):
#   本项目下的 run_inference.sh 顶部 **没有** `for kv in "$@"` 解析循环,
#   参数只能通过 **环境变量** 传入; 因此 sweep 里必须用
#       env KEY=VALUE ... bash run_inference.sh
#   而不能像 stepaudio 版那样把 KEY=VALUE 当作位置参数直接跟在 bash 后面
#   (那种写法在本项目会静默丢失便捷开关).
#
# 环境变量 (均可覆盖默认值, 也支持 `bash run_sweep_eval.sh KEY=VALUE ...` 形式):
#   OUTPUT_DIR    [必填] 训练输出目录, 例如:
#                 /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/output/qwen3_omni/v1_lora
#   CKPT_PATTERN  匹配 checkpoint 子目录的 glob, 默认 "checkpoint-*"
#   CKPT_LIST     可选: 用逗号或空格分隔的 step 列表, 例如 "1000,1500,2000"
#                 设置后只跑这些 step 的 checkpoint, 忽略 CKPT_PATTERN
#                 也支持形如 "1000-merged" 的 checkpoint 目录名
#   SKIP_EXISTING 1 表示已存在评估结果目录则跳过 (默认 1; 设 0 强制重跑)
#   CONTINUE_ON_ERROR  1 表示某个 checkpoint 失败时继续跑下一个 (默认 1)
#
#   ---- 便捷开关 (会被合并进 INFER_EXTRA_ARGS / EVAL_EXTRA_ARGS) ----
#   NPROC_PER_NODE       DDP 数据并行进程数 (默认不设, 走 run_inference.sh 默认 1;
#                        30B MoE 强烈建议 4)
#   CUDA_VISIBLE_DEVICES 可用 GPU 列表 (默认由 run_inference.sh 依 NPROC_PER_NODE 自动生成)
#   USE_SHM_CACHE        =1 时把基座模型预拷到 /dev/shm 加速多卡加载
#                        (run_inference.sh 默认已经是 1; sweep 场景强烈推荐保持默认)
#   SHM_CACHE_DIR        本地缓存目录 (默认由 run_inference.sh 决定: /dev/shm/swift_model_cache)
#   BASE_MODEL           LoRA 基座模型 (仅 LoRA sweep 需要, 默认走 run_inference.sh 内置默认值)
#   EVAL_BATCH_SIZE      每个进程的推理 batch (默认 1)
#   MAX_SAMPLES          每个 checkpoint 只推理前 N 条 (调试用, 0 表示全量)
#   VAL_JSONL            推理+评估共用的 jsonl (默认 data/val.jsonl;
#                        快速 sanity check 可用 data/val.small.jsonl)
#   TARGET_CLASS         阈值扫描目标类 (默认 porn), 会透传给 run_eval.sh
#   CLASSES              候选类列表, 逗号分隔 (默认走 run_eval.sh 内置)
#
#   ---- 完全透传 (KEY=VALUE 空格拼接串) ----
#   INFER_EXTRA_ARGS   透传给 run_inference.sh 的额外 KEY=VALUE 参数串
#                      例如: INFER_EXTRA_ARGS="ATTN_IMPL=flash_attn TOP_LOGPROBS=10"
#   EVAL_EXTRA_ARGS    透传给 run_eval.sh 的额外 KEY=VALUE 参数串
#                      例如: EVAL_EXTRA_ARGS="THRESHOLDS=0.3,0.5,0.7"
#
# 用法示例:
#   # 4 卡 DDP + /dev/shm 缓存 (推荐):
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/output/qwen3_omni/v1_lora \
#       NPROC_PER_NODE=4 \
#       USE_SHM_CACHE=1
#
#   # 只跑指定的几个 step:
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/path/to/qwen3_omni/v1_lora \
#       CKPT_LIST="1000,1500,2000" \
#       NPROC_PER_NODE=4 USE_SHM_CACHE=1
#
#   # 跑 LoRA + 对应的 merged 全参 (对比 merge 前后是否掉点):
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/path/to/qwen3_omni/v1_lora \
#       CKPT_LIST="1000,1000-merged" \
#       NPROC_PER_NODE=2 SKIP_EXISTING=0
#
#   # 快速 sanity check (小 val 集 + 少样本):
#   bash run_sweep_eval.sh \
#       OUTPUT_DIR=/path/to/qwen3_omni/v1_lora \
#       VAL_JSONL=$PWD/data/val.small.jsonl \
#       MAX_SAMPLES=200 \
#       NPROC_PER_NODE=2 USE_SHM_CACHE=1
#
#   # 强制重跑 (忽略已有评估目录):
#   bash run_sweep_eval.sh OUTPUT_DIR=/path/to/qwen3_omni/v1_lora SKIP_EXISTING=0

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
    echo "        示例: bash $0 OUTPUT_DIR=/path/to/qwen3_omni/v1_lora NPROC_PER_NODE=4 USE_SHM_CACHE=1" >&2
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

# 便捷开关 -> INFER_EXTRA_ARGS
# 注意: 这里只把 KEY=VALUE 追加到字符串, 后续调用时用
#       `env $INFER_EXTRA_ARGS bash run_inference.sh`
#       让 shell 展开成独立 argv 后被 env 认作环境变量赋值.
if [ -n "${NPROC_PER_NODE:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS NPROC_PER_NODE=$NPROC_PER_NODE"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
if [ -n "${USE_SHM_CACHE:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS USE_SHM_CACHE=$USE_SHM_CACHE"
fi
if [ -n "${SHM_CACHE_DIR:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS SHM_CACHE_DIR=$SHM_CACHE_DIR"
fi
if [ -n "${BASE_MODEL:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS BASE_MODEL=$BASE_MODEL"
fi
if [ -n "${EVAL_BATCH_SIZE:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS MAX_SAMPLES=$MAX_SAMPLES"
fi
# VAL_JSONL 需同时透传给 infer 和 eval, 保证两侧 GT 对齐
if [ -n "${VAL_JSONL:-}" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS VAL_JSONL=$VAL_JSONL"
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS VAL_JSONL=$VAL_JSONL"
fi
# 便捷开关 -> EVAL_EXTRA_ARGS
if [ -n "${TARGET_CLASS:-}" ]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS TARGET_CLASS=$TARGET_CLASS"
fi
if [ -n "${CLASSES:-}" ]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS CLASSES=$CLASSES"
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
    # 用 env 把 KEY=VALUE 变量展开后仍能被当作环境变量赋值
    # (直接前置 KEY=VALUE 会被当成命令名; 且 run_inference.sh 顶部没有
    #  `for kv in "$@"` 循环, 位置参数不会被解析成环境变量, 所以必须走 env)
    set +e
    env MODEL_PATH="$ckpt_dir" RESULT_PATH="$RESULT_PATH" $INFER_EXTRA_ARGS bash "$SCRIPT_DIR/run_inference.sh"
    rc_infer=$?
    set -e

    if [ $rc_infer -ne 0 ] || [ ! -f "$RESULT_PATH" ]; then
        echo "[FAIL] 推理失败 (rc=$rc_infer): $ckpt_dir" | tee -a "$SUMMARY_LOG"
        NUM_FAIL=$((NUM_FAIL + 1))
        if [ "$CONTINUE_ON_ERROR" = "1" ]; then
            continue
        else
            exit $rc_infer
        fi
    fi

    # 2) 评估
    # run_eval.sh 顶部有 `for kv in "$@"` 循环, 两种传参方式都行,
    # 这里为了与 infer 保持一致, 统一走 env.
    set +e
    env RESULT_PATH="$RESULT_PATH" OUTPUT_DIR="$EVAL_DIR" $EVAL_EXTRA_ARGS bash "$SCRIPT_DIR/run_eval.sh"
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
