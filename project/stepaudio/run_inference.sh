#!/usr/bin/env bash
# StepAudio2-mini 推理脚本（音频场景 5 分类）
# 任务: 输入一段音频，模型直接输出类别字符串，取值范围:
#       [speech, music, noise, porn, song]
# 因此本脚本默认:
#   - 关闭采样 (temperature=0, top_p=1)，使输出可复现，方便后续做 threshold 评估
#   - max_new_tokens 设小 (默认 8)，分类只需要少量 token
#   - 打开 --logprobs / --top_logprobs，让结果里带上 token 级别的概率分布，
#     供 eval_classification.py 做 threshold / precision / recall / F1 分析
#
# 环境变量（均可覆盖默认值）:
#   MODEL_PATH    [必填] 要推理的模型/checkpoint 路径
#                 - 全参微调: 直接指向 checkpoint-xxx 目录
#                 - LoRA 微调: 指向含 adapter_config.json 的目录, 同时通过 BASE_MODEL 指定基座
#                 - 评估原始模型: 直接指向 HF 模型目录, 例如
#                   /apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini
#   BASE_MODEL    LoRA 基座模型 (仅 LoRA 模式需要, 默认 Step-Audio-2-mini)
#   VAL_JSONL     待推理的 JSONL (默认 project/stepaudio/data/val.jsonl)
#   RESULT_PATH   结果保存路径   (默认 project/stepaudio/infer_results/result_<ckpt>_<ts>.jsonl)
#   MAX_NEW_TOKENS / TEMPERATURE / TOP_P / TOP_LOGPROBS / LOGPROBS  推理超参
#   MAX_SAMPLES   只推理前 N 条 (调试用，0 表示全量)
#   NPROC_PER_NODE       数据并行进程数 (默认 1; >1 时会启用 torch.distributed.run 在多卡上
#                        DDP 切分 val_dataset, 每卡加载一份模型副本, 结果合并到 RESULT_PATH)
#   CUDA_VISIBLE_DEVICES 可用 GPU 列表 (默认根据 NPROC_PER_NODE 自动选为 0,1,...,N-1)
#   EVAL_BATCH_SIZE      每个进程的推理 batch 大小 (默认 1; 增大可提升吞吐, 但需注意:
#                        1) 显存占用线性增长; 2) batch>1 时各样本会做 padding,
#                        请先小批量与 batch=1 对比 logprobs 是否一致再放大)
#
# 用法示例:
#   MODEL_PATH=/path/to/output/v0-xxx/checkpoint-1000 bash run_inference.sh
#   MODEL_PATH=/path/to/lora_ckpt BASE_MODEL=/path/to/base bash run_inference.sh
#   MODEL_PATH=/path/to/checkpoint MAX_SAMPLES=100 bash run_inference.sh
#   # 也支持 KEY=VALUE 直接通过命令行覆盖, 例如:
#   bash run_inference.sh MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=2
#   # 显式指定 RESULT_PATH 以便外层脚本拿到固定路径再喂给 run_eval.sh:
#   bash run_inference.sh MODEL_PATH=/path/to/ckpt RESULT_PATH=/tmp/result.jsonl
#   # 4 卡 DDP 并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=4 bash run_inference.sh
#   # 指定卡 1,2,3 三张卡并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=1,2,3 bash run_inference.sh

set -e

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 允许 `bash run_inference.sh KEY=VALUE ...` 这种方式覆盖环境变量,
# 同时把这些 KEY=VALUE 从 $@ 中剔除, 避免被当成 swift infer 的额外参数透传。
PASS_THROUGH_ARGS=()
for kv in "$@"; do
    case "$kv" in
        *=*)
            export "$kv"
            ;;
        *)
            PASS_THROUGH_ARGS+=("$kv")
            ;;
    esac
done
set -- "${PASS_THROUGH_ARGS[@]}"

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data/val.jsonl"}

# 分类任务默认参数（覆盖原来面向 chat/tts 的默认值）
# 注意: swift infer 没有 --do_sample 参数；temperature=0 即等价于贪心解码（do_sample=False）。
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
LOGPROBS=${LOGPROBS:-true}
TOP_LOGPROBS=${TOP_LOGPROBS:-20}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
if ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$EVAL_BATCH_SIZE" -lt 1 ]; then
    echo "[ERROR] EVAL_BATCH_SIZE 必须是 >=1 的整数, 当前=$EVAL_BATCH_SIZE" >&2
    exit 1
fi

# 多卡并行推理: NPROC_PER_NODE>1 时 swift CLI 会自动切换到 torch.distributed.run，
# 在每张卡上起一个进程，并在 swift/pipelines/infer/infer.py 里按 rank 对 val_dataset 调用
# dataset.shard(world_size, rank) 切片，最后合并写到 RESULT_PATH。
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [ "$NPROC_PER_NODE" -lt 1 ]; then
    echo "[ERROR] NPROC_PER_NODE 必须是 >=1 的整数, 当前=$NPROC_PER_NODE" >&2
    exit 1
fi

# 必须显式传入 MODEL_PATH (不再自动定位最新 checkpoint)
if [ -z "${MODEL_PATH:-}" ]; then
    echo "[ERROR] 请通过环境变量 MODEL_PATH 指定要推理的模型/checkpoint 目录。" >&2
    echo "        示例: MODEL_PATH=/path/to/output/v0-xxx/checkpoint-1000 bash $0" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH 不存在或不是目录: $MODEL_PATH" >&2
    exit 1
fi

# 区分 LoRA / full 权重：
#   - LoRA 产物目录特征是存在 adapter_config.json，此时需要同时给 swift infer
#     传 --model <base> 和 --adapters <adapter_dir>；否则仅 --model <full_ckpt> 即可。
BASE_MODEL=${BASE_MODEL:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
ADAPTERS=""
if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    ADAPTERS="$MODEL_PATH"
    MODEL_FOR_INFER="$BASE_MODEL"
else
    MODEL_FOR_INFER="$MODEL_PATH"
fi

# RESULT_PATH 默认带上 checkpoint 名与时间戳，避免不同实验互相覆盖
if [ -z "${RESULT_PATH+x}" ]; then
    CKPT_TAG=$(basename "$MODEL_PATH")
    PARENT_TAG=$(basename "$(dirname "$MODEL_PATH")")
    TS=$(date +%Y%m%d_%H%M%S)
    RESULT_PATH="$SCRIPT_DIR/infer_results/result_${PARENT_TAG}_${CKPT_TAG}_${TS}.jsonl"
fi

mkdir -p "$(dirname "$RESULT_PATH")"

# CUDA_VISIBLE_DEVICES: 未设置时默认根据 NPROC_PER_NODE 自动生成 0,1,...,N-1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
    else
        CUDA_VISIBLE_DEVICES=0
    fi
fi
export CUDA_VISIBLE_DEVICES

# 交叉校验: CUDA_VISIBLE_DEVICES 中的卡数要 >= NPROC_PER_NODE
NUM_VISIBLE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$NUM_VISIBLE" -lt "$NPROC_PER_NODE" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES (可见卡数=$NUM_VISIBLE: $CUDA_VISIBLE_DEVICES) 少于 NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 1
fi

# swift CLI 靠 NPROC_PER_NODE 环境变量判断是否走 torch.distributed.run
export NPROC_PER_NODE

# 如果设置了 MAX_SAMPLES > 0，则派生一个抽样后的小 val 集
EFFECTIVE_VAL_JSONL="$VAL_JSONL"
if [ "$MAX_SAMPLES" -gt 0 ] && [ -f "$VAL_JSONL" ]; then
    SUB_JSONL="$SCRIPT_DIR/infer_results/_subset_${MAX_SAMPLES}.jsonl"
    head -n "$MAX_SAMPLES" "$VAL_JSONL" > "$SUB_JSONL"
    EFFECTIVE_VAL_JSONL="$SUB_JSONL"
    echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES，使用子集: $SUB_JSONL"
fi

echo "[INFO] MODEL_PATH    = $MODEL_PATH"
if [ -n "$ADAPTERS" ]; then
    echo "[INFO] BASE_MODEL    = $BASE_MODEL  [LoRA adapter 模式]"
    echo "[INFO] ADAPTERS      = $ADAPTERS"
else
    echo "[INFO] BASE_MODEL    = [全参检查点, 不需要 --adapters]"
fi
echo "[INFO] VAL_JSONL     = $EFFECTIVE_VAL_JSONL"
echo "[INFO] RESULT_PATH   = $RESULT_PATH"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] NPROC_PER_NODE       = $NPROC_PER_NODE  [>=2 则启用 DDP 数据并行]"
echo "[INFO] EVAL_BATCH_SIZE      = $EVAL_BATCH_SIZE  [每个进程的推理 batch 大小]"
echo "[INFO] temperature=$TEMPERATURE  [0 表示贪心], top_p=$TOP_P, max_new_tokens=$MAX_NEW_TOKENS"
echo "[INFO] logprobs=$LOGPROBS, top_logprobs=$TOP_LOGPROBS"

# 选择 swift 入口
# 注意: 子 shell 中 python/swift 可能因为 rc 钩子被切到别的 conda 环境
# (例如机器上 /root/custom.bashrc 会把 python 改回 env-3.6.8)，
# 因此这里优先用「当前已激活环境」对应的解释器，避免 PATH 漂移。
#   1) 显式 CONDA_SWIFT_BIN
#   2) $CONDA_PREFIX/bin/swift (即激活的 conda env 中的 swift)
#   3) PATH 中的 swift
#   4) $CONDA_PREFIX/bin/python -m swift.cli.main
#   5) PATH 中的 python -m swift.cli.main (最后兜底)
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-}"
if [ -z "$CONDA_SWIFT_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/swift" ]; then
    CONDA_SWIFT_BIN="$CONDA_PREFIX/bin/swift"
fi

if [ -n "$CONDA_SWIFT_BIN" ] && [ -x "$CONDA_SWIFT_BIN" ]; then
    echo "[INFO] 使用 swift CLI: $CONDA_SWIFT_BIN"
    export PATH="$(dirname "$CONDA_SWIFT_BIN"):$PATH"
    SWIFT_CMD=("$CONDA_SWIFT_BIN")
elif command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    echo "[INFO] 未找到 swift CLI，回退到: $CONDA_PREFIX/bin/python -m swift.cli.main"
    SWIFT_CMD=("$CONDA_PREFIX/bin/python" -m swift.cli.main)
else
    echo "[WARN] 未找到 swift CLI 且未检测到 CONDA_PREFIX，回退到 PATH 中的 python -m swift.cli.main"
    echo "       若报 ModuleNotFoundError，请先 conda activate 正确的环境，或显式设置 CONDA_SWIFT_BIN。"
    SWIFT_CMD=(python -m swift.cli.main)
fi

INFER_ARGS=(
    --model "$MODEL_FOR_INFER"
    --model_type step_audio2_mini
    --val_dataset "$EFFECTIVE_VAL_JSONL"
    --result_path "$RESULT_PATH"
    --attn_impl eager
    --torch_dtype bfloat16
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --logprobs "$LOGPROBS"
    --top_logprobs "$TOP_LOGPROBS"
    --max_batch_size "$EVAL_BATCH_SIZE"
    --stream false
)
if [ -n "$ADAPTERS" ]; then
    INFER_ARGS+=(--adapters "$ADAPTERS")
fi

"${SWIFT_CMD[@]}" infer "${INFER_ARGS[@]}" "$@"

echo "[INFO] 推理完成，结果保存在: $RESULT_PATH"

if [ -f "$RESULT_PATH" ]; then
    num_results=$(wc -l < "$RESULT_PATH")
    echo "[INFO] 共生成 $num_results 条结果"
    echo "[INFO] 下一步可直接运行评估:"
    echo "       bash $SCRIPT_DIR/run_eval.sh \\"
    echo "            RESULT_PATH=$RESULT_PATH \\"
    echo "            VAL_JSONL=$VAL_JSONL"
fi
