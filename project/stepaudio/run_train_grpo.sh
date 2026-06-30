#!/usr/bin/env bash
# StepAudio2-mini GRPO 训练脚本（基于 MS-SWIFT）
#
# 关键设计：
#   1. 使用 swift rlhf --rlhf_type grpo（之前误用 sft，无法触发 GRPO 流程）。
#   2. 自定义 reward 插件（外置 plugin）：
#        examples/train/grpo/plugin/stepaudio/stepaudio_plugin.py
#      注册了 stepaudio_accuracy / stepaudio_format 两个 ORM。
#      内置 accuracy(MathAccuracy) / format 不适用：
#        - MathAccuracy 走 latex math_verify，stepaudio 标签是普通字符串("porn"等)，全 0；
#        - format 强制 "<think>...</think><answer>...</answer>" 结构，
#          stepaudio assistant 目标只是单个标签 token，反而会被惩罚。
#   3. **不**启用 vLLM：vLLM 暂不支持 step_audio2_mini 的音频多模态输入；
#      使用 transformers native generate 走 GRPO sampling，G=NUM_GENERATIONS。
#   4. 沿用 SFT 脚本里踩过的所有坑：
#        - attn_impl eager（step_audio2_mini 当前唯一支持值）
#        - torch_dtype bfloat16
#        - truncation_strategy delete（避免砍掉 audio_patch 和 mel 数量错位）
#        - PYTORCH_CUDA_ALLOC_CONF expandable_segments
#        - 训练前的 GPU 占用前置检查
# GRPO 显存压力极大（一步要生成 G 条 completion + 计算 ref/old logp），
# 默认走 LoRA + DeepSpeed ZeRO-2；想跑 full 请设 TUNER_TYPE=full + USE_DEEPSPEED=3。
# 注意：gradient_checkpointing 已禁用以避免 transformers 版本兼容性警告。
#   6. GRPO 不需要单独的 val_dataset，使用 --split_dataset_ratio 从 train 切 1%。

set -ex

export LOG_LEVEL=INFO
# Disable wandb（与 event grpo 脚本一致，避免缺包/网络导致 import 报错）
export WANDB_DISABLED=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# 强烈建议 GRPO 从一个已经 SFT 过的 ckpt 继续训练（冷启动 GRPO 几乎不收敛），
# 默认还是指向原始 mini，方便流程跑通；正式训练请用 SFT 后的 best ckpt。
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/stepaudio/grpo"}

# ---------------- Tuner ----------------
# lora（默认，强烈推荐）/ full（高显存）
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-6   # GRPO + full 必须用更小 LR，否则 KL 直接爆炸
else
    DEFAULT_LR=5e-6   # GRPO + LoRA 也比 SFT-LoRA(1e-4) 小一个量级
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（仅 TUNER_TYPE=lora 生效）
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 与 sft 脚本一致：仅注意力 4 个 proj，避免破坏多模态 connector
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- GRPO 关键超参 ----------------
# 每个 prompt 采样多少 completion 用于 group-relative advantage 估计。
# 太小（<2）方差大；太大显存炸。stepaudio 单 prompt 含音频，2~4 通常够用。
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
# 单条 completion 最长生成多少 token。stepaudio 输出基本是 1 个标签词，
# 给 16 token 留点 BOS/EOS 余量足够；过大会显著放慢 sampling。
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-16}
# 探索温度：太低会让 G 条 completion 高度相同（advantage≈0，几乎无梯度）；
# 1.0 是 GRPO 的常用经验值。
TEMPERATURE=${TEMPERATURE:-1.0}
# PPO clip 区间。epsilon_high > epsilon 提供非对称裁剪，鼓励正向更新（DAPO/GRPO 默认）。
EPSILON=${EPSILON:-0.2}
EPSILON_HIGH=${EPSILON_HIGH:-0.28}
# scale_rewards: none / std / batch_std。stepaudio 任务 reward 已是 0/1，不需要再标准化。
SCALE_REWARDS=${SCALE_REWARDS:-none}
# Reward 加权：accuracy 是主指标，format 只用作辅助（轻权重，避免过度规约措辞）
REWARD_WEIGHTS=${REWARD_WEIGHTS:-"1.0 0.1"}
# 梯度裁剪，KL 爆炸时第一道闸门
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---------------- 训练规模 ----------------
NUM_EPOCHS=${NUM_EPOCHS:-1}
# GRPO 一步实际计算量 ≈ batch * G，因此 BATCH_SIZE 一般取 1，靠 G 提供组内对比。
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}

# checkpointing
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
save_total_limit=${save_total_limit:-20}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---------------- 序列 / 截断 ----------------
# 增加到 3072 以容纳较长的音频序列（attention softmax buffer 随 L^2 增长）
MAX_LENGTH=${MAX_LENGTH:-3072}
# 截断策略必须 'delete'，原因详见 SFT 脚本注释（音频 placeholder 不会被同步缩减）
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# ---------------- DeepSpeed ----------------
# - lora: 默认 zero2 即可（参数量小，主要切优化器/梯度）
# - full: 强烈建议 zero3
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
case "$USE_DEEPSPEED" in
    1) DEEPSPEED_STAGE=$([ "$TUNER_TYPE" = "full" ] && echo zero3 || echo zero2) ;;
    2) DEEPSPEED_STAGE=zero2 ;;
    3) DEEPSPEED_STAGE=zero3 ;;
    0) DEEPSPEED_STAGE="" ;;
    *) DEEPSPEED_STAGE="$USE_DEEPSPEED" ;;
esac

# ---------------- DataLoader / 显存 ----------------
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}
# Suppress vLLM-related warnings even though we don't use vLLM, in case some
# transitive import path triggers them.
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"}

# ---------------- 数据集 ----------------
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
# GRPO 直接用 split_dataset_ratio 从 train 划 1% 做 eval（与 event grpo 脚本一致）
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- Reward / Plugin ----------------
PLUGIN_PATH=${PLUGIN_PATH:-"$SWIFT_ROOT/examples/train/grpo/plugin/stepaudio/stepaudio_plugin.py"}
# 默认 reward = stepaudio_accuracy(主) + stepaudio_format(辅)
REWARD_FUNCS=${REWARD_FUNCS:-"stepaudio_accuracy stepaudio_format"}

if [ ! -f "$PLUGIN_PATH" ]; then
    echo "[FATAL] reward plugin 不存在: $PLUGIN_PATH"
    echo "        预期文件路径已写死在脚本里，请检查 ms-swift 仓库是否完整。"
    exit 12
fi

# ---------------- 设备 ----------------
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        export CUDA_VISIBLE_DEVICES=$(nvidia-smi --list-gpus | awk '{printf "%s,", NR-1}' | sed 's/,$//')
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}

echo "[INFO] SWIFT_ROOT  = $SWIFT_ROOT"
echo "[INFO] MODEL_PATH  = $MODEL_PATH"
echo "[INFO] OUTPUT_DIR  = $OUTPUT_DIR"
echo "[INFO] TRAIN_JSONL = $TRAIN_JSONL  (split_ratio=$SPLIT_DATASET_RATIO)"
echo "[INFO] TUNER_TYPE  = $TUNER_TYPE (LR=$LEARNING_RATE, WARMUP_RATIO=$WARMUP_RATIO)"
echo "[INFO] DEEPSPEED   = ${DEEPSPEED_STAGE:-<disabled>}"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] GRPO num_generations=$NUM_GENERATIONS max_completion=$MAX_COMPLETION_LENGTH temperature=$TEMPERATURE"
echo "[INFO] GRPO epsilon=$EPSILON epsilon_high=$EPSILON_HIGH scale_rewards=$SCALE_REWARDS"
echo "[INFO] REWARD_FUNCS=$REWARD_FUNCS  WEIGHTS=$REWARD_WEIGHTS"
echo "[INFO] PLUGIN_PATH=$PLUGIN_PATH"

# ---------------- GPU 占用前置检查（与 SFT 脚本一致） ----------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true

    GPU_PREALLOC_GUARD_MB=${GPU_PREALLOC_GUARD_MB:-1024}
    GPU_PREALLOC_SKIP=${GPU_PREALLOC_SKIP:-0}
    if [ "$GPU_PREALLOC_SKIP" != "1" ]; then
        IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        for _gid in "${_GPU_IDS[@]}"; do
            _used=$(nvidia-smi --id="$_gid" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
            if [ "$_used" -gt "$GPU_PREALLOC_GUARD_MB" ]; then
                echo "[FATAL] GPU $_gid 已被占用 ${_used} MiB (> ${GPU_PREALLOC_GUARD_MB} MiB)，"
                echo "        GRPO 显存压力极大，强烈建议先释放。设 GPU_PREALLOC_SKIP=1 可跳过。"
                exit 11
            fi
        done
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡)"
    fi
fi

# ---------------- swift 入口探测（沿用 SFT 脚本逻辑） ----------------
_check_torch_cuda() {
    "$1" - <<'PY' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

PY_CANDIDATES=(
    "${CONDA_SWIFT_PY:-}"
    "/data/miniconda3/envs/env-3.12.11/bin/python"
    "/data/miniconda3/envs/swift/bin/python"
)

SWIFT_CMD=()
for py in "${PY_CANDIDATES[@]}"; do
    [ -n "$py" ] && [ -x "$py" ] || continue
    if _check_torch_cuda "$py"; then
        echo "[INFO] 使用 python（torch.cuda 可用）: $py"
        export PATH="$(dirname "$py"):$PATH"
        SWIFT_CMD=("$py" -m swift.cli.main)
        break
    else
        echo "[WARN] 跳过 $py（torch 不可用或 CUDA 不匹配）"
    fi
done
if [ ${#SWIFT_CMD[@]} -eq 0 ]; then
    if command -v swift >/dev/null 2>&1; then
        echo "[WARN] 未找到 CUDA 可用的环境，回退到 PATH 上的 swift: $(command -v swift)"
        SWIFT_CMD=(swift)
    else
        echo "[WARN] 未找到 swift CLI，回退到当前 python: $(command -v python) -m swift.cli.main"
        SWIFT_CMD=(python -m swift.cli.main)
    fi
fi

# ---------------- 组装参数 ----------------
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")
if [ "$TUNER_TYPE" = "lora" ]; then
    _tm_normalized=${LORA_TARGET_MODULES//,/ }
    # shellcheck disable=SC2206
    _tm_array=($_tm_normalized)
    TUNER_ARGS+=(
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --target_modules "${_tm_array[@]}"
    )
fi

DS_ARGS=()
if [ -n "$DEEPSPEED_STAGE" ]; then
    DS_ARGS+=(--deepspeed "$DEEPSPEED_STAGE")
fi

# reward_funcs 可能含多个空格分隔的名字，需展开为多个 token
# shellcheck disable=SC2206
REWARD_FUNCS_ARR=($REWARD_FUNCS)
# shellcheck disable=SC2206
REWARD_WEIGHTS_ARR=($REWARD_WEIGHTS)

# ---------------- 启动 GRPO 训练 ----------------
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    "${TUNER_ARGS[@]}" \
    "${DS_ARGS[@]}" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs "${REWARD_FUNCS_ARR[@]}" \
    --reward_weights "${REWARD_WEIGHTS_ARR[@]}" \
    --use_vllm false \
    --dataset "$TRAIN_JSONL" \
    --split_dataset_ratio "$SPLIT_DATASET_RATIO" \
    --load_from_cache_file true \
    --attn_impl eager \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --truncation_strategy $TRUNCATION_STRATEGY \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --epsilon $EPSILON \
    --epsilon_high $EPSILON_HIGH \
    --scale_rewards $SCALE_REWARDS \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_checkpointing false \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit $save_total_limit \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --log_completions true \
    --seed 42 \
    "$@"

echo "[INFO] GRPO 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
