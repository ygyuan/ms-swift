#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B 训练脚本（基于 MS-SWIFT）
#
# 参考：
#   - examples/models/qwen3_omni/transformers.sh   （单机 2*48GiB）
#   - examples/models/qwen3_omni/zero3.sh          （DeepSpeed ZeRO-3 2*60GiB）
#   - examples/train/multimodal/omni/sft.sh
#   - docs/source/Instruction/Command-line-parameters.md
#
# Qwen3-Omni 关键差异（相对一般 LLM / StepAudio2-mini）：
#   1. 30B-A3B 是 MoE（thinker 用 SparseMoE），ms-swift 已自动把
#      Qwen3OmniMoeThinkerTextSparseMoeBlock 标记为 zero3 leaf module。
#   2. 模型由 thinker(文本/视觉/音频理解) + talker(语音生成) 组成。
#      做 SFT 时仅训 thinker，需要 ENABLE_AUDIO_OUTPUT=0 关闭 talker，
#      否则 zero3 / 显存都会出问题（官方文档明确要求）。
#   3. attn_impl 应使用 flash_attn（不是 eager），否则 30B 序列长一点就会 OOM。
#   4. LoRA + freeze_vit + freeze_aligner 是官方推荐的轻量微调方式，
#      只训 LLM/MoE 的 linear 层，避免 ViT / AudioTower / Projector 也跟着训。
#   5. 多模态 token 数量由环境变量控制：
#      IMAGE_MAX_TOKEN_NUM / VIDEO_MAX_TOKEN_NUM / FPS_MAX_FRAMES。

set -ex

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ============================================================================
# 模型 / 输出路径（可通过环境变量覆盖）
# ============================================================================
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/Qwen/Qwen3-Omni-30B-A3B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/qwen3_omni/v1_lora"}

# 是否在 OUTPUT_DIR 下再自动创建 v<n>-<timestamp> 子目录。
#   true  -> 默认行为，最终 checkpoint 落到 "$OUTPUT_DIR/v0-20260626-153744/..."
#   false -> 直接落到 "$OUTPUT_DIR" 本身（要求该目录为空或不存在，否则会与已有 checkpoint 冲突）
# 对应 ms-swift SftArguments.add_version（默认 True）。
ADD_VERSION=${ADD_VERSION:-false}

# ============================================================================
# 训练超参（可通过环境变量覆盖）
# ============================================================================
# tuner_type: lora（默认，推荐先用小数据验证流程）| full（全参微调，30B MoE 几乎不现实）
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-5
else
    DEFAULT_LR=1e-4
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}
NUM_EPOCHS=${NUM_EPOCHS:-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}

# 30B-A3B（MoE，激活约 3B）+ flash_attn + LoRA + zero3：
#   单卡 batch 1 + grad_accum 4 是较稳的起点（官方 transformers.sh 也是 bs=1, ga=4）。
#   如果显存富裕（>=80GiB / GPU），可以把 BATCH_SIZE 提到 2，GRAD_ACCUM 调到 4。
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}

SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
save_total_limit=${save_total_limit:-5}
LOGGING_STEPS=${LOGGING_STEPS:-5}

# Val 子采样：原 val.jsonl 太大会让每次 eval 几分钟。训练期间用小集监控趋势即可。
# 设为 0 表示不抽样，直接使用完整 VAL_JSONL。
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-5000}
VAL_SAMPLE_SHUFFLE=${VAL_SAMPLE_SHUFFLE:-1}

# Qwen3-Omni 在 flash_attn 下显存压力远小于 eager，可以使用更长上下文。
# 4096 与官方示例一致；如果数据普遍很短可以缩短到 2048 进一步省显存。
MAX_LENGTH=${MAX_LENGTH:-4096}

# 截断策略：保持 'delete' 作为最稳妥的策略——丢弃超长样本，
# 避免 right/left 截断时 <image_pad>/<audio_pad> 占位 token 与实际多模态特征数不一致而 forward 报错。
# 如果你确认数据集里没有任何样本超过 MAX_LENGTH，可以改回默认（不传该参数）。
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# ============================================================================
# Qwen3-Omni 专属：多模态 token 控制 / talker 关闭
# ============================================================================
# IMAGE_MAX_TOKEN_NUM: 单张图最多产出多少视觉 token（默认 1024，越大越精细但显存涨）
# VIDEO_MAX_TOKEN_NUM: 单段视频最多产出多少视觉 token
# FPS_MAX_FRAMES    : 视频抽帧上限
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}

# 训练时关闭 talker（语音输出），否则 zero3 加载会报错，且白白占用大量显存。
# 文档：docs/source/Instruction/Command-line-parameters.md
#   "🔥ENABLE_AUDIO_OUTPUT: 默认为None... 若使用zero3进行训练，请设置为False。"
export ENABLE_AUDIO_OUTPUT=${ENABLE_AUDIO_OUTPUT:-0}

# ============================================================================
# LoRA 相关（仅 TUNER_TYPE=lora 时生效）
# ============================================================================
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 'all-linear' 会覆盖 thinker 内所有 linear 层（包括 MoE experts）；配合
# freeze_vit / freeze_aligner 即可避免训练到 ViT / AudioTower / Projector。
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

# 多模态层冻结：默认全部冻结，仅训 LLM/MoE 部分（与官方示例一致）。
FREEZE_VIT=${FREEZE_VIT:-true}
FREEZE_ALIGNER=${FREEZE_ALIGNER:-true}
FREEZE_LLM=${FREEZE_LLM:-false}

# ============================================================================
# DeepSpeed / 数据并行
# ============================================================================
# 30B-A3B 单卡（即使 80GiB）也很难放下完整模型 + 优化器状态，强烈建议 zero3。
# 设为空字符串可关闭 DeepSpeed（仅用于多卡 80GiB+ 的实验场景）。
DEEPSPEED=${DEEPSPEED:-zero3}

# padding_free 在 zero3 + flash_attn 下吞吐最高（官方 zero3.sh 用法）。
# 若 DEEPSPEED 关闭，建议改用 packing=true。
PADDING_FREE=${PADDING_FREE:-true}
PACKING=${PACKING:-false}

# DataLoader / 显存优化
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
# 让 PyTorch 在分配器里启用 expandable_segments，缓解 MoE/长序列下的碎片
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# ============================================================================
# 数据集（可通过环境变量覆盖）
# ============================================================================
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"$DATA_DIR/val.jsonl"}

# 如果开启了 VAL_MAX_SAMPLES 且原 VAL_JSONL 行数超过限制，则派生一个小 val 集供训练期间使用。
if [ "$VAL_MAX_SAMPLES" -gt 0 ] && [ -f "$VAL_JSONL" ]; then
    VAL_FULL_LINES=$(wc -l < "$VAL_JSONL")
    if [ "$VAL_FULL_LINES" -gt "$VAL_MAX_SAMPLES" ]; then
        VAL_SMALL_JSONL="$DATA_DIR/val.small.jsonl"
        echo "[INFO] val 集 $VAL_FULL_LINES 条 -> 派生 $VAL_MAX_SAMPLES 条小集 (shuffle=$VAL_SAMPLE_SHUFFLE): $VAL_SMALL_JSONL"
        if [ "$VAL_SAMPLE_SHUFFLE" = "1" ]; then
            shuf --random-source=<(yes 42) "$VAL_JSONL" | head -n "$VAL_MAX_SAMPLES" > "$VAL_SMALL_JSONL"
        else
            head -n "$VAL_MAX_SAMPLES" "$VAL_JSONL" > "$VAL_SMALL_JSONL"
        fi
        VAL_JSONL="$VAL_SMALL_JSONL"
    fi
fi

# ============================================================================
# 设备
# ============================================================================
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
echo "[INFO] TRAIN_JSONL = $TRAIN_JSONL"
echo "[INFO] VAL_JSONL   = $VAL_JSONL"
echo "[INFO] TUNER_TYPE  = $TUNER_TYPE (LR=$LEARNING_RATE)"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] EVAL_STEPS=$EVAL_STEPS SAVE_STEPS=$SAVE_STEPS VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES"
echo "[INFO] DEEPSPEED=$DEEPSPEED PADDING_FREE=$PADDING_FREE PACKING=$PACKING"
echo "[INFO] FREEZE_VIT=$FREEZE_VIT FREEZE_ALIGNER=$FREEZE_ALIGNER FREEZE_LLM=$FREEZE_LLM"
echo "[INFO] IMAGE_MAX_TOKEN_NUM=$IMAGE_MAX_TOKEN_NUM VIDEO_MAX_TOKEN_NUM=$VIDEO_MAX_TOKEN_NUM FPS_MAX_FRAMES=$FPS_MAX_FRAMES"
echo "[INFO] ENABLE_AUDIO_OUTPUT=$ENABLE_AUDIO_OUTPUT"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# 友情提示：如果其它进程占用了同一张 GPU，请先释放再启动训练
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况（仅供参考，若其它进程占用过多请释放后再训）："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true
fi

# ============================================================================
# 选择 swift 入口
# ============================================================================
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-/data/miniconda3/envs/env-3.12.11/bin/swift}"
if command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
    PY_BIN=$(command -v python)
elif [ -x "$CONDA_SWIFT_BIN" ]; then
    echo "[INFO] 当前 shell 未找到 swift CLI，自动使用: $CONDA_SWIFT_BIN"
    export PATH="$(dirname "$CONDA_SWIFT_BIN"):$PATH"
    SWIFT_CMD=("$CONDA_SWIFT_BIN")
    PY_BIN="$(dirname "$CONDA_SWIFT_BIN")/python"
else
    echo "[INFO] 未找到 swift CLI，回退到: python -m swift.cli.main"
    SWIFT_CMD=(python -m swift.cli.main)
    PY_BIN=$(command -v python)
fi

# ============================================================================
# 依赖前置检查：Qwen3-Omni 必装包，避免起完多 rank 才发现缺包
#   - qwen_omni_utils >= 0.0.9：模型加载强制依赖（swift/model/models/qwen.py）
#   - soundfile / decord     ：音视频样本预处理依赖
# 设 SKIP_DEP_CHECK=1 可跳过该检查
# ============================================================================
if [ "${SKIP_DEP_CHECK:-0}" != "1" ]; then
    "$PY_BIN" - <<'PY' || {
import importlib, sys
missing = []
for pkg in ("qwen_omni_utils", "soundfile", "decord"):
    try:
        importlib.import_module(pkg)
    except Exception as e:
        missing.append((pkg, repr(e)))
if missing:
    sys.stderr.write("[ERROR] Qwen3-Omni 缺失以下依赖，请先安装后再启动训练：\n")
    for pkg, err in missing:
        sys.stderr.write(f"  - {pkg}: {err}\n")
    sys.stderr.write('\n建议：\n  pip install -U "qwen_omni_utils>=0.0.9" soundfile decord\n')
    sys.exit(2)
PY
        echo "[FATAL] 依赖检查未通过，已中断启动。"
        exit 2
    }
    echo "[INFO] Qwen3-Omni 依赖检查通过 (qwen_omni_utils / soundfile / decord)"
fi

# ============================================================================
# 组装可选参数
# ============================================================================
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")
if [ "$TUNER_TYPE" = "lora" ]; then
    TUNER_ARGS+=(
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --target_modules "$LORA_TARGET_MODULES"
    )
fi

EXTRA_ARGS=()
if [ -n "$DEEPSPEED" ]; then
    EXTRA_ARGS+=(--deepspeed "$DEEPSPEED")
fi
if [ "$PADDING_FREE" = "true" ]; then
    EXTRA_ARGS+=(--padding_free true)
fi
if [ "$PACKING" = "true" ]; then
    EXTRA_ARGS+=(--packing true)
fi
EXTRA_ARGS+=(--add_version "$ADD_VERSION")

# ============================================================================
# 启动训练
# ============================================================================
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" sft \
    --model "$MODEL_PATH" \
    --model_type qwen3_omni_moe \
    --template qwen3_omni \
    "${TUNER_ARGS[@]}" \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALIGNER" \
    --freeze_llm "$FREEZE_LLM" \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$VAL_JSONL" \
    --load_from_cache_file true \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --max_length $MAX_LENGTH \
    --truncation_strategy $TRUNCATION_STRATEGY \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    --save_total_limit ${save_total_limit} \
    --dataset_num_proc $DATASET_NUM_PROC \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --save_only_model true \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[INFO] 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
