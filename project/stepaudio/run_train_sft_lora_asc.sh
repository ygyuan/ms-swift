#!/usr/bin/env bash
# StepAudio2-mini 训练脚本（基于 MS-SWIFT）
# 使用本仓库自带的小样本数据集 (project/stepaudio/data/train.jsonl) 进行端到端训练

set -ex

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 模型 / 输出路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output"}

# 训练超参（可通过环境变量覆盖）
# tuner_type: lora（默认，推荐先用小数据验证流程）| full（全参微调，需要更多显存）
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-5
else
    DEFAULT_LR=1e-4
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}
# 默认 epoch 从 8 下调到 2：
# - 8 epoch 迭代末期 train loss=0、token_acc=1.0 明显过拟合；
# - 分类任务标签空间小，LoRA 越多圳 epoch 越可能记住训练集倌仅与高频先验；
# - 配合下面的 load_best_model_at_end + early_stopping 可随时在收敛后提前停。
NUM_EPOCHS=${NUM_EPOCHS:-2}
# OOM 优化：默认 batch=1, grad_accum=8（等效 batch=8/GPU），显著降低单步显存峰值；
# 因为 step_audio2_mini 当前只支持 attn_impl=eager，attention 显存随 L^2 增长，必须把序列截短。
BATCH_SIZE=${BATCH_SIZE:-1}
# Eval 阶段没有反向梯度/优化器状态/激活缓存，显存开销远小于训练；
# 把 EVAL_BATCH_SIZE 从 1 提升到 8，是加速 val 最大的杠杆（吞吐近线性）。
# 但 eager attention 下 attn_weights = (B,H,L,L) 仍随 B 线性增长，
# 当 MAX_LENGTH 被显式抬高（如 4096+）时建议同步把 EVAL_BATCH_SIZE 调小到 4 或 2。
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
# eval/save 间隔从 100 拉到 500，eval 总次数 ~1/5，最显著地省掉训练循环中的 eval 总耗时。
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
save_total_limit=${save_total_limit:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# Val 子采样：原 val.jsonl 太大（数千条 + 单卡音频 forward）会导致每次 eval 数分钟。
# 训练期间使用一个抽样后的小 val 集（默认 1000 条）足够监控趋势；最终评估时再换回完整 val。
# 设为 0 表示不抽样，直接使用完整 VAL_JSONL。
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1000}
# 是否打乱后再取前 N 条（默认打乱，以便覆盖各类标签更均匀）。
VAL_SAMPLE_SHUFFLE=${VAL_SAMPLE_SHUFFLE:-1}
# 4096 -> 2048：eager attention 下 attn_weights = (B,H,L,L)（fp32 softmax buffer），
# 显存随 L^2 增长，从 4096 降到 2048 后单层 softmax buffer 约变成原来的 1/4。
# 实测 L=4096 + bs=1 在 80GB H800 上仍会在 softmax 处 OOM（"Tried to allocate 1.25 GiB"），
# 且训练初期可能短暂触发；保留 2048 作为稳态默认值。
# 如确需更长序列，请同时下调 BATCH_SIZE/EVAL_BATCH_SIZE 或开启更激进的 grad-ckpt。
MAX_LENGTH=${MAX_LENGTH:-2048}
# 每秒音频对应的 token 数 (StepAudio2 audio encoder + adapter):
#   mel hop=160 (100 frames/sec) -> conv2(stride 2) + avg_pool(stride 2) -> /4
#   + adapter(stride 2) -> 再 /2，最终 ≈ 12.5 token/sec
# 实际 step 还按 25s 一段切，每段加 <audio_start>/<audio_end> 共 2 token，
# 实测约 ~14 token/sec。这里取保守值 13，留点余量给 system+user prompt。
AUDIO_TOKENS_PER_SEC=${AUDIO_TOKENS_PER_SEC:-13}
# prompt (system+user 文本+特殊 token) 大约固定开销，预留给非音频部分。
PROMPT_RESERVED_TOKENS=${PROMPT_RESERVED_TOKENS:-256}
# 截断策略：必须用 'delete'（丢弃超长样本），不能用 'right/left'。
# 原因：StepAudio2MiniTemplate 没有声明 placeholder_tokens，'right/left' 截断
# 会无差别砍掉 input_ids 末尾的 <audio_patch>，但音频侧的 mels 数量不会同步缩
# 减，导致模型 forward 时 hidden_states 的占位槽数 != 音频特征帧数，报：
#   RuntimeError: The expanded size of the tensor (X) must match the existing size (Y)
# 'delete' 策略让超出 max_length 的样本被整条丢弃（占比通常很小），保证留下的
# 样本中 input_ids 的 <audio_patch> 与 mel 帧严格对齐。
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# LoRA 相关（仅 TUNER_TYPE=lora 时生效）
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 默认将 target_modules 从 'all-linear' 收紧到仅注意力部分。
# 'all-linear' 会覆盖所有 nn.Linear（包括音频连接层、MLP、gate 等），
# 在小词汇量分类任务上极易过拟合并损坏原有生成能力，
# 表现为推理时生成 <tts_end>/<audio_xxxx> 这类“模板泄漏” token。
# 仅拉 q/k/v/o + 输出 proj 足够调动分类决策边界，并保护原有表示。
# 注意：swift / peft 要求 --target_modules 传多个独立参数（以空格分隔），
# 如果写成“q_proj,k_proj,...”则会被解析为单个名为 'q_proj,k_proj,...' 的模块，
# 报错: ValueError: Target modules {'q_proj,k_proj,v_proj,o_proj'} not found。
# 这里默认以空格分隔，下面组装参数时需要 "不加引号"展开，
# 以保证 bash 拆出多个 token 分别作为独立 CLI 参数。
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# 评估与最佳检查点选择（分类任务现在应该看“生成后的字面是否与 GT 一致”）
# - predict_with_generate=true: 评估阶段真实调 model.generate，而不是 teacher forcing
#   默认会产生 rouge-l 指标；单 token 分类下 rouge-l 近似=一致率。
# - metric_for_best_model=rouge-l + greater_is_better=true 以 rouge-l 选最佳 ckpt。
# - load_best_model_at_end=true 训练结束后加载 best ckpt（同时 save_strategy/eval_strategy/save_steps/eval_steps
#   必须一致，hf trainer 要求）。
PREDICT_WITH_GENERATE=${PREDICT_WITH_GENERATE:-true}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-rouge-l}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-true}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-true}
# Early stopping：swift / HF Trainer 没有提供命令行开关，要启用需在代码中注册
# EarlyStoppingCallback。这里退而求其次：
#   - NUM_EPOCHS 下调到 2，限制总迭代次数；
#   - load_best_model_at_end=true 保证末状态是验证集最佳点。
# 如仍需 early stopping，可后续在 swift sft trainer 注入 callback。
# 生成评估时限制新 token 数 (分类只需一两个 token)
EVAL_GEN_MAX_NEW_TOKENS=${EVAL_GEN_MAX_NEW_TOKENS:-8}

# DataLoader / 显存优化
# 64 个 worker 不仅 CPU 开销大，还会放大 pinned memory，对当前 OOM 也是一个潜在压力源
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
# 让 PyTorch 在分配器里启用 expandable_segments + 更激进的回收，缓解碎片
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# 数据集（可通过环境变量覆盖）
# 默认指向本目录下的 data/，由 project/stepaudio/convert_hf_asc_to_jsonl.py
# (--format messages 或 --format both) 生成的 ms-swift sft 格式 jsonl。
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"$DATA_DIR/val.jsonl"}

# 如果开启了 VAL_MAX_SAMPLES 且原 VAL_JSONL 行数超过限制，则派生一个小 val 集供训练期间使用。
# 注意：predict_with_generate=true 路径下，swift 评估会调用
#   infer_engine._batch_encode -> template.encode -> _encode_truncated
# 这条路径对超长样本是 "truncation_strategy='raise'" 行为：只要遇到一条
# length>max_length 的样本就直接抛 MaxLengthError，让整个评估崩溃。
# 所以派生 val.small.jsonl 时必须先按 max_length 过滤掉所有可能超长的样本。
# 我们用 wav header 读时长，估算 audio token 数 ≈ duration * AUDIO_TOKENS_PER_SEC，
# 加上 PROMPT_RESERVED_TOKENS 后判断是否超过 MAX_LENGTH。
if [ "$VAL_MAX_SAMPLES" -gt 0 ] && [ -f "$VAL_JSONL" ]; then
    VAL_FULL_LINES=$(wc -l < "$VAL_JSONL")
    VAL_SMALL_JSONL="$DATA_DIR/val.small.jsonl"
    echo "[INFO] val 集 $VAL_FULL_LINES 条 -> 过滤超长样本(>${MAX_LENGTH} tokens) 并派生 $VAL_MAX_SAMPLES 条小集 (shuffle=$VAL_SAMPLE_SHUFFLE): $VAL_SMALL_JSONL"
    SHUF_FLAG="$VAL_SAMPLE_SHUFFLE" \
    VAL_JSONL="$VAL_JSONL" \
    VAL_SMALL_JSONL="$VAL_SMALL_JSONL" \
    VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
    MAX_LENGTH="$MAX_LENGTH" \
    AUDIO_TOKENS_PER_SEC="$AUDIO_TOKENS_PER_SEC" \
    PROMPT_RESERVED_TOKENS="$PROMPT_RESERVED_TOKENS" \
    /data/miniconda3/envs/env-3.12.11/bin/python - <<'PYEOF'
import json, os, random, wave, contextlib

VAL = os.environ['VAL_JSONL']
OUT = os.environ['VAL_SMALL_JSONL']
MAXN = int(os.environ['VAL_MAX_SAMPLES'])
MAX_LEN = int(os.environ['MAX_LENGTH'])
TPS = float(os.environ['AUDIO_TOKENS_PER_SEC'])
RESERVED = int(os.environ['PROMPT_RESERVED_TOKENS'])
SHUF = os.environ.get('SHUF_FLAG', '1') == '1'
MAX_AUDIO_SEC = max(1.0, (MAX_LEN - RESERVED) / TPS)

def wav_dur(p):
    try:
        with contextlib.closing(wave.open(p, 'rb')) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None

lines = []
with open(VAL, 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        if line:
            lines.append(line)

if SHUF:
    random.seed(42)
    random.shuffle(lines)

kept, dropped_long, dropped_missing = 0, 0, 0
with open(OUT, 'w') as fo:
    for line in lines:
        if kept >= MAXN:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        ap = (d.get('audios') or [None])[0]
        if not ap or not os.path.isfile(ap):
            dropped_missing += 1
            continue
        s = wav_dur(ap)
        if s is None:
            dropped_missing += 1
            continue
        if s > MAX_AUDIO_SEC:
            dropped_long += 1
            continue
        fo.write(line + '\n')
        kept += 1

print(f'[VAL FILTER] kept={kept} dropped_long={dropped_long} dropped_missing={dropped_missing} '
      f'(MAX_AUDIO_SEC={MAX_AUDIO_SEC:.1f}s = (MAX_LEN={MAX_LEN}-RESERVED={RESERVED})/TPS={TPS:.1f})')
PYEOF
    VAL_JSONL="$VAL_SMALL_JSONL"
fi

# 设备
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
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# 友情提示：如果其它进程占用了同一张 GPU（常见于共享机器），请先释放再启动训练
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况（仅供参考，若其它进程占用过多请释放后再训）："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true

    # OOM 前置检查：如果将要使用的卡上已经有 >GPU_PREALLOC_GUARD_MB 的其它进程占用，
    # 强制中止，避免训练运行到 attention softmax 时再触发 OOM（非常浪费时间）。
    # 触发条件可放宽：默认 1024 MiB（即 ≥1GiB 已被他人占用就不让启动）。
    GPU_PREALLOC_GUARD_MB=${GPU_PREALLOC_GUARD_MB:-1024}
    GPU_PREALLOC_SKIP=${GPU_PREALLOC_SKIP:-0}
    if [ "$GPU_PREALLOC_SKIP" != "1" ]; then
        IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        for _gid in "${_GPU_IDS[@]}"; do
            # 取出该卡上所有进程的 used_memory 之和（MiB）
            _used=$(nvidia-smi --id="$_gid" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
            if [ "$_used" -gt "$GPU_PREALLOC_GUARD_MB" ]; then
                echo "[FATAL] GPU $_gid 已被其它进程占用 ${_used} MiB (> ${GPU_PREALLOC_GUARD_MB} MiB)，"
                echo "        如果继续训练大概率会在 attention softmax 时 OOM。"
                echo "        请先释放 GPU 或设置 GPU_PREALLOC_SKIP=1 跳过此检查。"
                exit 11
            fi
        done
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡)"
    fi
fi

# 选择 swift 入口：
#   1) 优先用 PATH 中的 swift 命令
#   2) 其次尝试本机已存在的 conda env env-3.12.11（仓库推荐环境，已预装依赖）
#   3) 最后回退到 python -m swift.cli.main（需当前 Python 已安装 ms-swift）
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-/data/miniconda3/envs/env-3.12.11/bin/swift}"
if command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
elif [ -x "$CONDA_SWIFT_BIN" ]; then
    echo "[INFO] 当前 shell 未找到 swift CLI，自动使用: $CONDA_SWIFT_BIN"
    # 同时把同环境的 python 放到 PATH 最前面，避免子进程使用错误的 python
    export PATH="$(dirname "$CONDA_SWIFT_BIN"):$PATH"
    SWIFT_CMD=("$CONDA_SWIFT_BIN")
else
    echo "[INFO] 未找到 swift CLI，回退到: python -m swift.cli.main"
    SWIFT_CMD=(python -m swift.cli.main)
fi

# 组装 tuner 相关参数
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")
if [ "$TUNER_TYPE" = "lora" ]; then
    TUNER_ARGS+=(
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
    )
    # --target_modules 接受多个独立 token，这里以空格为分隔符将字符串
    # 拆成多个参数迫入数组，同时也兼容“q_proj,k_proj,v_proj,o_proj”
    # 这种逗号分隔的写法（自动转为空格分隔）。
    _tm_normalized=${LORA_TARGET_MODULES//,/ }
    # shellcheck disable=SC2206
    _tm_array=($_tm_normalized)
    TUNER_ARGS+=(--target_modules "${_tm_array[@]}")
fi

# 评估 / best ckpt / early stopping 的额外参数
EVAL_ARGS=()
if [ "$PREDICT_WITH_GENERATE" = "true" ] || [ "$PREDICT_WITH_GENERATE" = "True" ]; then
    EVAL_ARGS+=(
        --predict_with_generate true
        --max_new_tokens "$EVAL_GEN_MAX_NEW_TOKENS"
        --temperature 0.0
        --top_p 1.0
    )
fi
EVAL_ARGS+=(
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
    --greater_is_better "$GREATER_IS_BETTER"
    --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"
)

echo "[INFO] PREDICT_WITH_GENERATE=$PREDICT_WITH_GENERATE METRIC_FOR_BEST_MODEL=$METRIC_FOR_BEST_MODEL"
echo "[INFO] LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END"
echo "[INFO] LORA_TARGET_MODULES=$LORA_TARGET_MODULES (调为 q_proj,k_proj,v_proj,o_proj 以避免过拟合)"

# 启动训练
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" sft \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    "${TUNER_ARGS[@]}" \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$VAL_JSONL" \
    --attn_impl eager \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
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
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --save_only_model true \
    "${EVAL_ARGS[@]}" \
    "$@"

    # --new_special_tokens '<speak_reply>' '<speak_backchannel>' '<tts_start>' '<tts_end>' \
echo "[INFO] 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
