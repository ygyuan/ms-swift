#!/usr/bin/env bash
# StepAudio2-mini 全参微调脚本（基于 MS-SWIFT, SFT full）
# 与 run_train_swift.sh (LoRA) 的主要差异：
#   1. TUNER_TYPE 强制为 full，不再注入 lora_rank / lora_alpha / target_modules 等参数；
#   2. 默认学习率下调为 5e-6（v17 用 1e-5 出现类别塌陷，见下）；
#   3. 默认开启 DeepSpeed ZeRO-3 (--deepspeed zero3) 以缓解全参微调的显存压力；
#      若单卡显存富余/不希望走 deepspeed，可设 USE_DEEPSPEED=0 关闭；
#   4. save_total_limit 默认下调到 3：full ckpt 体积≈整模型大小，磁盘占用敏感；
#   5. NUM_EPOCHS 默认 2：full 比 LoRA 更易过拟合，配合 load_best_model_at_end 选最佳点；
#   6. 默认开启 warmup_ratio=0.03，避免前几个 step 大梯度震荡。
#   7. [新增] FREEZE_EMBED_LMHEAD=true 默认冻结 embedding 与 lm_head，防止训练破坏
#            词表分布导致的"类别塌陷"（自回归生成只输出 noise/speech，即使 token_acc
#            仍有 0.98+）。可通过 EXTRA_FREEZE_PREFIXES 追加更多前缀。
#   8. [新增] BALANCE_TRAIN=1 默认在训练前派生一个类均衡后的 train.balanced.jsonl，
#            对多数类降采样、少数类上采样，缓解原始 speech 69% 主导的问题。

set -ex

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 模型 / 输出路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output"}

# 训练超参（可通过环境变量覆盖）
# 本脚本固定 TUNER_TYPE=full（全参微调），不再支持 lora 切换。
TUNER_TYPE=full
# Full 微调推荐的稳健 LR：默认 1e-6（比原来的 1e-5 更保守）。
# 经验教训：v17 训练用 lr=1e-5 + 全参 + 未冻结 embedding/lm_head，导致模型在
# 400~800 步就出现"类别塌陷"（自回归生成只输出 noise/speech，虽然 token_acc
# 仍然维持 0.98+）。全参微调时 lr 过大 + LM head 一起训 + 类别不均衡是塌陷的
# 三大主因，故本脚本默认更保守的 1e-6，并配合下方 FREEZE_EMBED_LMHEAD=true。
# 如仍观察到塌陷可继续下调到 2e-7 / 1e-7。
LEARNING_RATE=${LEARNING_RATE:-1e-6}

# ---- 冻结策略（针对分类任务的类别塌陷问题） ----
# 全参微调时，即使 loss_scale 只对 assistant 段计 loss，只更新 1~2 个分类词
# (speech/music/noise/porn/song) 的梯度也会通过 tie/untie 的 embedding 与
# lm_head 反向传播扰动整个词表分布，进而让 argmax 塌陷到多数类 (noise/speech)。
# 建议至少冻结 lm_head 与 embed_tokens，让 LLM 中间层去学"语音特征 -> 类别"
# 这层映射，避免破坏预训练词表的先验分布。
#
# 也可通过 EXTRA_FREEZE_PREFIXES 追加更多前缀，例如
#   EXTRA_FREEZE_PREFIXES="model.layers.0 model.layers.1 model.layers.2 model.layers.3"
# 冻结前 4 层 transformer，进一步减轻塌陷。
FREEZE_EMBED_LMHEAD=${FREEZE_EMBED_LMHEAD:-true}
EXTRA_FREEZE_PREFIXES=${EXTRA_FREEZE_PREFIXES:-""}

# ---- 类别均衡采样（针对训练数据严重不均衡） ----
# 观察到的原始 train.jsonl 分布 (56302 条):
#   speech 69.4% / noise 16.4% / music 5.7% / porn 4.8% / song 3.7%
# 多数类主导会让全参微调直接学到"永远输出 speech / noise"的捷径。
#
# 打开 BALANCE_TRAIN=1 时会在训练前派生一个类均衡后的 train.balanced.jsonl:
#   - 对多数类 down-sample 到 BALANCE_MAJORITY_CAP 条上限 (默认 8000)
#   - 对少数类 up-sample 到不少于 BALANCE_MINORITY_MIN 条 (默认 4000, 通过重复实现)
# 关闭时 (BALANCE_TRAIN=0) 直接使用 TRAIN_JSONL 原样训练。
BALANCE_TRAIN=${BALANCE_TRAIN:-1}
BALANCE_MAJORITY_CAP=${BALANCE_MAJORITY_CAP:-8000}
BALANCE_MINORITY_MIN=${BALANCE_MINORITY_MIN:-4000}
BALANCE_SEED=${BALANCE_SEED:-42}
# 默认 epoch 设为 2：
# - 全参微调下学习能力强，2 epoch 通常已足够拟合；
# - 配合 load_best_model_at_end + 频繁 eval 在收敛点选 best ckpt；
# - 如观察到欠拟合，可增大到 3~4。
NUM_EPOCHS=${NUM_EPOCHS:-2}
# OOM 优化：默认 batch=1, grad_accum=16（等效 batch=16/GPU）。
# 全参微调相比 LoRA 多了所有权重的优化器状态(Adam: 2x param, fp32 master copy: 1x param)，
# 显存峰值显著更高，故 grad_accum 默认翻倍以维持 effective batch。
BATCH_SIZE=${BATCH_SIZE:-1}
# Eval 阶段没有反向梯度/优化器状态/激活缓存，显存开销远小于训练，可适当放大。
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-16}
# Warmup：避免前几个 step 大梯度震荡，对 full 微调尤其重要。
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
# eval/save 间隔
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-200}
# Full 微调的 ckpt 体积≈整模型，磁盘占用敏感，默认仅保留 50 个。
save_total_limit=${save_total_limit:-50}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# Val 子采样（与 LoRA 脚本一致）：训练期间用一个抽样后的小 val 集监控趋势。
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1000}
VAL_SAMPLE_SHUFFLE=${VAL_SAMPLE_SHUFFLE:-1}
# 序列长度：4096 -> 2048（attention softmax buffer 随 L^2 增长，详见 LoRA 脚本注释）。
MAX_LENGTH=${MAX_LENGTH:-2048}
AUDIO_TOKENS_PER_SEC=${AUDIO_TOKENS_PER_SEC:-13}
PROMPT_RESERVED_TOKENS=${PROMPT_RESERVED_TOKENS:-256}
# 截断策略必须用 'delete'，原因详见 LoRA 脚本注释（音频 placeholder 不会被同步缩减）。
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# DeepSpeed ZeRO 配置：全参微调强烈建议开启 ZeRO-3 切分优化器+梯度+参数，
# 否则单卡 80GB 也很难放下完整 7B 量级模型 + Adam 状态。
# - USE_DEEPSPEED=1（默认）：使用 swift 内置的 zero3 配置；
# - USE_DEEPSPEED=2：使用 zero2（显存换通信，参数不切分）；
# - USE_DEEPSPEED=0：完全关闭 deepspeed（仅适合单卡且显存极充裕的场景）。
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
case "$USE_DEEPSPEED" in
    1) DEEPSPEED_STAGE=zero3 ;;
    2) DEEPSPEED_STAGE=zero2 ;;
    0) DEEPSPEED_STAGE="" ;;
    *) DEEPSPEED_STAGE="$USE_DEEPSPEED" ;;  # 允许直接传 zero3-offload / 自定义 json 路径
esac

# 评估与最佳检查点选择（与 LoRA 脚本一致）
PREDICT_WITH_GENERATE=${PREDICT_WITH_GENERATE:-true}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-rouge-l}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-true}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-true}
EVAL_GEN_MAX_NEW_TOKENS=${EVAL_GEN_MAX_NEW_TOKENS:-8}

# save_only_model：仅保存模型权重，不保存 optimizer / scheduler / RNG 状态。
# - 默认 true：节省磁盘（full ckpt 本就庞大，再加 optimizer 状态尤其在 ZeRO 切分下
#   会进一步膨胀），适合"训完即用"的场景。
# - 但 transformers Trainer 有一条硬约束：DeepSpeed 启用时，
#   `save_only_model=true` 与 `load_best_model_at_end=true` 不能共存
#   （因为加载 best ckpt 需要恢复 DS 优化器分片状态，而这些状态没被保存）。
#   对应报错: "DeepSpeed can't be used with `save_only_model` along with `load_best_model_at_end`."
# 因此当 USE_DEEPSPEED!=0 且 LOAD_BEST_MODEL_AT_END=true 时，下面会自动把 SAVE_ONLY_MODEL
# 强制改为 false，以保证 best ckpt 能被加载回来。
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}
if [ -n "$DEEPSPEED_STAGE" ] \
    && { [ "$LOAD_BEST_MODEL_AT_END" = "true" ] || [ "$LOAD_BEST_MODEL_AT_END" = "True" ]; } \
    && { [ "$SAVE_ONLY_MODEL" = "true" ] || [ "$SAVE_ONLY_MODEL" = "True" ]; }; then
    echo "[WARN] 检测到 DeepSpeed=$DEEPSPEED_STAGE + LOAD_BEST_MODEL_AT_END=true + SAVE_ONLY_MODEL=true 的不兼容组合，"
    echo "       自动将 SAVE_ONLY_MODEL 置为 false (transformers Trainer 硬约束)。"
    echo "       如需保留 SAVE_ONLY_MODEL=true，请设置 LOAD_BEST_MODEL_AT_END=false 或 USE_DEEPSPEED=0。"
    SAVE_ONLY_MODEL=false
fi

# DataLoader / 显存优化
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# 数据集（可通过环境变量覆盖）
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"$DATA_DIR/val.jsonl"}

# ---- 派生类均衡后的 train.balanced.jsonl ----
if [ "$BALANCE_TRAIN" = "1" ] && [ -f "$TRAIN_JSONL" ]; then
    TRAIN_BALANCED_JSONL="$DATA_DIR/train.balanced.jsonl"
    echo "[INFO] 生成类均衡训练集 -> $TRAIN_BALANCED_JSONL (majority_cap=$BALANCE_MAJORITY_CAP minority_min=$BALANCE_MINORITY_MIN seed=$BALANCE_SEED)"
    TRAIN_JSONL_IN="$TRAIN_JSONL" \
    TRAIN_BALANCED_JSONL="$TRAIN_BALANCED_JSONL" \
    BALANCE_MAJORITY_CAP="$BALANCE_MAJORITY_CAP" \
    BALANCE_MINORITY_MIN="$BALANCE_MINORITY_MIN" \
    BALANCE_SEED="$BALANCE_SEED" \
    /data/miniconda3/envs/env-3.12.11/bin/python - <<'PYEOF'
import json, os, random
from collections import defaultdict

SRC = os.environ['TRAIN_JSONL_IN']
DST = os.environ['TRAIN_BALANCED_JSONL']
CAP = int(os.environ['BALANCE_MAJORITY_CAP'])
MIN_N = int(os.environ['BALANCE_MINORITY_MIN'])
SEED = int(os.environ['BALANCE_SEED'])

random.seed(SEED)
buckets = defaultdict(list)
with open(SRC, 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        ans = ''
        for m in d.get('messages', []):
            if m.get('role') == 'assistant':
                ans = str(m.get('content', '')).strip().lower()
                break
        buckets[ans].append(line)

print('[BALANCE] source distribution:')
for k, v in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f'    {k or "<empty>"}: {len(v)}')

out_lines = []
for cls, items in buckets.items():
    if not cls:
        # skip samples without a valid label
        continue
    n = len(items)
    if n >= CAP:
        random.shuffle(items)
        picked = items[:CAP]
    elif n >= MIN_N:
        picked = list(items)
    else:
        reps = MIN_N // n
        rem = MIN_N - reps * n
        pool = items * reps
        random.shuffle(items)
        pool.extend(items[:rem])
        picked = pool
    out_lines.extend(picked)

random.shuffle(out_lines)
with open(DST, 'w') as f:
    for l in out_lines:
        f.write(l + '\n')

final = defaultdict(int)
for l in out_lines:
    d = json.loads(l)
    for m in d.get('messages', []):
        if m.get('role') == 'assistant':
            final[str(m.get('content', '')).strip().lower()] += 1
            break
print(f'[BALANCE] balanced total = {len(out_lines)}, distribution:')
for k, v in sorted(final.items(), key=lambda x: -x[1]):
    print(f'    {k}: {v}')
PYEOF
    TRAIN_JSONL="$TRAIN_BALANCED_JSONL"
fi

# 派生小 val 集（逻辑与 LoRA 脚本完全一致：按 max_length 过滤超长样本 + 抽样）
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
echo "[INFO] TUNER_TYPE  = $TUNER_TYPE (LR=$LEARNING_RATE, WARMUP_RATIO=$WARMUP_RATIO)"
echo "[INFO] FREEZE_EMBED_LMHEAD = $FREEZE_EMBED_LMHEAD  EXTRA_FREEZE_PREFIXES = ${EXTRA_FREEZE_PREFIXES:-<none>}"
echo "[INFO] BALANCE_TRAIN = $BALANCE_TRAIN (majority_cap=$BALANCE_MAJORITY_CAP minority_min=$BALANCE_MINORITY_MIN)"
echo "[INFO] DEEPSPEED   = ${DEEPSPEED_STAGE:-<disabled>}"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] EVAL_STEPS=$EVAL_STEPS SAVE_STEPS=$SAVE_STEPS VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES save_total_limit=$save_total_limit"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# 友情提示：如果其它进程占用了同一张 GPU（常见于共享机器），请先释放再启动训练
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况（仅供参考，若其它进程占用过多请释放后再训）："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true

    # OOM 前置检查：默认 1024 MiB。
    # 注：在共享 GPU / 容器环境下，宿主机的 nvidia driver 常会残留几百 MiB 的"伪占用"
    # （typical: nvidia-smi 中 process_name 显示 [Not Found] 的句柄，ps 查不到对应进程，
    # 实际并不会与训练抢显存），故阈值不宜过低。真正在跑训练的任务通常占用 >= 数 GB，
    # 1024 MiB 足以区分。如仍有误判，可设 GPU_PREALLOC_GUARD_MB=4096 或
    # GPU_PREALLOC_SKIP=1 跳过。
    GPU_PREALLOC_GUARD_MB=${GPU_PREALLOC_GUARD_MB:-1024}
    GPU_PREALLOC_SKIP=${GPU_PREALLOC_SKIP:-0}
    if [ "$GPU_PREALLOC_SKIP" != "1" ]; then
        IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        for _gid in "${_GPU_IDS[@]}"; do
            _used=$(nvidia-smi --id="$_gid" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
            if [ "$_used" -gt "$GPU_PREALLOC_GUARD_MB" ]; then
                echo "[FATAL] GPU $_gid 已被其它进程占用 ${_used} MiB (> ${GPU_PREALLOC_GUARD_MB} MiB)，"
                echo "        全参微调对显存极度敏感，继续训练大概率会 OOM。"
                echo "        请先释放 GPU 或设置 GPU_PREALLOC_SKIP=1 跳过此检查。"
                exit 11
            fi
        done
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡)"
    fi
fi

# 选择 swift 入口（与 LoRA 脚本一致）
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-/data/miniconda3/envs/env-3.12.11/bin/swift}"
if command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
elif [ -x "$CONDA_SWIFT_BIN" ]; then
    echo "[INFO] 当前 shell 未找到 swift CLI，自动使用: $CONDA_SWIFT_BIN"
    export PATH="$(dirname "$CONDA_SWIFT_BIN"):$PATH"
    SWIFT_CMD=("$CONDA_SWIFT_BIN")
else
    echo "[INFO] 未找到 swift CLI，回退到: python -m swift.cli.main"
    SWIFT_CMD=(python -m swift.cli.main)
fi

# 组装 tuner 相关参数（full 模式下仅指定 tuner_type，无 LoRA 子参数）
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")

# ---- 组装冻结参数 ----
# swift 支持 --freeze_parameters <前缀1> <前缀2> ... 冻结所有以这些前缀开头的参数。
# 分类任务下强烈建议冻结 embedding 与 lm_head，避免训练破坏词表分布。
FREEZE_LIST=()
if [ "$FREEZE_EMBED_LMHEAD" = "true" ] || [ "$FREEZE_EMBED_LMHEAD" = "True" ]; then
    # StepAudio2 沿用 Qwen2 结构，embedding 名为 model.embed_tokens，输出层为 lm_head
    FREEZE_LIST+=(model.embed_tokens lm_head)
fi
if [ -n "$EXTRA_FREEZE_PREFIXES" ]; then
    for p in $EXTRA_FREEZE_PREFIXES; do
        FREEZE_LIST+=("$p")
    done
fi
FREEZE_ARGS=()
if [ ${#FREEZE_LIST[@]} -gt 0 ]; then
    FREEZE_ARGS+=(--freeze_parameters "${FREEZE_LIST[@]}")
    echo "[INFO] FREEZE_PARAMETERS = ${FREEZE_LIST[*]}"
else
    echo "[INFO] FREEZE_PARAMETERS = <none> (未冻结 embedding/lm_head，全参训练)"
fi

# DeepSpeed 参数
DS_ARGS=()
if [ -n "$DEEPSPEED_STAGE" ]; then
    DS_ARGS+=(--deepspeed "$DEEPSPEED_STAGE")
fi

# 评估 / best ckpt 的额外参数
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
echo "[INFO] LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END SAVE_ONLY_MODEL=$SAVE_ONLY_MODEL"

# 启动训练
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" sft \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    "${TUNER_ARGS[@]}" \
    "${FREEZE_ARGS[@]}" \
    "${DS_ARGS[@]}" \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$VAL_JSONL" \
    --attn_impl eager \
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
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --save_only_model $SAVE_ONLY_MODEL \
    "${EVAL_ARGS[@]}" \
    "$@"

echo "[INFO] 全参微调完成，Checkpoint 保存在: $OUTPUT_DIR"
