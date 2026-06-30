#!/usr/bin/env bash
# StepAudio2-mini OPSD（On-Policy Self-Distillation）后训练脚本（基于 MS-SWIFT）
#
# === 这是什么 ===
# OPSD = On-Policy Self-Distillation：
#   "用同一个模型的 '带答案版本' 教 '不带答案版本'。"
#   - 学生 (student)：仅看到原始 prompt（音频 + 分类问题），照常推理；
#   - 教师 (teacher)：看到原始 prompt + 参考答案的 hint（特权信息），
#                     从而对正确标签 token 给出更尖锐 / 更可靠的概率分布；
#   - 训练目标：用 JSD/KL 让学生分布对齐教师分布（GKD trainer），
#               教师与学生共享同一份权重，故称 *self*-distillation。
#
# 论文:  https://arxiv.org/abs/2601.18734
# 文档:  docs/source/Instruction/GKD.md  (OPSD 章节)
# 官方示例: examples/train/rlhf/opsd/{opsd.sh,opsd_plugin.py}
#
# === 为什么选 OPSD（相对 SFT / GRPO）===
#   - SFT 学的是 *one-hot* 标签，分类任务下信息量极低，容易过拟合到训练集高频先验；
#   - GRPO 需要可验证 reward + 大量 on-policy sampling，stepaudio 单标签输出
#     reward 几乎只有 0/1，方差小、信号稀疏；
#   - OPSD：教师在「已知答案」的条件下产出 *软分布*（含次优类别的相对概率），
#     学生学到的是 \"label + 类间相似度\" 的更丰富信号，对小类目分类尤其有效。
#
# === ms-swift 入口 ===
#   swift rlhf --rlhf_type gkd
#   通过 OPSD 模式触发：每条样本带一个额外字段 \`teacher_prompt\`，
#   GKD trainer 内部会把 user 的最后一轮 content 替换为该 teacher_prompt
#   作为教师视角的输入，其它字段（音频路径 audios / messages 结构）原样透传。
#
# === 与 grpo / sft 脚本共享的 stepaudio 踩坑 ===
#   - attn_impl eager（step_audio2_mini 当前唯一支持值，不能用 flash_attn）
#   - torch_dtype bfloat16
#   - truncation_strategy delete（音频 placeholder 不会被同步缩减，必须整条丢弃）
#   - PYTORCH_CUDA_ALLOC_CONF expandable_segments
#   - 训练前 GPU 占用前置检查
#   - swift CLI 入口探测（PATH / conda env-3.12.11 / python -m swift.cli.main）
#
# === stepaudio 特定约束 ===
#   1. 教师的 user content **必须保留 <audio> 占位符**，否则 mel 帧数与
#      input_ids 中 <audio_patch> 数量不一致，模型 forward 崩溃。
#      派生 jsonl 时我们用「在原 user content 末尾追加 hint」的方式构造
#      teacher_prompt，自然保留 <audio>。
#   2. **不**启用 vLLM：vLLM 暂不支持 step_audio2_mini 的音频多模态输入。
#      OPSD on-policy 采样将走 transformers native generate（速度比 GRPO 慢些，
#      但 stepaudio 输出极短，可接受）。
#   3. 由于教师需要看到 ground-truth 标签，**对没有 label 字段的样本无法做 OPSD**。
#      派生过程会跳过这些样本（实际数据每行都有 label，正常情况下不会跳）。

set -ex

export LOG_LEVEL=INFO
# 与 grpo 脚本一致：禁用 wandb（避免缺包/网络导致 import 报错）
export WANDB_DISABLED=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# OPSD 既可"冷启动"（直接拿原始 mini 同时当教师+学生），也可在 SFT 后做精炼。
# 实测建议：先 SFT 让模型对任务格式熟悉，再用 OPSD 做软标签精修，效果最稳。
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/stepaudio/opsd"}

# ---------------- OPSD 教师模式 ----------------
# OPSD_TEACHER_MODE:
#   dynamic (默认): 不传 --teacher_model；教师权重 == 学生当前权重，
#                   随训练步动态更新，无需额外加载模型，显存最省；
#                   论文/官方示例均推荐此模式。
#   fixed         : 传 --teacher_model = $MODEL_PATH；教师权重固定为
#                   训练起点的初始模型，与 LoRA 学生天然分离（disable_adapter）。
#                   适合学生想"学到 base 行为"而非"和自己赛跑"。
OPSD_TEACHER_MODE=${OPSD_TEACHER_MODE:-dynamic}

# ---------------- Tuner ----------------
# OPSD 显存消耗 ≈ 学生 forward + 教师 forward + 学生 backward。
# dynamic 自蒸馏模式下：LoRA 训练时教师走 disable_adapter() 共享 base 权重，
# 不会真正多加载一份模型；这是 OPSD 与普通 GKD 最大的实用差异，强烈建议 LoRA。
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    # full + OPSD：dynamic 模式下教师=学生当前权重，教师 forward 不需要额外参数副本，
    # 但需要禁用 dropout/grad；fixed 模式下需要再加载一份完整 base 模型，显存翻倍，
    # 强烈建议配合 ZeRO-3。LR 也得给得很小，否则 KL 容易直接发散。
    DEFAULT_LR=2e-6
else
    # 论文 OPSD 推荐 LoRA + lr=2e-5；stepaudio 输出空间小，调小到 1e-5 更稳。
    DEFAULT_LR=1e-5
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（仅 TUNER_TYPE=lora 生效）
# 注意：OPSD 论文用 r=64 / alpha=128（让 LoRA 容量足够承接软分布信号），
# 但 stepaudio 任务窄，沿用 SFT 脚本同款的 r=8 已足够；如训练曲线偏欠拟合，
# 可调到 r=32 / alpha=64。
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 与 SFT/GRPO 一致：仅注意力 4 个 proj，避免破坏多模态 connector。
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- GKD / OPSD 关键超参 ----------------
# beta：JSD 插值系数（详见 GKD.md）
#   0.0 = Forward KL  (mode-covering, 学生覆盖教师全部模态)
#   0.5 = 标准 JSD    (平衡, 论文 OPSD 默认)
#   1.0 = Reverse KL  (mode-seeking, 学生只锁定教师峰值)
# 分类任务标签离散且少，0.5 通常最稳。
BETA=${BETA:-0.5}
# lmbda：on-policy 触发概率
#   1.0 = 每个 step 都用学生当前权重 generate 一段 completion 做训练（纯 on-policy，
#         即 \"On-Policy Distillation\" 设定，论文/Thinking Machines blog 推荐）
#   0.0 = 不做 on-policy 采样，直接用数据集里的 assistant 答案作为目标序列
#         （等价于 offline distillation，速度最快）
#   0~1 = 混合
# stepaudio 输出很短（基本 1 个标签 token），on-policy 采样开销低，
# 默认 1.0 走纯 on-policy；若想图快可设 0.0 跑 offline 自蒸馏。
LMBDA=${LMBDA:-1.0}
# 采样温度（仅 lmbda>0 时影响学生 generate；>1 鼓励多样性，让自蒸馏暴露更多潜在错误）
TEMPERATURE=${TEMPERATURE:-1.2}
# Top-K logits 蒸馏：默认 None = 全词表 KL；stepaudio 词表 ~150K，全词表 KL
# 在 bf16 下显存约 (B*L*V) * 2bytes，L=2048, V=152K 时单步可达数 GB。
# 默认设 32（比官方 64 更激进），stepaudio 单标签输出 32 个候选已远超实际有效类，
# 显存富余可调到 64 / 128 或设为空字符串走全词表 KL。
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-32}
# sft_alpha：在非 student 生成的样本上额外混入一份 SFT loss 比例
#   0.0 = 纯蒸馏；>0 = 蒸馏 + 部分 NLL，能让分类任务收敛更快
SFT_ALPHA=${SFT_ALPHA:-0.0}
# 单条 completion 最长生成多少 token（仅 lmbda>0 时生效）。
# stepaudio 输出基本是 1 个标签词，给 16 token 足够。
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-16}
# 梯度裁剪
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---------------- 训练规模 ----------------
NUM_EPOCHS=${NUM_EPOCHS:-1}
# 注意：OPSD 一步要做：教师 forward + 学生 forward (+ 学生 generate if on-policy)，
# 显存压力比 SFT 大但比 GRPO 小（不需要 G 条 completion）。bs=1, grad_accum=8 是稳妥起点。
# 关于 gradient_checkpointing：
#   step_audio2_mini 走 eager attention（无 flash_attn），attention 中间张量
#   形状 [B, H, L, L] 在 fp32 softmax 下显存开销随 L^2 暴涨；
#   再叠加 OPSD 的「教师 forward + 学生 forward + on-policy generate」，
#   95G 单卡在 MAX_LENGTH=4096 下不开 GC 几乎必 OOM。
#   官方 examples/train/rlhf/opsd/opsd.sh 也是默认开启的。
#   如确认显存富余想加速，可设 GRADIENT_CHECKPOINTING=0 关闭。
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
case "$GRADIENT_CHECKPOINTING" in
    1|true|True|TRUE) GC_FLAG=true ;;
    0|false|False|FALSE) GC_FLAG=false ;;
    *) GC_FLAG=true ;;
esac

# checkpointing
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
save_total_limit=${save_total_limit:-20}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---------------- 序列 / 截断 ----------------
# eager attention 下 attention softmax buffer ≈ B * H * L * L * 4 bytes (fp32)，
# Qwen2 的 stepaudio_mini 配置约 14 头，L=4096 时单层即 ~3.7GB，再叠加 OPSD 教师/学生
# 双 forward + on-policy generate，95G 卡几乎必爆。
# 默认压到 3072；若 95G 多卡 + GC 已开启仍想拉长，可显式 export MAX_LENGTH=4096。
MAX_LENGTH=${MAX_LENGTH:-3072}
# 截断必须 'delete'（音频 placeholder 不会被同步缩减）
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# ---------------- DeepSpeed ----------------
# - lora + dynamic：默认 zero2 即可（教师不消耗额外参数副本）
# - lora + fixed  ：建议 zero2/zero3，会多加载一份 base 教师
# - full          ：强烈建议 zero3
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
# 即使不用 vLLM 也屏蔽相关 warning，避免 transitive import 干扰日志
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"}

# ---------------- 数据集 ----------------
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
# 沿用 grpo 脚本：用 split_dataset_ratio 切 1% 作为 eval（OPSD 评估只看 ce loss/rouge，
# 不需要单独维护 val 文件）
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- 教师 hint 文案 ----------------
# 教师 user 文本 = 学生 user 文本 + "（已知正确答案是 X，请按要求格式输出）"
# 这条 hint 文案就是 OPSD 的"特权信息"，决定了教师概率分布的质量。
# 如要替换为更复杂的链式思考引导，可通过环境变量覆盖。
OPSD_TEACHER_HINT_TEMPLATE=${OPSD_TEACHER_HINT_TEMPLATE:-$'\n\n(Teacher hint, do NOT echo: the ground-truth label for this audio is "{LABEL}". Now output exactly that label following the required answer format.)'}

# ---------------- 派生带 teacher_prompt 的训练 jsonl ----------------
# OPSD 要求每条样本携带 teacher_prompt 字段（详见 swift/rlhf_trainers/gkd_loss.py
# 中的 build_opsd_teacher_data）。原始 train.jsonl 里没有，这一步在内存里
# 扫一遍生成 train.opsd.jsonl。
#
# 派生逻辑：
#   - 取最后一条 user 的 content（含 <audio> 占位符），原样保留；
#   - 在末尾追加 hint，把 ground-truth 标签暴露给"教师视角"；
#   - 学生视角（messages.user.content）保持原文，不变。
TRAIN_OPSD_JSONL=${TRAIN_OPSD_JSONL:-"$DATA_DIR/train.opsd.jsonl"}
OPSD_REBUILD=${OPSD_REBUILD:-auto}  # auto/1/0：auto = 仅当输出不存在或 mtime 早于源时重建

_need_rebuild=0
case "$OPSD_REBUILD" in
    1) _need_rebuild=1 ;;
    0) _need_rebuild=0 ;;
    auto)
        if [ ! -f "$TRAIN_OPSD_JSONL" ]; then
            _need_rebuild=1
        elif [ "$TRAIN_JSONL" -nt "$TRAIN_OPSD_JSONL" ]; then
            _need_rebuild=1
        fi
        ;;
    *) _need_rebuild=1 ;;
esac

if [ "$_need_rebuild" = "1" ]; then
    echo "[INFO] 派生 OPSD 训练集 (含 teacher_prompt): $TRAIN_OPSD_JSONL"
    SRC_JSONL="$TRAIN_JSONL" \
    DST_JSONL="$TRAIN_OPSD_JSONL" \
    HINT_TEMPLATE="$OPSD_TEACHER_HINT_TEMPLATE" \
    /data/miniconda3/envs/env-3.12.11/bin/python - <<'PYEOF'
import json, os, sys

SRC = os.environ['SRC_JSONL']
DST = os.environ['DST_JSONL']
HINT_TEMPLATE = os.environ['HINT_TEMPLATE']  # 含 {LABEL} 占位符

def pick_label(d):
    # 优先级：显式 label > label_str_origin > 末尾 assistant content
    for k in ('label', 'label_str_origin'):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    for m in reversed(d.get('messages', []) or []):
        if m.get('role') == 'assistant' and isinstance(m.get('content'), str) and m['content']:
            return m['content']
    return None

kept, dropped = 0, 0
with open(SRC, 'r') as fi, open(DST, 'w') as fo:
    for ln, line in enumerate(fi, 1):
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception as e:
            sys.stderr.write(f'[WARN] line {ln} json decode failed: {e}\n')
            dropped += 1
            continue

        msgs = d.get('messages') or []
        # 找最后一条 user
        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get('role') == 'user':
                last_user_idx = i
                break
        if last_user_idx is None:
            dropped += 1
            continue

        label = pick_label(d)
        if not label:
            dropped += 1
            continue

        user_content = msgs[last_user_idx].get('content') or ''
        hint = HINT_TEMPLATE.replace('{LABEL}', label)
        # 教师 prompt = 原 user 内容（含 <audio>）+ hint
        # 这里是 OPSD 的核心：不动学生侧 messages，仅多写一个 teacher_prompt 列。
        teacher_prompt = user_content + hint

        d['teacher_prompt'] = teacher_prompt
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        kept += 1

print(f'[OPSD BUILD] kept={kept} dropped={dropped} -> {DST}')
PYEOF
else
    echo "[INFO] 复用已存在的 OPSD 训练集: $TRAIN_OPSD_JSONL"
fi
TRAIN_JSONL_FOR_SWIFT="$TRAIN_OPSD_JSONL"

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
echo "[INFO] TRAIN_JSONL = $TRAIN_JSONL_FOR_SWIFT  (split_ratio=$SPLIT_DATASET_RATIO)"
echo "[INFO] TUNER_TYPE  = $TUNER_TYPE (LR=$LEARNING_RATE, WARMUP_RATIO=$WARMUP_RATIO)"
echo "[INFO] OPSD_TEACHER_MODE = $OPSD_TEACHER_MODE"
echo "[INFO] DEEPSPEED   = ${DEEPSPEED_STAGE:-<disabled>}"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH"
echo "[INFO] gradient_checkpointing=$GC_FLAG"
echo "[INFO] OPSD beta=$BETA lmbda=$LMBDA temperature=$TEMPERATURE max_completion=$MAX_COMPLETION_LENGTH"
echo "[INFO] OPSD gkd_logits_topk=${GKD_LOGITS_TOPK:-<full vocab>} sft_alpha=$SFT_ALPHA"

# ---------------- GPU 占用前置检查（与 SFT/GRPO 脚本一致） ----------------
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
                echo "        OPSD 显存压力较大（教师+学生 forward），强烈建议先释放。"
                echo "        设 GPU_PREALLOC_SKIP=1 可跳过此检查。"
                exit 11
            fi
        done
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡)"
    fi
fi

# ---------------- swift 入口探测（沿用 GRPO 脚本逻辑） ----------------
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

# OPSD 教师模式参数：
#   dynamic：完全不传 --teacher_model
#   fixed  ：传 --teacher_model = MODEL_PATH，触发 swift 的 self-distillation
#            with fixed teacher 路径（LoRA 下走 disable_adapter，无需多份模型）
TEACHER_ARGS=()
if [ "$OPSD_TEACHER_MODE" = "fixed" ]; then
    TEACHER_ARGS+=(--teacher_model "$MODEL_PATH")
fi

# 可选 top-K logits 蒸馏（空字符串则不传，走全词表 KL）
TOPK_ARGS=()
if [ -n "$GKD_LOGITS_TOPK" ]; then
    TOPK_ARGS+=(--gkd_logits_topk "$GKD_LOGITS_TOPK")
fi

# ---------------- 启动 OPSD 训练 ----------------
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" rlhf \
    --rlhf_type gkd \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    "${TEACHER_ARGS[@]}" \
    "${TUNER_ARGS[@]}" \
    "${DS_ARGS[@]}" \
    --use_vllm false \
    --dataset "$TRAIN_JSONL_FOR_SWIFT" \
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
    --beta $BETA \
    --lmbda $LMBDA \
    --temperature $TEMPERATURE \
    --sft_alpha $SFT_ALPHA \
    "${TOPK_ARGS[@]}" \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_checkpointing $GC_FLAG \
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
    --seed 42 \
    "$@"

echo "[INFO] OPSD 后训练完成，Checkpoint 保存在: $OUTPUT_DIR"
