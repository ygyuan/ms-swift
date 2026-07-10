#!/usr/bin/env bash
# StepAudio2-mini 全参微调脚本（MELD 情感 7 分类版, 基于 MS-SWIFT SFT full）
# ═══════════════════════════════════════════════════════════════════════
# 【v2 关键调整 · 2026-07-07 · Word 版数据集】
# ═══════════════════════════════════════════════════════════════════════
# 背景: v0/v1 用 letter 版 (S/A/N/J/D/F/G 单字母 label) 训练所有 checkpoint 都塌陷:
#   - v0 (full)  ckpt-50   → 99.9% 预测 N, acc=48.12% (=N 类占比)
#   - v6 (LoRA)  ckpt-150  → 99.9% 预测 N, acc=48.12%
#   - baseline   base 模型 → 93.5% 预测 J, acc=17.05%
#
# 但用 UltraEval-Audio 的完整单词 prompt (未微调 base 模型)  → acc=55.47% ✓
# 差距 3× 的唯一来源就是 prompt: UltraEval 让模型输出完整单词 (surprise/anger/...),
# 而不是 letter. 我们之前的 letter 版 prompt 在正文里显式写了
#   "S=surprise, A=anger, N=neutral, J=joy, D=sadness, F=fear, G=disgust"
# → base LLM 通过 attention 直接从 prompt 里复制字母, 塌陷到 J/N.
#
# v2 策略: 与 UltraEval-Audio 完全对齐:
#   1. 数据集切换到 project/stepaudio/data_meld/ 下的 word 版:
#        - train.jsonl / dev.jsonl / test.jsonl
#        - assistant label 是完整单词: surprise / anger / neutral / joy / sadness / fear / disgust
#   2. prompt 已经在 build_meld_jsonl.py 里生成为 UltraEval 完全相同的文本:
#        "listen the audio and judge the emotion of the speaker, the answer must be
#         one of [surprise,anger,neutral,joy,sadness,fear,disgust], answer without explain"
#   3. 7 个词的首 token 完全互不相同 (surprise=sur, anger=anger, neutral=neutral,
#      joy=joy, sadness=sad, fear=f, disgust=dis), 所以 first-token argmax 仍可分类;
#      同时这些首 token 都是【完整语义 token】, base LLM 的语言先验对情感 word 敏感,
#      能直接给出接近 GT 的分布 (UltraEval 55.47% 就是证据).
#
# 其它 (来自 v1) 关键设计保留:
#   - LABEL_SMOOTHING=0.0 (避开 swift+step_audio2 的 label_smoother shift bug);
#   - predict_with_generate=false + eval_metric=acc + metric_for_best_model=token_acc
#     (word 版 label 是多 token, token_acc 会对模板+label 所有位置平均, 依然稳定可用);
#   - EARLY_STOP_INTERVAL=5 防过训塌陷;
#   - VAL 派生 (音频时长过滤+抽样) 与 LoRA 脚本一致.
# ═══════════════════════════════════════════════════════════════════════

set -ex

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ─── 模型 / 输出路径 ───────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/meld"}
ADD_VERSION=${ADD_VERSION:-"false"}

# ─── 训练超参 ───────────────────────────────────────────────────
TUNER_TYPE=full

# Full 微调推荐 LR: 1e-6 保守起步.
# 经验教训 (来自另一个 5 分类任务): lr=1e-5 + 全参 + 未冻结 embedding/lm_head
# → 400~800 步就"类别塌陷"(argmax 恒等于多数类, 而 token_acc 仍 0.98+).
# 全参微调时 lr 过大 + LM head 一起训 + 类别不均衡是塌陷的三大主因, 故本脚本默认更保守的 1e-6,
# 并配合 FREEZE_EMBED_LMHEAD=true. 如仍观察到塌陷可继续下调到 5e-7 / 2e-7.
LEARNING_RATE=${LEARNING_RATE:-5e-7}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

# ══════════════════════════════════════════════════════════════════
# 【label_smoothing bug 说明】默认关闭 label smoothing (0.0)
# ══════════════════════════════════════════════════════════════════
# 之前 v4/v5 的 LoRA 训练均出现 train_loss ≈ 51.6 卡住 / token_acc 0.08 远低于随机猜
# 的现象, 排查根因发现:
#   - 我们设了 label_smoothing_factor=0.05, 触发 HF Trainer 的 label_smoother 分支
#   - swift/trainers/seq2seq_trainer.py 里对 label_smoother 的调用:
#         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#             loss = self.label_smoother(outputs, labels, shift_labels=True)
#         else:
#             loss = self.label_smoother(outputs, labels)   # ← shift_labels=False!
#   - 'StepAudio2ForCausalLM' 是自定义架构, 不在 transformers 的映射表里
#     → 走 else 分支 → 计算 loss 时 logits 与 labels 没做 shift 对齐
#   - 结果对唯一非 -100 的 assistant 标签位置, 用的是"该 token 之后应该是什么"的 logits
#     去评估当前 token, -log P 直接飙到 ~12; 再加 label smoothing 那部分 sum(log_probs)/vocab
#     贡献 ~40, 总 loss ~52 与观测完全吻合.
#
# 修复: 关闭 label_smoothing → compute_loss 走 outputs.loss (由模型自身 loss_function
# 用 mean reduction 正确计算), 才能真正开始训练.
# 类别不均衡问题改用其它手段解决 (温和上采样 / 分层 / class-weight / EARLY_STOP).
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}

# 【v3 关键修复 · 对齐 UltraEval-Audio 55.47% baseline】SYSTEM prompt
# ─────────────────────────────────────────────────────────
# 之前 base 推理用同 UltraEval 相同的 wav + prompt 只能拿 48.08%, 而 UltraEval 官方 55.47%.
# 排查 UltraEval-Audio/audio_evals/models/step_audio_2_mini.py:180 发现:
#     if not has_system:
#         messages.append({"role":"system","content":"You are a helpful assistant."})
# 它【强制】注入了 system prompt, 而我们 swift 默认 --system=None 导致 prompt 缺失 <|BOT|>system…
# 前缀, 模型分布完全偏移. 本字段确保训练时也使用相同的 system prompt, 避免
# "训练用无 system, 推理用有 system" 的前后不一致导致二次塌陷.
# 同时 run_inference_meld.sh 也默认传 --system="You are a helpful assistant.", 两边一致.
SYSTEM=${SYSTEM:-"You are a helpful assistant."}

# ─── 冻结策略 ─────────────────────────────────────────────────
# swift full 微调下, 分类任务尤其容易出现"类别塌陷": 即使 loss_scale 只对 assistant
# 段计 loss, 只更新 1 个分类 token (S/A/N/J/D/F/G) 的梯度也会通过 embedding 与
# lm_head 反向传播扰动整个词表分布, 让 argmax 塌陷到多数类.
# 强烈建议冻结 embedding + lm_head, 让 LLM 中间层去学"语音特征 → letter"的映射.
# 注意: MELD letter S/A/N/J/D/F/G 都是普通英文单字母, 存在词表里, 冻结 embedding
# 只是不让"这几个字母的向量"被推来推去而已, 不影响正常学习.
FREEZE_EMBED_LMHEAD=${FREEZE_EMBED_LMHEAD:-true}
# 可追加更多冻结前缀, 例如冻结前 4 层 transformer 进一步减轻塌陷:
#   EXTRA_FREEZE_PREFIXES="model.layers.0 model.layers.1 model.layers.2 model.layers.3"
EXTRA_FREEZE_PREFIXES=${EXTRA_FREEZE_PREFIXES:-""}

# vit (audio encoder) 保持冻结: 32 层 conformer 参数量巨大, 全参训练不稳且显存吃紧
FREEZE_VIT=${FREEZE_VIT:-true}
# aligner (adapter) 是 audio → LLM 的关键桥梁, 全参训练它才能真正把
# "语音内容特征" 变形到 "情感语义" 空间. 只有 linear1+linear2 两层 ~10M 参数, 可控.
FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
# LLM 主体默认不冻结. 若想只训 aligner+少量层, 可以 FREEZE_LLM=true.
FREEZE_LLM=${FREEZE_LLM:-false}

# ─── 类别均衡采样 ────────────────────────────────────────────
# MELD 分布 (train, 9989 条):
#   neutral 47.2% / joy 17.4% / anger 12.1% / surprise 11.9% / sadness 7.2% /
#   disgust 2.7% / fear 2.7%
# 上一版 (v3) 用 5.3x 上采样 fear/disgust 到 balanced, 让 LoRA 学会了"死记那 268 条 fear 音频"的
# 捷径, 反而害了 test acc. 因此 full 版默认关闭 balance, 走原始分布.
# 如需重开, 可 BALANCE_TRAIN=1 bash ..., 并且调整 BALANCE_MAJORITY_CAP/MINORITY_MIN 到
# 适合 MELD 数量级 (例如 majority_cap=3000, minority_min=800).
BALANCE_TRAIN=${BALANCE_TRAIN:-0}
BALANCE_MAJORITY_CAP=${BALANCE_MAJORITY_CAP:-3000}
BALANCE_MINORITY_MIN=${BALANCE_MINORITY_MIN:-800}
BALANCE_SEED=${BALANCE_SEED:-42}

# ─── 迭代次数 / batch ─────────────────────────────────────────
# Full 微调下学习能力比 LoRA 强, 2~3 epoch 通常已够; 配合 load_best_model_at_end 挑最佳点
NUM_EPOCHS=${NUM_EPOCHS:-4}
# 单卡 batch=1 + 多卡 DDP + grad_accum=16 = effective batch 16*n_gpu
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-16}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}

# ─── eval / save 频率 ───────────────────────────────────────
# MELD train 9989 条 * 3 epoch / (bs=1 * accum=16 * n_gpu=8) ≈ 234 步
# 所以 SAVE_STEPS=50 → 全程 4~5 个 ckpt; EVAL_STEPS=50 与 save 同步保证 load_best_model_at_end 生效
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-50}
# full ckpt ≈ 完整模型, 磁盘敏感; 默认保留 5 个避免占用
save_total_limit=${save_total_limit:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ─── val 抽样 (对齐 LoRA 脚本) ────────────────────────────────
# MELD dev 1108 条, 抽样上限 1000 足以覆盖全集 (basically no shrink);
# 保持这个变量是为了如果切到 test 集 (2610 条) 时能限制 eval 时长
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1500}
VAL_SAMPLE_SHUFFLE=${VAL_SAMPLE_SHUFFLE:-0}

# ─── 序列长度 ─────────────────────────────────────────────
MAX_LENGTH=${MAX_LENGTH:-4096}
AUDIO_TOKENS_PER_SEC=${AUDIO_TOKENS_PER_SEC:-13}
PROMPT_RESERVED_TOKENS=${PROMPT_RESERVED_TOKENS:-256}
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# ─── DeepSpeed ZeRO 配置 ─────────────────────────────────
# 全参微调强烈建议 ZeRO-3 (切分 优化器状态+梯度+参数);
# 单卡场景可以关掉 (USE_DEEPSPEED=0), 但需要至少 40GB 显存
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
case "$USE_DEEPSPEED" in
    1) DEEPSPEED_STAGE=zero3 ;;
    2) DEEPSPEED_STAGE=zero2 ;;
    0) DEEPSPEED_STAGE="" ;;
    *) DEEPSPEED_STAGE="$USE_DEEPSPEED" ;;
esac

# ─── 评估策略 (对齐 LoRA 脚本 v4+ 的方式) ─────────────────
# 【关键】历史配置 predict_with_generate=true + metric=rouge-l 在 MELD 上给了误导性信号:
# 只要输出 neutral 就能拿到 rouge-l ≈ 42.5, 500 步内根本没变化, load_best_model_at_end
# 挑的 "best" 毫无区分度. 切成:
#   - predict_with_generate=false → 走 teacher-forcing logits, 无 generate 开销;
#   - eval_metric=acc → swift AccMetrics, 只有 logits.argmax 与 GT token 完全一致才算对,
#     对"预测 neutral 覆盖一切"会立刻显现 token_acc ≈ 0.47 (=neutral 先验) 而不是被 rouge-l 掩盖;
#   - metric_for_best_model=token_acc → 与 eval_metric=acc 产出的 key 对齐.
PREDICT_WITH_GENERATE=${PREDICT_WITH_GENERATE:-false}
EVAL_METRIC=${EVAL_METRIC:-acc}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-token_acc}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-true}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-true}
# 【v2 word 版】label 最长是 disgust=3 tokens (dis+g+ust); generate 时也只需一次前向,
# 但如果开启 PREDICT_WITH_GENERATE=true 得给它足够步数. 默认还是 disable generate,
# 走 teacher-forcing logits, 这个变量只在 PREDICT_WITH_GENERATE=true 时才生效.
EVAL_GEN_MAX_NEW_TOKENS=${EVAL_GEN_MAX_NEW_TOKENS:-4}
# ─── 早停 ──────────────────────────────────────────────────
# 连续 N 次 eval 指标不提升就自动停训, 避免过训带来的塌陷.
# EVAL_STEPS=50, EARLY_STOP=5 → 250 步 (>1 epoch) 无提升就停.
# 设为 0 关闭.
EARLY_STOP_INTERVAL=${EARLY_STOP_INTERVAL:-10}

# ─── save_only_model 兼容处理 ─────────────────────────────
# DeepSpeed 启用时, save_only_model=true 与 load_best_model_at_end=true 冲突
# (加载 best ckpt 需要恢复 DS 分片的 optimizer 状态, 但 save_only_model 没存这些)
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}
if [ -n "$DEEPSPEED_STAGE" ] \
    && { [ "$LOAD_BEST_MODEL_AT_END" = "true" ] || [ "$LOAD_BEST_MODEL_AT_END" = "True" ]; } \
    && { [ "$SAVE_ONLY_MODEL" = "true" ] || [ "$SAVE_ONLY_MODEL" = "True" ]; }; then
    echo "[WARN] DeepSpeed=$DEEPSPEED_STAGE + LOAD_BEST_MODEL_AT_END=true + SAVE_ONLY_MODEL=true 不兼容,"
    echo "       自动将 SAVE_ONLY_MODEL 置为 false (transformers Trainer 硬约束)."
    SAVE_ONLY_MODEL=false
fi

# ─── DataLoader / 显存 ──────────────────────────────────
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# ─── 数据集 (MELD word 版, 与 UltraEval-Audio 55.47% baseline 对齐) ─────────
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.r1omni_self_asr_reversed.jsonl"}
VAL_JSONL=${VAL_JSONL:-"$DATA_DIR/dev.r1omni_self_asr_reversed.jsonl"}


# ─── 设备 ─────────────────────────────────────────────
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        export CUDA_VISIBLE_DEVICES=$(nvidia-smi --list-gpus | awk '{printf "%s,", NR-1}' | sed 's/,$//')
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}

echo "[INFO] ═══════════════════════════════════════════════════════════════"
echo "[INFO] SWIFT_ROOT  = $SWIFT_ROOT"
echo "[INFO] MODEL_PATH  = $MODEL_PATH"
echo "[INFO] OUTPUT_DIR  = $OUTPUT_DIR"
echo "[INFO] TRAIN_JSONL = $TRAIN_JSONL"
echo "[INFO] VAL_JSONL   = $VAL_JSONL"
echo "[INFO] TUNER_TYPE  = $TUNER_TYPE (LR=$LEARNING_RATE, WARMUP=$WARMUP_RATIO, WD=$WEIGHT_DECAY)"
echo "[INFO] LABEL_SMOOTHING = $LABEL_SMOOTHING  (默认 0.0 是为了避开 swift+step_audio2 的 label_smoother bug)"
echo "[INFO] 【v2 · Word 版数据集】data_meld/train.jsonl + dev.jsonl (label=完整单词, 与 UltraEval-Audio 55.47% baseline 对齐)"
echo "[INFO] FREEZE_EMBED_LMHEAD=$FREEZE_EMBED_LMHEAD  EXTRA_FREEZE_PREFIXES=${EXTRA_FREEZE_PREFIXES:-<none>}"
echo "[INFO] FREEZE_VIT=$FREEZE_VIT  FREEZE_ALIGNER=$FREEZE_ALIGNER  FREEZE_LLM=$FREEZE_LLM"
echo "[INFO] BALANCE_TRAIN=$BALANCE_TRAIN  (majority_cap=$BALANCE_MAJORITY_CAP  minority_min=$BALANCE_MINORITY_MIN)"
echo "[INFO] NUM_EPOCHS=$NUM_EPOCHS  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "[INFO] EVAL_STEPS=$EVAL_STEPS  SAVE_STEPS=$SAVE_STEPS  save_total_limit=$save_total_limit"
echo "[INFO] PREDICT_WITH_GENERATE=$PREDICT_WITH_GENERATE  EVAL_METRIC=$EVAL_METRIC  METRIC_FOR_BEST_MODEL=$METRIC_FOR_BEST_MODEL"
echo "[INFO] LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END  EARLY_STOP_INTERVAL=$EARLY_STOP_INTERVAL"
echo "[INFO] DEEPSPEED=${DEEPSPEED_STAGE:-<disabled>}  SAVE_ONLY_MODEL=$SAVE_ONLY_MODEL"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "[INFO] MAX_LENGTH=$MAX_LENGTH  TRUNCATION_STRATEGY=$TRUNCATION_STRATEGY"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[INFO] ═══════════════════════════════════════════════════════════════"

# ─── GPU 前置检查 ───────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况（仅供参考）："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true

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

# ─── swift 入口 ──────────────────────────────────────
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

# ─── 组装 tuner 参数 (full 模式无 LoRA 子参数) ───────
TUNER_ARGS=(--tuner_type "$TUNER_TYPE")

# ─── 组装冻结参数 ─────────────────────────────────
FREEZE_LIST=()
if [ "$FREEZE_EMBED_LMHEAD" = "true" ] || [ "$FREEZE_EMBED_LMHEAD" = "True" ]; then
    # StepAudio2 沿用 Qwen2 结构, embedding=model.embed_tokens, 输出=lm_head
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
# 传入 vit / aligner / llm 的三档冻结开关 (swift 原生支持)
FREEZE_ARGS+=(
    --freeze_vit "$FREEZE_VIT"
    --freeze_aligner "$FREEZE_ALIGNER"
    --freeze_llm "$FREEZE_LLM"
)

# ─── DeepSpeed 参数 ──────────────────────────────
DS_ARGS=()
if [ -n "$DEEPSPEED_STAGE" ]; then
    DS_ARGS+=(--deepspeed "$DEEPSPEED_STAGE")
fi

# ─── 评估参数 ─────────────────────────────────
EVAL_ARGS=()
if [ "$PREDICT_WITH_GENERATE" = "true" ] || [ "$PREDICT_WITH_GENERATE" = "True" ]; then
    EVAL_ARGS+=(
        --predict_with_generate true
        --max_new_tokens "$EVAL_GEN_MAX_NEW_TOKENS"
        --temperature 0.0
        --top_p 1.0
    )
else
    EVAL_ARGS+=(--predict_with_generate false)
fi
EVAL_ARGS+=(--eval_metric "$EVAL_METRIC")
EVAL_ARGS+=(
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
    --greater_is_better "$GREATER_IS_BETTER"
    --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"
)

# ─── 早停参数 ─────────────────────────────────
EARLY_STOP_ARGS=()
if [ "${EARLY_STOP_INTERVAL:-0}" -gt 0 ]; then
    EARLY_STOP_ARGS+=(--early_stop_interval "$EARLY_STOP_INTERVAL")
fi

# ─── 启动训练 ─────────────────────────────────
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" sft \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    --system "$SYSTEM" \
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
    --weight_decay $WEIGHT_DECAY \
    --label_smoothing_factor $LABEL_SMOOTHING \
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
    --add_version $ADD_VERSION \
    --report_to tensorboard \
    --save_total_limit ${save_total_limit} \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --save_only_model $SAVE_ONLY_MODEL \
    "${EARLY_STOP_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "$@"

echo "[INFO] MELD 全参微调完成，Checkpoint 保存在: $OUTPUT_DIR"
