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
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/meld"}
ADD_VERSION=${ADD_VERSION:-false}
SEED=${SEED:-42}

# 训练超参（可通过环境变量覆盖）
# tuner_type: lora（默认，推荐先用小数据验证流程）| full（全参微调，需要更多显存）
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-5
else
    # 【v4 复盘 v3 塌陷问题的修复项】
    # v3 用 lr=1e-4 训练后, checkpoint-200 完全塌陷到 N (99.8% 预测 N),
    # checkpoint-400 学到了对 F 的"复制过拟合", checkpoint-639 分类 acc 反而降到 33.9%.
    # 根本原因: 1e-4 对 LoRA 而言过大, 相当于在 warmup 结束后立刻把 LoRA 权重推到某个
    # 强多数类先验; 而 balanced 上采样的 F 类只有 268 unique 样本, LoRA 表示空间被高 lr
    # 强行"记忆化"了这些少量样本 -> 出现 638 fp 的 F.
    # 5e-5 是 LoRA + audio LM 的稳态经验值, 配合 cosine + 3% warmup 更平滑.
    DEFAULT_LR=5e-5
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# 【v4 新增】WEIGHT_DECAY:
# swift 默认 weight_decay=0.1, 对 LoRA 权重来说太大 (会把 LoRA \Delta W 直接拉回 0,
# 等价于抑制学习). 尤其 LoRA A/B 初始接近 0 时, WD=0.1 让梯度基本白干.
# LoRA fine-tune 经验值 0.0 ~ 0.01, 默认给 0.0.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

# 【v6 关键修复】LABEL_SMOOTHING 默认 0.0（关闭）
# ─────────────────────────────────────────────────────────────
# v4/v5 训练崩溃 (train_loss ≈ 51.6 卡住, token_acc 0.08 比随机猜 1/7 还差) 的
# 真正根因是 label_smoothing_factor=0.05 触发了 HuggingFace Trainer 的 label_smoother
# 分支, 而 swift/trainers/seq2seq_trainer.py 的 compute_loss 里有这样一段:
#
#     if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#         loss = self.label_smoother(outputs, labels, shift_labels=True)
#     else:
#         loss = self.label_smoother(outputs, labels)   # shift_labels=False !!
#
# Step-Audio-2-mini 的架构名 'StepAudio2ForCausalLM' 不在 transformers 的
# MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 里 (它是自定义 modeling_step_audio_2.py 加载),
# 所以走 else 分支, 也就是 shift_labels=False. label_smoother 拿到的 logits 和
# labels 完全错位:
#   - logits[i] 应预测 labels[i+1] (自回归 next-token)
#   - 但 label_smoother 里没 shift, 用 logits[i] 去评估 labels[i]
# 结果对唯一非 -100 的 "N" 位置, 模型输出的其实是"N 之后应该是什么"的分布
# (期望是 <|EOT|>), 与 GT "N" 完全对不上 -> -log P ≈ 12 (nll 部分).
# 再加 label smoothing 那部分 log_probs.sum() 除以 vocab_size 项 ≈ 40,
# 总 loss ≈ 0.95*11 + 0.05*812 ≈ 51.6, 与实际观测完全一致.
#
# 修复: 关闭 label_smoothing (设 0.0), 让 compute_loss 走标准的 outputs.loss 路径
# (由模型 self.loss_function 用 mean reduction 正确计算).
# 类别不均衡问题改用其它手段解决 (温和上采样 / 分层采样 / class-weight).
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}

# 【v3 关键修复 · 对齐 UltraEval-Audio 55.47% baseline】SYSTEM prompt
# ─────────────────────────────────────────────────────────
# UltraEval-Audio/audio_evals/models/step_audio_2_mini.py 强制注入 system prompt:
#     if not has_system:
#         messages.append({"role":"system","content":"You are a helpful assistant."})
# 而 swift 默认 --system=None, prompt 就缺失了 <|BOT|>system… 前缀, 模型分布归一.
# 同 swift-based 推理下 (run_inference_meld.sh) 、训练这里都必须一致传同一个 system.
SYSTEM=${SYSTEM:-"You are a helpful assistant."}
# 默认 epoch 从 2 上调到 3：
# - MELD 是 7 分类的高不均衡任务（neutral 47%, fear/disgust ≈ 3%）；
# - 之前 2 epoch 内 eval_rouge-l 完全水平（42.5 → 42.5），说明 LoRA 根本没被驱动
#   去区分类别，只学到了"永远输出 neutral"的多数类先验；
# - 3 epoch + 更细的 eval_steps 可以让 minority 类得到多次更新机会。
# 如需省时快速验证流程，可显式覆盖: NUM_EPOCHS=2 bash run_train_sft_lora_meld.sh
NUM_EPOCHS=${NUM_EPOCHS:-3}
# 训练早期 lr 直接拉到峰值容易冲塌 minority 类的 fine-tuning 表示，
# 加 warmup 让 LoRA 权重先小步试探 (与 SFT 常见 recipe 保持一致)。
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
# OOM 优化：默认 batch=1, grad_accum=8（等效 batch=8/GPU），显著降低单步显存峰值；
# 因为 step_audio2_mini 当前只支持 attn_impl=eager，attention 显存随 L^2 增长，必须把序列截短。
BATCH_SIZE=${BATCH_SIZE:-1}
# Eval 阶段没有反向梯度/优化器状态/激活缓存，显存开销远小于训练；
# 把 EVAL_BATCH_SIZE 从 1 提升到 8，是加速 val 最大的杠杆（吞吐近线性）。
# 但 eager attention 下 attn_weights = (B,H,L,L) 仍随 B 线性增长，
# 当 MAX_LENGTH 被显式抬高（如 4096+）时建议同步把 EVAL_BATCH_SIZE 调小到 4 或 2。
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
# eval/save 间隔 50 步 (总步约 312 * (5/2) ≈ 780) 让我们能在训练全程扫到 15+ 个 ckpt,
# 便于 sweep 时挑最佳; 如果嫌产物太多可以再拉大.
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-50}
save_total_limit=${save_total_limit:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# Val 子采样：MELD 的 dev.jsonl 总共约 1100 条，全部用上也不算大，默认不再抽样。
# 但是这里的“派生小集”流程仍然需要保留：predict_with_generate=true 路径下
# swift 遇到超长样本会直接抛 MaxLengthError 让 eval 崩溃，所以必须先按 max_length
# 过滤掉超长样本再喂给 trainer（见下方 truncation 说明）。
# 因此 VAL_MAX_SAMPLES 默认给一个远大于 val 总数的值，等价于“不限量、只做超长过滤”。
# 如仍要抽样，可通过环境变量覆盖，例如： VAL_MAX_SAMPLES=500 bash run_train_sft_lora_meld.sh
# 设为 0 表示彻底跳过派生流程、直接使用原始 VAL_JSONL（此时也不做超长过滤，慎用）。
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1000000}
# 是否打乱后再取前 N 条：由于默认不再抽样，保持原始顺序即可（更可复现）。
VAL_SAMPLE_SHUFFLE=${VAL_SAMPLE_SHUFFLE:-0}
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
# rank 从 8 → 16: 之前 r=8 + lr=1e-4 + 2 epoch 组合下 token_acc 一直卡在 ~0.85 (=多数类先验),
# 说明 LoRA 表示空间不足以捕捉音频 → 情感的判别方向; r=16 大约把可训练参数翻倍,
# 显存增长可忽略, 但对稀有类的表达力有明显提升.
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 【v7 关键修正】LoRA target_regex 必须包含 adapter.linear1/linear2
# ─────────────────────────────────────────────────────────────
# v6 实测结果 (labelsmoothing 修复后): 训练曲线看似正常 (train_loss 从 0.87→0.31,
# train_acc 0.71→0.91), 但推理阶段 99.9% 全预测 N, macro-F1 只有 9.3%.
# 根本原因: 【adapter (audio→LLM 桥梁) 完全没被训练】.
#   - 检查 checkpoint 目录只有 adapter_model.safetensors (LoRA 参数),
#     没有 adapter/aligner 全参 safetensors → 说明 adapter 参数没变
#   - 排查 swift/arguments/tuner_args.py:207 发现:
#       if not self.model_meta.is_multimodal or ... or self.tuner_type != 'full':
#           return
#     即【LoRA 模式下 freeze_aligner 参数完全无效】, 不会把 aligner 加进 trainable_parameters
#   - 排查 swift/pipelines/train/tuner.py:382 又发现:
#       elif args.tuner_type == 'full':
#           ...
#           if args.trainable_parameters ...:
#               activate_parameters(...)
#     即【LoRA 模式下 --trainable_parameters 也不生效】, activate_parameters 只在 full 分支调用
#   - 结论: LoRA 模式下唯一能让 adapter 被训练的办法就是【把 adapter 的 Linear 加进 LoRA target_regex】
#
# 之前误判 "v4 崩溃 = adapter LoRA 冲突"的锅其实是 label_smoothing (已在 v6 修好).
# 现在 v7 把 adapter.linear[12] 加回 target_regex, 与 label_smoothing=0 配合应能正常训练.
#
# 【v7 target_regex】覆盖 3 类模块:
#   1. LLM attention: q_proj / k_proj / v_proj / o_proj
#   2. LLM MLP:        gate_proj / up_proj / down_proj
#   3. audio adapter:  adapter.linear1 / adapter.linear2  ← 【关键】让 audio→情感 投影被学
# 注意: audio encoder 的 query/key/value/out 用的是自定义 Linear, 不会误匹配.
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}
LORA_TARGET_REGEX=${LORA_TARGET_REGEX:-'.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$|^adapter\.linear[12]$'}

# 【v7 说明】FREEZE_ALIGNER 在 LoRA 模式下实际【无效】:
#   - swift/arguments/tuner_args.py 的 __post_init__ 中, tuner_type != 'full' 就直接 return
#   - 所以设 true 还是 false 都不影响 aligner 的可训练性
# 但我们仍保留这个开关是为了 (1) 语义可读性, (2) 后续切到 full 模式时能沿用同一份配置.
# 若切到 full 模式: freeze_aligner=false 会把 aligner 加进 trainable_parameters (全参).
FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
# vit (audio encoder) 一律保持冻结: encoder 有 32 层 conformer, 参数量巨大.
# encoder 冻结、adapter 通过 LoRA 学习"情感投影", 是最佳平衡.
FREEZE_VIT=${FREEZE_VIT:-true}

# 【v4 新增】EARLY_STOP_INTERVAL:
# swift 支持 --early_stop_interval N: 连续 N 次 eval 指标不提升就自动停训, 节省时间
# 并避免 v3 那种 step 400 之后学偏 (转而过拟合 F) 的失败模式.
# 5 次 eval * eval_steps 50 = 250 步不提升 -> 停止, 足够宽松.
# 设为 0 表示禁用早停.
EARLY_STOP_INTERVAL=${EARLY_STOP_INTERVAL:-5}

# 评估与最佳检查点选择（分类任务现在应该看"生成后的字面是否与 GT 一致"）
# 【重要】历史配置 predict_with_generate=true + metric_for_best_model=rouge-l 在 MELD
# 上给了一个很有误导性的信号: 只输出 neutral 就能拿到 rouge-l≈42.5, 300 步内根本没变化,
# load_best_model_at_end 挑的"best" 毫无区分度. 现在切成:
#   - predict_with_generate=false: 让 trainer 在 eval 阶段跑 logits (teacher-forcing),
#     不再花费大量时间在 generate 上;
#   - eval_metric=acc: 走 swift 的 AccMetrics, 只有 logits.argmax 与 GT token 一致才算对,
#     对于"预测 neutral 覆盖一切" 会立刻显现 token_acc 停留在 ~0.47 (=neutral 先验)
#     而不是被 rouge-l 的 partial match 掩盖.
#   - metric_for_best_model=token_acc: 与 eval_metric=acc 产出的 key 一致.
# 如果你需要看生成质量, 可显式改回 PREDICT_WITH_GENERATE=true + METRIC_FOR_BEST_MODEL=rouge-l,
# 但对分类任务不推荐.
PREDICT_WITH_GENERATE=${PREDICT_WITH_GENERATE:-false}
EVAL_METRIC=${EVAL_METRIC:-acc}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-token_acc}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-true}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-true}
# Early stopping：swift / HF Trainer 没有提供命令行开关，要启用需在代码中注册
# EarlyStoppingCallback。这里退而求其次：
#   - NUM_EPOCHS 下调到 2，限制总迭代次数；
#   - load_best_model_at_end=true 保证末状态是验证集最佳点。
# 如仍需 early stopping，可后续在 swift sft trainer 注入 callback。
# 生成评估时限制新 token 数 (分类只需一两个 token)
# 生成评估时限制新 token 数.
# 【方案 1.A / 单字母标签】label 只有 1 个 token (S/A/N/J/D/F/G), max_new_tokens=1 完全够用.
# 好处:
#   • 生成阶段省掽 7-8x 的评估时间;
#   • 无法再看到之前“assistant 第一个 token 长得像 sur、但后面又变成 <tts_end>”这种泄漏情况;
#   • 不会发生模型多吧嘕“neutral\n…” 自言自语模板。
EVAL_GEN_MAX_NEW_TOKENS=${EVAL_GEN_MAX_NEW_TOKENS:-1}

# DataLoader / 显存优化
# 64 个 worker 不仅 CPU 开销大，还会放大 pinned memory，对当前 OOM 也是一个潜在压力源
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
# 让 PyTorch 在分配器里启用 expandable_segments + 更激进的回收，缓解碎片
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

# 数据集（可通过环境变量覆盖）
# meld 任务：默认指向本目录下的 data_meld/, 由 build_meld_jsonl.py + to_letter_jsonl.py
# 一起产出的单字母版 jsonl (方案 1.A):
#   train_letter.jsonl / train.balanced_letter.jsonl / dev_letter.jsonl / test_letter.jsonl
# 【为什么默认用 letter 版】
#   word 版标签 (surprise/sadness/fear/disgust 都是多个 token) 会导致:
#     * SFT 内 token_acc 被 label 中间/末尾 token 的 teacher-forcing 命中率虚高 (实测 0.82 但分类 acc 只有 0.48);
#     * inference argmax 因 tokenization 偏差 需要专门的 prefix 匹配逻辑
#   单字母版则 label 只有 1 个 token, token_acc ≡ 分类 acc, 一切指标都真实不受塑造.
#
# 【v4 复盘 v3 训练结果的关键变更: 默认不再用 balanced 上采样版】
# v3 用 train.balanced_letter.jsonl (F/G/D/A/S 都上采样到 1427) 训练后,
# checkpoint-400 出现 638 个 F fp, checkpoint-639 直接把 F 当成新的多数类,
# 分类 acc 从 48% 降到 34%. 原因: F 类只有 268 条 unique 样本, 5.3x 复制导致
# LoRA 记忆了这些少量样本作为"F shortcut", 而不是学到 fear 的判别边界.
#
# v4 默认改回原始不上采样版 train_letter.jsonl, 配合下方 label_smoothing + 更小 lr
# + adapter LoRA target 一起解决类不均衡问题, 走"温和"路线; 如果想再试 balanced
# 可显式:  TRAIN_JSONL=$DATA_DIR/train.balanced_letter.jsonl bash run_train_sft_lora_meld.sh
# 如需回退到 word 版, 显式传:
#     TRAIN_JSONL=$DATA_DIR/train.jsonl VAL_JSONL=$DATA_DIR/dev.jsonl bash run_train_sft_lora_meld.sh
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train_letter.jsonl"}
VAL_JSONL=${VAL_JSONL:-"$DATA_DIR/dev_letter.jsonl"}
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
    if [ "$VAL_MAX_SAMPLES" -ge "$VAL_FULL_LINES" ]; then
        echo "[INFO] val 集 $VAL_FULL_LINES 条 -> 全量保留并过滤超长样本(>${MAX_LENGTH} tokens) (shuffle=$VAL_SAMPLE_SHUFFLE): $VAL_SMALL_JSONL"
    else
        echo "[INFO] val 集 $VAL_FULL_LINES 条 -> 过滤超长样本(>${MAX_LENGTH} tokens) 并派生上限 $VAL_MAX_SAMPLES 条 (shuffle=$VAL_SAMPLE_SHUFFLE): $VAL_SMALL_JSONL"
    fi
    SHUF_FLAG="$VAL_SAMPLE_SHUFFLE" \
    VAL_JSONL="$VAL_JSONL" \
    VAL_SMALL_JSONL="$VAL_SMALL_JSONL" \
    VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
    MAX_LENGTH="$MAX_LENGTH" \
    AUDIO_TOKENS_PER_SEC="$AUDIO_TOKENS_PER_SEC" \
    PROMPT_RESERVED_TOKENS="$PROMPT_RESERVED_TOKENS" \
    /data/miniconda3/envs/env-3.12.11/bin/python - <<'PYEOF'
import json, os, random, wave, contextlib, sys, struct

VAL = os.environ['VAL_JSONL']
OUT = os.environ['VAL_SMALL_JSONL']
MAXN = int(os.environ['VAL_MAX_SAMPLES'])
MAX_LEN = int(os.environ['MAX_LENGTH'])
TPS = float(os.environ['AUDIO_TOKENS_PER_SEC'])
RESERVED = int(os.environ['PROMPT_RESERVED_TOKENS'])
SHUF = os.environ.get('SHUF_FLAG', '1') == '1'
MAX_AUDIO_SEC = max(1.0, (MAX_LEN - RESERVED) / TPS)

# 优先使用 soundfile（原生支持 wav/flac/ogg 等），
# 其次退回到标准库 wave（仅 wav 可用），
# 最后手工解析 FLAC STREAMINFO block 兜底（离线场景不装 soundfile 也能跑）。
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False

def _dur_soundfile(p):
    try:
        info = sf.info(p)
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None

def _dur_wave(p):
    try:
        with contextlib.closing(wave.open(p, 'rb')) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None

def _dur_flac_manual(p):
    # 解析 FLAC 头 STREAMINFO 中的 sample_rate(20bit) + total_samples(36bit)
    try:
        with open(p, 'rb') as fp:
            if fp.read(4) != b'fLaC':
                return None
            # 第一个 METADATA_BLOCK_HEADER: 1B(type) + 3B(len)
            hdr = fp.read(4)
            if len(hdr) < 4:
                return None
            block_type = hdr[0] & 0x7F
            block_len = int.from_bytes(hdr[1:4], 'big')
            data = fp.read(block_len)
            if block_type != 0 or len(data) < 18:
                return None
            # STREAMINFO 结构（bit 拼装）:
            #  min_block_size 16, max_block_size 16, min_frame 24, max_frame 24,
            #  sample_rate 20, channels 3, bits_per_sample 5, total_samples 36
            b = int.from_bytes(data[:18], 'big')
            total_samples = b & ((1 << 36) - 1)
            b >>= 36
            b >>= 5   # bits_per_sample
            b >>= 3   # channels
            sample_rate = b & ((1 << 20) - 1)
            if sample_rate <= 0 or total_samples <= 0:
                return None
            return float(total_samples) / float(sample_rate)
    except Exception:
        return None

def audio_dur(p):
    lp = p.lower()
    # flac 优先走 soundfile / 手工解析，避免 wave 抛错
    if lp.endswith('.flac'):
        if _HAS_SF:
            d = _dur_soundfile(p)
            if d is not None:
                return d
        return _dur_flac_manual(p)
    # 其它格式（wav 等）先 wave，再 soundfile 兜底
    d = _dur_wave(p)
    if d is not None:
        return d
    if _HAS_SF:
        return _dur_soundfile(p)
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
        s = audio_dur(ap)
        if s is None:
            dropped_missing += 1
            continue
        if s > MAX_AUDIO_SEC:
            dropped_long += 1
            continue
        fo.write(line + '\n')
        kept += 1

print(f'[VAL FILTER] kept={kept} dropped_long={dropped_long} dropped_missing={dropped_missing} '
      f'(MAX_AUDIO_SEC={MAX_AUDIO_SEC:.1f}s = (MAX_LEN={MAX_LEN}-RESERVED={RESERVED})/TPS={TPS:.1f}) '
      f'soundfile={"yes" if _HAS_SF else "no"}')

if kept == 0:
    print('[FATAL] 过滤后的 val 集为空 (kept=0)。'
          '常见原因：音频路径失效、或所有样本时长都超过 MAX_AUDIO_SEC、或音频格式无法解析。'
          '请检查上面的 dropped_missing / dropped_long 计数以及 VAL_JSONL 里的 audios 字段。',
          file=sys.stderr)
    sys.exit(21)
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
    # 【v4】优先使用 LORA_TARGET_REGEX (正则), 否则退回 LORA_TARGET_MODULES (模块名列表)
    # 用正则的好处: 可以精确覆盖 audio adapter 里的 `adapter.linear1 / adapter.linear2`,
    # 这两个路径带点号, 用普通 target_modules 无法命中.
    if [ -n "$LORA_TARGET_REGEX" ]; then
        TUNER_ARGS+=(--target_regex "$LORA_TARGET_REGEX")
    else
        # --target_modules 接受多个独立 token，这里以空格为分隔符将字符串
        # 拆成多个参数迫入数组，同时也兼容“q_proj,k_proj,v_proj,o_proj”
        # 这种逗号分隔的写法（自动转为空格分隔）。
        _tm_normalized=${LORA_TARGET_MODULES//,/ }
        # shellcheck disable=SC2206
        _tm_array=($_tm_normalized)
        TUNER_ARGS+=(--target_modules "${_tm_array[@]}")
    fi
    # 【v4】freeze_aligner=false 是让 LoRA 能作用到 adapter 的必要条件.
    # freeze_vit 保持默认 true, 让 audio encoder 冻结, 只调 adapter + LLM.
    TUNER_ARGS+=(
        --freeze_aligner "$FREEZE_ALIGNER"
        --freeze_vit "$FREEZE_VIT"
    )
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
else
    # 明确关闭 generate 路径，走 logits+acc, 加速一到两个数量级并让指标可信.
    EVAL_ARGS+=(--predict_with_generate false)
fi
# 把 eval_metric 传给 swift, 让 trainer 装配 AccMetrics / NlgMetrics.
# 当 PREDICT_WITH_GENERATE=false + EVAL_METRIC=acc 时, eval 会产出 token_acc,
# 与 metric_for_best_model=token_acc 完全对齐.
EVAL_ARGS+=(--eval_metric "$EVAL_METRIC")
EVAL_ARGS+=(
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
    --greater_is_better "$GREATER_IS_BETTER"
    --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"
)

echo "[INFO] PREDICT_WITH_GENERATE=$PREDICT_WITH_GENERATE EVAL_METRIC=$EVAL_METRIC METRIC_FOR_BEST_MODEL=$METRIC_FOR_BEST_MODEL"
echo "[INFO] LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END"
echo "[INFO] LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA"
echo "[INFO] LORA_TARGET_MODULES=$LORA_TARGET_MODULES"
echo "[INFO] LORA_TARGET_REGEX=$LORA_TARGET_REGEX"
echo "[INFO] FREEZE_ALIGNER=$FREEZE_ALIGNER FREEZE_VIT=$FREEZE_VIT"
echo "[INFO] WARMUP_RATIO=$WARMUP_RATIO NUM_EPOCHS=$NUM_EPOCHS LEARNING_RATE=$LEARNING_RATE"
echo "[INFO] WEIGHT_DECAY=$WEIGHT_DECAY LABEL_SMOOTHING=$LABEL_SMOOTHING EARLY_STOP_INTERVAL=$EARLY_STOP_INTERVAL"

# 【v4】早停参数: 仅当 EARLY_STOP_INTERVAL > 0 时启用
EARLY_STOP_ARGS=()
if [ "${EARLY_STOP_INTERVAL:-0}" -gt 0 ]; then
    EARLY_STOP_ARGS+=(--early_stop_interval "$EARLY_STOP_INTERVAL")
fi

# 启动训练
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" sft \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    --system "$SYSTEM" \
    "${TUNER_ARGS[@]}" \
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
    --seed $SEED \
    --report_to tensorboard \
    --save_total_limit ${save_total_limit} \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --save_only_model true \
    "${EARLY_STOP_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "$@"

    # --new_special_tokens '<speak_reply>' '<speak_backchannel>' '<tts_start>' '<tts_end>' \
echo "[INFO] 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
