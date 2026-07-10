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
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# ★ 强烈建议 GRPO 从一个已经 SFT 过的 ckpt 继续训练（冷启动 GRPO 几乎不收敛）。
#
# v4/v6/v9/v10 事故复盘（评测数据 2026-07-03）：
#   * v4 (base 冷启动, 4G, temp=1.0, top_p=0.95, beta=0.04):
#         acc 68.5% / macro-F1 27% (200 步) → 26% (1400 步)
#         speech 86% / noise 55% / rare 全 0% —— 只是"rare 塌"。
#   * v6 (base 冷启动, 8G, temp=1.2, top_p=1.0, top_k=0, beta=0.02, weighted-r):
#         acc 66.5% / macro-F1 16% —— 连 noise (support 1137) 都塌了；
#         confusion: noise/porn/song/music 全部被预测成 speech。
#   * v9 (同 v6 + stratified 数据):
#         acc 66.3% / macro-F1 16% —— 与 v6 完全相同的 speech-only 崩塌。
#   * v10 (SFT-warmup + 8G + temp=1.0 + top_p=0.95 + beta=0.04 + weighted+format):
#         base = SFT ckpt-1600-merged  →  acc 97.9% / macro-F1 95.7%
#         GRPO ckpt-150               →  acc 80.7% / macro-F1 62.8%（-33 pp!!）
#         GRPO ckpt-300               →  acc 79.1% / macro-F1 59.2%（继续塌）
#         rare-class 全线倒退：
#             noise recall 98.1% → 19.9% → 11.3%
#             porn  recall 93.5% →  9.5% →  6.5%
#             song  recall 95.1% → 84.4% → 79.9%
#         训练日志：KL 从 step 1 的 0.005 → step 100 的 1.31 → step 200 的
#         2.06（400× 暴涨），frac_reward_zero_std 从 10% → 42%，clip_high≈0
#         而 clip_low≈3%（正梯度几乎没触发，负梯度稳定通过）。
#
# v10 → v11 的核心诊断：
#   base（SFT）已经 macro-F1 95.7%，处在"最优点附近"，此时 GRPO 的期望
#   收益 (EV) 是 **负** 的 —— 组内 rollout 的 advantage 信号极其稀疏、
#   且被"偶发错误"主导；而 KL 漂移是持续的，最终 policy 稳定地朝
#   majority class (speech) 移动，把 rare-class 全部拆掉。
#
#   具体几条锁链：
#   (a) BETA=0.04 太弱，压不住 KL；
#   (b) train.balanced.jsonl 里 speech 28.6%，speech-group 是唯一稳定
#       产生正梯度的组，policy 的更新方向被 speech 主导；
#   (c) weighted reward (porn=3.5 / song=4.0) 反而放大 rare-class 偶
#       发错误的负 advantage（8 条 rollout 只要错 1 条，那条负 advantage
#       就被 group std 归一后再乘 3.5+，policy 学到"少出 rare 类"）；
#   (d) format reward 全 1.0（SFT 后从不出格式错误），std=0 完全无梯度，
#       只是给 loss 加常数、稀释 accuracy；
#   (e) dynamic_sample=true 丢弃了 40% 的 std=0 组，剩下的边界样本
#       其实很多是 SFT 已答对但采样时抖动的样本，梯度反而把它们从对
#       推到错；
#   (f) LR=2e-6 + warmup 15 步：KL 在 warmup 结束前就开始爆，学习率
#       进入不了 cosine 尾部；
#   (g) MAX_STEPS=500 太长：ckpt-150 已是全程最好，之后每一步都在退化。
#
# 结论 —— v11 (对症下药，脚本默认参数已按下调整)：
#   (A) BETA 0.04 → 0.2   （5× 加大 KL 拘束，压住漂移）
#   (B) LR   2e-6 → 5e-7  （4× 缩小步长）
#   (C) TEMPERATURE 1.0 → 0.7 / TOP_P 0.95 → 0.9 / TOP_K 50 → 20
#         （降低采样噪声，减少假正/假负 advantage 信号）
#   (D) DYNAMIC_SAMPLE true → false（保留 KL 项对全对/全错组的作用）
#   (E) EPSILON_HIGH 0.28 → 0.24（收紧正 clip，配合负 clip 更对称）
#   (F) REWARD_FUNCS: weighted+format → 单一 stepaudio_accuracy
#         REWARD_WEIGHTS: "1.0 0.1" → "1.0"
#         （消除 weighted 对 rare-class 错误的放大，去掉常量 format 项）
#   (G) MAX_STEPS 500 → 150 / SAVE_STEPS 25 → 10 / GRAD_ACCUM 4 → 8 /
#       WARMUP_RATIO 0.03 → 0.1
#         （最短训练窗口 + 密采样 + 更稳的 effective batch + 缓启动）
#
# 期望目标（v11）：ckpt 在 acc 保持 >97% 的前提下，rare-class recall
#   相比 SFT baseline 变化 |Δ| < 3pp，即 GRPO 至少不倒退；如果观察到
#   任何 rare-class 塌陷，立即启用早停并把 BETA 抬到 0.3~0.5。
#
# ============================ SFT warmup 三步流程 ============================
#   1) SFT warmup（约 1000~2000 步足够让 rare-class 蒙对率 >20%）：
#         bash project/stepaudio/run_train_swift.sh
#      产出 checkpoint-XXXX (adapter 目录) 在
#         output/stepaudio/sft/vN-yyyymmdd-HHMMSS/checkpoint-XXXX/
#
#   2) merge LoRA (本脚本 _ensure_full_model_dir 会在检测到 adapter 目录时
#      自动执行，也可手动执行):
#         swift export --adapters <sft_ckpt> --merge_lora true \
#                      --output_dir <sft_ckpt>-merged
#
#   3) 启动 GRPO（推荐路径）：
#         MODEL_PATH=<sft_ckpt> bash project/stepaudio/run_train_grpo.sh
#      或（已 merge）：
#         MODEL_PATH=<sft_ckpt>-merged bash project/stepaudio/run_train_grpo.sh
#
#   如坚持从 base 冷启动（不推荐，会重现 v4/v6/v9 曲线）：
#         STRICT_SFT_WARMUP=0 bash project/stepaudio/run_train_grpo.sh
# =========================================================================
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
if [ "$MODEL_PATH" = "/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini" ]; then
    echo ""
    echo "[WARN] ============================================================"
    echo "[WARN] MODEL_PATH 指向的是原始 base 模型（未 SFT）。"
    echo "[WARN] 三次实验复盘（v4/v6/v9）已证明冷启动 GRPO 无法学会 rare-class："
    echo "[WARN]   v4  macro-F1 27%（rare 全 0）"
    echo "[WARN]   v6  macro-F1 16%（连 noise 都塌成 speech-only）"
    echo "[WARN]   v9  macro-F1 16%（与 v6 相同崩塌）"
    echo "[WARN] 强烈建议先跑 SFT warmup 再进入 GRPO："
    echo "[WARN]     bash project/stepaudio/run_train_swift.sh"
    echo "[WARN]     然后 MODEL_PATH=<sft_ckpt> bash \$0"
    echo "[WARN] 若坚持从 base 冷启动，请显式设置：STRICT_SFT_WARMUP=0"
    echo "[WARN] ============================================================"
    if [ "${STRICT_SFT_WARMUP:-1}" = "1" ]; then
        echo "[FATAL] 拒绝冷启动 GRPO（STRICT_SFT_WARMUP=1，默认）。请指定 SFT 后的 MODEL_PATH。"
        exit 16
    fi
    echo "[WARN] 已通过 STRICT_SFT_WARMUP=0 允许冷启动，请注意 rare-class 崩塌风险。"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/stepaudio/grpo"}
# ms-swift 默认 add_version=True, 会在 OUTPUT_DIR 后自动追加 v<idx>-<timestamp>
# 子目录 (见 swift/arguments/sft_args.py 中 _add_version)。设 ADD_VERSION=false
# 可让 checkpoints 直接落到 $OUTPUT_DIR 下, 便于外部脚本按固定路径消费。
# 注意：关闭 add_version 后，若同一 OUTPUT_DIR 重复启动训练会覆盖之前的
# checkpoints/logs, 请自行确认是否是想要的行为。
ADD_VERSION=${ADD_VERSION:-true}

# ---------------- Tuner ----------------
# lora（默认，强烈推荐）/ full（高显存）
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-6   # GRPO + full 必须用更小 LR，否则 KL 直接爆炸
else
    # v11: 2e-6 → 5e-7。v10 (LR=2e-6) evaluation：
    #   ckpt-150 acc 80.7% / macro-F1 62.8%（比 SFT baseline 97.9% / 95.7% 大幅退步），
    #   ckpt-300 acc 79.1% / macro-F1 59.2%（继续往 speech-only 漂）。
    #   训练日志显示 KL 从 step-1 的 0.005 → step-100 的 1.31 → step-200 的
    #   2.06（400× 暴涨），BETA=0.04 完全拉不住。
    # 结论：base（SFT ckpt-1600-merged, macro-F1 95.7%）已经非常强，GRPO 更
    # 新的 EV 是负的——组内几乎没有正 advantage 信号，噪声梯度却在持续把
    # policy 从最优点推走。必须把 LR 再压 4×，配合更强的 KL/更早的早停。
    DEFAULT_LR=5e-7
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（仅 TUNER_TYPE=lora 生效）
# NOTE: 历史坑
#   - v0-20260630-141430 实验在 ckpt-1400→2800 发生明显 mode collapse（porn/music/song
#     recall=0），根因是 train.jsonl 五类分布严重失衡（speech 69% / song 3.7%），
#     GRPO 组内几乎抽不到少数类样本 → 稳定坍缩到 majority class；
#   - v2-20260701-105056 用 rank=4 / dropout=0.1 想靠正则解决，但 374 步内 porn GT
#     一次都没被采到（数据侧问题，正则救不了）。
#   - v3 起改用 train_balanced.jsonl（五类各 20%，见 tools/balance_train_jsonl.py），
#     数据均衡后不再需要极端正则，rank 恢复到 8、dropout 收回 0.05，让 policy 有
#     足够表达力去学习 minority class 的判别边界。
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# 与 sft 脚本一致：仅注意力 4 个 proj，避免破坏多模态 connector
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- GRPO 关键超参 ----------------
# ★★★ v4→v5 大改：v4 (checkpoint 200/800/1400) 评测显示典型的
# "稀有-且-难-类别 GRPO 冷启动死锁"（porn recall 8% → 1.5% → 0.3% 指数衰减）：
#
#   * v4 训练 1400 步 × 32 completion/step = 45248 次生成里，模型只吐出 6 次
#     "porn" token（confusion: speech→porn=3, music→porn=2, song→porn=1）；
#   * 每次遇到 label=porn 的 batch，4 条 completion 全部为非 porn → reward=0 →
#     组内 std=0 → dynamic_sample 直接 skip 该组 → **porn 从未产生过正梯度**；
#   * beta=0.04 的 KL 项把 policy 拉回 "永不输出 porn" 的初始分布。
#
# 修复思路（每一项都是死锁的一根锁链）：
#   (1) 必须先 SFT warmup：GRPO 无法"从零学会做题"，需要 base model 至少对
#       porn 有 ~20% 的蒙对率，才能启动 group-relative 学习。
#       操作：把 MODEL_PATH 指向 SFT 后的 merged ckpt（详见 SFT_WARMUP_CKPT）。
#   (2) NUM_GENERATIONS 4→8：G 更大更容易在 rare-class group 里凑出 std>0；
#   (3) TEMPERATURE 1.0→1.2：提升组内多样性；
#   (4) DYNAMIC_SAMPLE true→false：不再丢弃 std=0 组，保留 KL/exploration 信号；
#   (5) SCALE_REWARDS group→none：group std 归一会把 rare-class 的微弱正信号
#       与 majority-class 的强信号"归一到同尺度"，反而弱化 rare-class；none
#       (Dr.GRPO) 让绝对 reward 差直接进 loss；
#   (6) BETA 0.04→0.02：减小 KL 拘束，允许 policy 探索 rare token；
#   (7) TOP_P 0.95→1.0 / TOP_K 50→0：**关键**——如果 porn 的先验概率 < 5%，
#       top-p=0.95 会直接把它从采样池里砍掉，porn 永远不出现在 completion 里；
#   (8) EPSILON_HIGH 0.28→0.32：一旦碰上 porn 蒙对，让正向 clip 更充分吸收；
#   (9) LOG_ENTROPY=true：观察 entropy 是否随训练 collapse；
#   (10) MAX_STEPS 4000→2000：v0 最佳 ckpt=1400；balanced 数据每步信息量高，
#        再长也是过拟合。

# 每个 prompt 采样多少 completion 用于 group-relative advantage 估计。
# v4 用 4 / v6-v9 用 8。v10 保留 8：v4 的 std=0 死锁在 SFT-warmup 之后不再
# 是主要瓶颈，8 让 rare-class 组内更容易凑出 std>0。
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
# 单条 completion 最长生成多少 token。stepaudio 输出基本是 1 个标签词，
# 给 16 token 留点 BOS/EOS 余量足够；过大会显著放慢 sampling。
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-16}
# 探索温度：v11 从 1.0 → 0.7。v10 completions 分析发现 base 模型对 rare-class
# 已经非常确定（entropy/mean 通常在 0.02~0.13），温度 1.0 下 8 条 rollout 里
# 出现"意外错到 speech"的概率反而放大——group-relative advantage 会把这条
# 错样本的负梯度传给整个 group，加速漂移。0.7 让 rollout 更贴近 policy 的
# argmax 分布，减少假信号。
TEMPERATURE=${TEMPERATURE:-0.7}
# PPO clip 区间。v11 收紧到 0.2/0.24（v10 是 0.2/0.28）。v10 观察到
# clip_ratio/high ≈ 0（正梯度几乎没被 clip，说明可吸收的正信号本就很少），
# clip_ratio/low ≈ 3%（负梯度稳定通过）——把 epsilon_high 收窄可减少偶发
# speech 正 advantage 被过度放大。
EPSILON=${EPSILON:-0.2}
EPSILON_HIGH=${EPSILON_HIGH:-0.24}
# scale_rewards 合法值：{group, batch, none, gdpo}
# v10 回滚到 v4 的 group：v6/v9 换 none 的假设（"none 让 rare-class 的绝对
# reward 优势直接进 loss"）在冷启动场景不成立——rare-class 组几乎全 0，
# 绝对优势也是 0。而 group 归一可以放大"偶然一条对的" completion 的
# advantage，反而是 rare-class 少数正梯度的关键。SFT-warmup 之后 rare-class
# 蒙对率 >20%，group 归一才真正开始给出稳定信号。
SCALE_REWARDS=${SCALE_REWARDS:-group}
# Reward 加权：accuracy 是主指标；v2 观测到 completions/mean_length≈2.5 已经受控，
# 无效响应（<tts_*> 泄漏）比例极低，format 权重回到 0.1 即可。
# v7 起默认换成 stepaudio_accuracy_weighted（inverse-frequency 权重），rare-class
# 判对时 reward = w[gt]（>=1），majority-class 判对时 reward = 1；从而在 GRPO
# group-relative advantage 里给 rare-class 一个绝对更大的正向信号，弥补它们
# rollout 命中率低导致的 std 归一后 signal 被压扁的问题。
# 权重可通过 STEPAUDIO_CLASS_WEIGHTS='speech=1.0,noise=1.5,music=3.0,porn=3.5,song=4.0' 覆盖。
#
# v11 复盘：v10 用 [1.0, 0.1] 组合 + weighted accuracy 反而加剧 rare-class 塌缩：
#   * SFT baseline rare-class recall 已经 93~98%，rare-class group 里
#     rollout 通常是 8/8 全对（advantage=0）或偶尔 1/8 错（错的那条负
#     advantage 被 group std 归一后放大到很大）；
#   * weighted reward (porn=3.5 / song=4.0) 又把这个负 advantage 再放大数倍，
#     policy 学到的是"少出 rare 类以躲避大罚"；
#   * format reward 因为 base 早已 100% 满足格式，std=0 完全无梯度贡献，
#     只是给 loss 抬常数。
# 结论：base 强的场景要
#   (a) 关掉 weighted，改回等权 accuracy；
#   (b) 去掉 format 项（无信号）；
#   见下方 REWARD_FUNCS 默认值。REWARD_WEIGHTS 单值 1.0 与之匹配。
REWARD_WEIGHTS=${REWARD_WEIGHTS:-"1.0"}
# 梯度裁剪，KL 爆炸时第一道闸门
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# 抗 mode-collapse / 过拟合的额外开关
# BETA: KL 惩罚系数。v11 从 0.04 → 0.2（5×）。v10 训练日志：
#   step 1  : kl 0.005
#   step 100: kl 1.31   （BETA=0.04 已经压不住）
#   step 200: kl 2.06   （policy 严重漂离 SFT baseline）
#   同时期 evaluation：ckpt-150 macro-F1 62.8%，ckpt-300 崩到 59.2%。
# base（SFT）已经很强，reference log-prob 就是"最优附近"，KL 项必须
# 足够强（0.2 起步）才能把每步更新压到局部微调，避免 policy 漂到
# speech-only 塌缩点。若 kl 稳定 <0.3 且 rare-class recall 无退步，可尝试
# 逐步下调到 0.1；若 kl 仍然 >1.0，抬到 0.3~0.5。
BETA=${BETA:-0.2}
# TOP_P/TOP_K: v11 收紧到 0.9/20（v10 是 0.95/50）。base 模型已经在
# rare-class 上 recall 93~98%，采样池里前 5 个 token 覆盖 >99% 的正确答案；
# 更宽的采样池只会引入"意外错误"→ group 负 advantage → 漂移。0.9 是
# nucleus 的常用低温挡位，同时保留必要的多样性（若 top_k=20 卡住则触发
# nucleus 兜底）。
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-20}
# 动态采样：v11 关闭（true → false）。v10 训练日志：
#   frac_reward_zero_std 从 step 1 的 10% 上升到 step 200 的 42% ——
#   base 已经太强，一半组内 8 条 rollout 全对/全错，dynamic_sample 会把
#   这些组直接丢弃。剩下的"边界样本"其实很多是 SFT 已经答对但采样时抖了
#   一次的样本，梯度反而是把 policy 从对推到错。
# 关掉 dynamic_sample：
#   (1) 保留 KL 项对 zero-advantage 组的作用（把 policy 拉回 SFT ref）；
#   (2) 用更强的 BETA=0.2 保证漂移被压制；
#   (3) 减少"只学错样本"导致的 majority-shift。
# 若 v11 观察到训练完全无梯度（loss ~ 0 且 reward flat），可放回 true。
DYNAMIC_SAMPLE=${DYNAMIC_SAMPLE:-false}
# LOG_ENTROPY: 记录 policy entropy，观察是否 collapse。v4 tensorboard 里 entropy
# 尺度变化能第一时间反映 rare-token 是否被 softmax 挤压到 0。
LOG_ENTROPY=${LOG_ENTROPY:-true}
# 注意：swift rlhf 当前无 --mask_truncated_completions CLI 参数，故不再注入；
# 若需过滤超长被截断样本，可在生成阶段减小 MAX_COMPLETION_LEN，或后续版本升级后再引入。

# ---------------- 稳定性 / 恢复训练（可选） ----------------
# 以下变量默认已有"抗坍缩"值（BETA/TOP_P/TOP_K 见上），如需回到旧行为可显式
# 传空字符串（例如 BETA=""）。
#
# RESUME_CHECKPOINT : 从已有 ckpt 恢复训练；传入 checkpoint-XXXX 目录绝对路径。
# RESUME_ONLY_MODEL : 只加载模型权重，不恢复 optimizer/scheduler/RNG/DS 分片状态。
#                  本脚本默认 --save_only_model true，保存的 ckpt 里没有 optimizer 等
#                  完整训练状态；此时恢复训练必须设 RESUME_ONLY_MODEL=true，否则
#                  transformers 会报 "Can't find a valid checkpoint"。
#                  等价于把 ckpt 当作"新的初始化权重"，从 step 0 重新走 warmup。
#                  当 RESUME_CHECKPOINT 非空时，本变量默认 true；可显式覆盖。
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-true}
else
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-}
fi

# ---------------- 训练规模 ----------------
# v11: v10 evaluation 显示 150 步已经是 GRPO 的"最佳（也是最不坏）ckpt"，
#   ckpt-300 已经在往 speech-only 漂。既然 base 强、GRPO 是负 EV，训练窗口
#   越短越好——只保留一个明确的"停在 KL 尚未爆炸"的窗口。
#   (1) MAX_STEPS 500 → 150（配合 BETA=0.2 + LR=5e-7，把 KL 压在 <0.5
#       的目标区间内）；
#   (2) SAVE/EVAL_STEPS 25 → 10（15 个 ckpt / 15 次 eval，密采样挑最佳）；
#   (3) WARMUP_RATIO 0.03 → 0.1（缓启动，避免 step 1~15 之间 LR 突然拉起
#       就把 policy 撞崩）。
NUM_EPOCHS=${NUM_EPOCHS:-1}
# GRPO 一步实际计算量 ≈ batch * G，因此 BATCH_SIZE 一般取 1，靠 G 提供组内对比。
BATCH_SIZE=${BATCH_SIZE:-1}
# EVAL_BATCH_SIZE 必须满足：per_device_eval_batch_size × world_size 能被
# num_generations_eval（默认 = NUM_GENERATIONS）整除，否则 TRL GRPOConfig 会报
#   ValueError: The global eval batch size (X * Y) must be divisible by the
#                number of generations used for evaluation (G).
# NUM_GENERATIONS=8 + 4 卡 → per_device_eval_batch_size 至少 = 2（2*4=8 ✓）；
# 若卡数为 8 则 1 也可（1*8=8 ✓）。默认设 2 覆盖 2/4 卡场景；
# 8 卡时可显式 EVAL_BATCH_SIZE=1 省显存。脚本下面会做 sanity check 自动兜底。
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
# NUM_GENERATIONS 4→8 使 per-step 生成成本 ×2，为保持 optimizer 更新节奏，把
# grad_accum 8→4，让 effective batch = 4*4*8 = 128 个 completion，与 v4 (4*4*8=128)
# 一致，训练时长基本持平；OOM 则回落 GRAD_ACCUM=8。
# v11: 4 → 8。base 强的场景下噪声梯度是主要问题，把 effective batch 翻倍
# （4 卡 × 1 × 8 × G=8 = 256 completions/step）能显著减小方差。
GRAD_ACCUM=${GRAD_ACCUM:-8}
# v11: 0.03 → 0.1。v10 warmup 只有 15 步，KL 从 step 1 就开始堆积；缓启动
# 让前 50 步 LR 极小，policy 不会在早期就漂离 reference。
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
# MAX_STEPS: 提前终止训练总步数。
#   >0  : 显式限制到该步数（推荐，用于早停 / 节省时间）；
#   0/负: 走完整个 epoch（v6 数据 4 卡下 ≈ 3875 步 / 约 2 天）。
# v4 观测：Overall acc 在 800 步已到 91.5%，1400 步 91.6%（收敛平台），继续训
# 只会让 rare-class recall 更差；v5/v6 缩到 2000 已经充分覆盖收敛区且省算力。
# v7 目标：验证 stratified interleave + weighted reward 是否能把 mean_recall
# 拉到 >0.6；先跑 1500 步（略长于 v4 的 1407 步）以便与 v4 直接对比。
MAX_STEPS=${MAX_STEPS:-150}

# checkpointing
# v11: 10 步 × 150 步上限 = 15 个 ckpt，密采样窗口足够挑到 rare-class 未
# 塌缩的最佳点。若关心节省磁盘，可将 SAVE_STEPS 抬到 15/20。
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-10}
save_total_limit=${save_total_limit:-20}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---------------- 序列 / 截断 ----------------
# 增加到 2048 以容纳较长的音频序列（attention softmax buffer 随 L^2 增长）
MAX_LENGTH=${MAX_LENGTH:-2048}
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
# v6 (v5 参数配套) 使用 train_balanced_v2.jsonl (温和均衡, 62000 行, 5 类分布
# speech 32% / porn 19% / noise 19% / music 16% / song 13%)。相比 v3 的
# train_balanced.jsonl (严格 20%×5 = 195330 行) 有三个关键改进：
#   1) 保留 speech 少量优势 (32% vs 20%), 更接近 val 集真实分布 (69%),
#      避免 GRPO 在均衡数据上"学偏"后到 val 时 over-predict 稀有类;
#   2) rare-class 上采样倍数从 14~19x 降到 3~4x, 显著减少"重复记忆同一
#      音频"的过拟合风险 (v3 的 porn 是 2725 复制 14 次);
#   3) 总量 62k 与原始 56k 接近, 1 epoch 步数从 v3 的 6100 降到约 3875,
#      配合 MAX_STEPS=2000 训练时长可控 (~1/3 v3 时长)。
# 若要回到 v3 的严格均衡 (对照实验) 或原始失衡数据:
#   TRAIN_JSONL=$SCRIPT_DIR/data/train_balanced.jsonl  bash run_train_grpo.sh   # v3 old
#   TRAIN_JSONL=$SCRIPT_DIR/data/train.jsonl           bash run_train_grpo.sh   # 原始失衡
# ---- v10 默认：train.balanced.jsonl（已存在的温和均衡数据，28000 行，5 类
#      分布 speech 28.6% / noise 28.6% / music 14.3% / porn 14.3% / song 14.3%）
#
# 选它的三个理由：
#   1) 直接可用，无需再生成；文件在 project/stepaudio/data/train.balanced.jsonl。
#   2) 温和均衡：rare 三类 14.3% 相比 val 的 4-5% 放大 ~3x，是让 GRPO 在 rare
#      class 上产生 std>0 组的必要放大倍数（v3 严格 20%×5 放大 4-5x 反而
#      引入过多重复样本，v10 现有的 14.3% 是折中）。
#   3) 与 val 集 (69/17/5/5/4%) 的分布 shift 比 v7 stratified 更小。
#
# v7 的 stratified interleave（train_stratified_v7.jsonl）实际测下来
# （v9 evaluation）与 v6 的全局 shuffle 结果**完全一样**（macro-F1 都是 16%）——
# 说明 stratified 顺序对 GRPO 塌缩的作用可以忽略；真正决定成败的是
# BETA / TOP_P / SFT-warmup。因此 v10 不再默认走 stratified，改用现成数据。
#
# 若要对照实验：
#   TRAIN_JSONL=$DATA_DIR/train_stratified_v7.jsonl bash run_train_grpo.sh   # v9 stratified
#   TRAIN_JSONL=$DATA_DIR/train_balanced_v2.jsonl   bash run_train_grpo.sh   # v6 温和均衡 62k
#   TRAIN_JSONL=$DATA_DIR/train.jsonl               bash run_train_grpo.sh   # 原始失衡
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.balanced.jsonl"}
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[FATAL] TRAIN_JSONL 不存在: $TRAIN_JSONL"
    echo "        v10 默认使用已存在的 train.balanced.jsonl（28000 行温和均衡）。"
    echo "        若数据被误删，可用以下命令重新生成一份等价的（推荐 balanced_v2, 62000 行）:"
    echo "        python $SCRIPT_DIR/tools/balance_train_jsonl.py \\"
    echo "               -i $DATA_DIR/train.jsonl \\"
    echo "               -o $DATA_DIR/train_balanced_v2.jsonl \\"
    echo "               --custom-target 'speech=20000,noise=12000,music=10000,porn=12000,song=8000' \\"
    echo "               --pre-filter-est-L 4096 --pre-filter-workers 16 --pre-filter-use-librosa"
    echo "        然后："
    echo "        TRAIN_JSONL=$DATA_DIR/train_balanced_v2.jsonl bash \$0"
    echo ""
    echo "        或使用 v7 stratified 数据集（对照实验）:"
    echo "        python $SCRIPT_DIR/tools/balance_train_jsonl.py \\"
    echo "               -i $DATA_DIR/train.jsonl \\"
    echo "               -o $DATA_DIR/train_stratified_v7.jsonl \\"
    echo "               --custom-target 'speech=20000,noise=12000,music=10000,porn=12000,song=8000' \\"
    echo "               --stratified-interleave \\"
    echo "               --stratified-cycle 'speech,noise,music,porn,song,speech' \\"
    echo "               --pre-filter-est-L 4096 --pre-filter-workers 16 --pre-filter-use-librosa"
    exit 14
fi
# GRPO 直接用 split_dataset_ratio 从 train 划 1% 做 eval（与 event grpo 脚本一致）
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- Reward / Plugin ----------------
PLUGIN_PATH=${PLUGIN_PATH:-"$SWIFT_ROOT/examples/train/grpo/plugin/stepaudio/stepaudio_plugin.py"}
# v10 默认使用 weighted accuracy（inverse-frequency 类加权）+ format。
# 说明：weighted reward 在 v6/v9 (base 冷启动) 里"看起来无效"是因为 policy 已经
# 塌缩到只输出 speech，rare class 组的 tp 全为 0，权重乘 0 = 0——权重是
# "救火水"不是"防火墙"。SFT-warmup 后 rare-class 蒙对率 >20%，权重才真正
# 开始产生正 advantage 差异。要恢复等权 accuracy：
#   REWARD_FUNCS="stepaudio_accuracy stepaudio_format" bash run_train_grpo.sh
# v11 默认改回等权 accuracy 单目标（去掉 weighted、去掉 format）。理由见
# REWARD_WEIGHTS 上方注释：
#   * base 强 (macro-F1 95.7%) 场景下，weighted 只会放大 rare-class 偶发错误
#     的负 advantage，加速 speech-only 塌缩；
#   * format reward 在 SFT 后 std=0（永远给 1.0），完全没有梯度信号，只是
#     稀释 accuracy 的 loss 权重。
# 若想恢复 weighted+format 组合（例如 base 较弱、rare-class recall <50% 时）：
#   REWARD_FUNCS="stepaudio_accuracy_weighted stepaudio_format" \
#   REWARD_WEIGHTS="1.0 0.1" bash run_train_grpo.sh
REWARD_FUNCS=${REWARD_FUNCS:-"stepaudio_accuracy"}

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
echo "[INFO] STABILITY: BETA='${BETA}' TOP_P='${TOP_P}' TOP_K='${TOP_K}' DYNAMIC_SAMPLE='${DYNAMIC_SAMPLE}' LOG_ENTROPY='${LOG_ENTROPY}'"
echo "[INFO] RESUME_CHECKPOINT='${RESUME_CHECKPOINT}'  RESUME_ONLY_MODEL='${RESUME_ONLY_MODEL}'"

# ---------------- GPU 占用前置检查（与 SFT 脚本一致） ----------------
# 目的：拦截"两个 GRPO 训练同时抢同一批 GPU"这种坑（现象是 step_time 从 40s 恶化到
# 240s，参见 v3-20260701-155217 事故复盘）。检查两层：
#   1) 目标 GPU 上的 already-used memory > GPU_PREALLOC_GUARD_MB 视为忙
#   2) 目标 GPU 上存在 python/swift 训练进程 视为忙（更精准）
# 触发任一条件即 abort，除非显式 GPU_PREALLOC_SKIP=1（旧兼容）或 FORCE=1（新推荐名）。
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况："
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader || true

    GPU_PREALLOC_GUARD_MB=${GPU_PREALLOC_GUARD_MB:-1024}
    GPU_PREALLOC_SKIP=${GPU_PREALLOC_SKIP:-${FORCE:-0}}
    if [ "$GPU_PREALLOC_SKIP" != "1" ]; then
        IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        _busy=0
        for _gid in "${_GPU_IDS[@]}"; do
            _used=$(nvidia-smi --id="$_gid" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
            _procs=$(nvidia-smi --id="$_gid" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -Ei 'python|swift|torchrun|deepspeed' || true)
            if [ "$_used" -gt "$GPU_PREALLOC_GUARD_MB" ] || [ -n "$_procs" ]; then
                echo "[FATAL] GPU $_gid 已被占用 ${_used} MiB (阈值 ${GPU_PREALLOC_GUARD_MB} MiB)"
                if [ -n "$_procs" ]; then
                    echo "        检测到疑似训练进程："
                    echo "$_procs" | sed 's/^/            /'
                fi
                _busy=1
            fi
        done
        if [ "$_busy" = "1" ]; then
            echo ""
            echo "        GRPO 与其它训练共用 GPU 会让 step_time 线性劣化数倍（历史坑）。"
            echo "        处置建议："
            echo "          1) 找到旧训练进程后 kill：pgrep -af '(swift|torchrun).*rlhf' | awk '{print \$1}' | xargs -r kill"
            echo "          2) 或显式换空闲卡：export CUDA_VISIBLE_DEVICES=<free_ids>"
            echo "          3) 或明确要共享（不推荐）：FORCE=1 bash $(basename "$0")"
            exit 11
        fi
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡, 且无其它 python/swift 训练进程)"
    else
        echo "[WARN] 已通过 GPU_PREALLOC_SKIP=1 / FORCE=1 跳过 GPU 占用检查，可能与其它训练抢卡。"
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

# ---------------- LoRA adapter checkpoint 自动 merge ----------------
# 场景：MODEL_PATH 指向 SFT 后保存的 checkpoint-XXXX 目录，但该目录只是 LoRA adapter
#       （只含 adapter_config.json / adapter_model.safetensors，没有 config.json /
#        tokenizer 文件 / base 权重）。此时直接 --model 传该目录，AutoTokenizer 会在
#       adapter 目录里找不到 tokenizer 文件，抛 TypeError: stat NoneType。
# 处理策略：
#   1) 检测 MODEL_PATH 是否是纯 adapter 目录（存在 adapter_config.json 且不存在 config.json）；
#   2) 若已存在同级 <ckpt_name>-merged → 直接复用；
#   3) 否则调用 `swift export --adapters MODEL_PATH --merge_lora true --output_dir ...`，
#      swift 读取 adapter_config.json 中的 base_model_name_or_path 加载 base 并合并 LoRA；
#   4) 切换 MODEL_PATH 到 merged 目录（含完整 base 权重 + tokenizer + processor）。
_ensure_full_model_dir() {
    local mdir="$1"
    if [ ! -d "$mdir" ]; then
        echo "$mdir"
        return 0
    fi
    if [ -f "$mdir/config.json" ]; then
        echo "$mdir"
        return 0
    fi
    if [ ! -f "$mdir/adapter_config.json" ]; then
        echo "$mdir"
        return 0
    fi

    local mdir_abs
    mdir_abs="$(cd "$mdir" && pwd)"
    local parent base merged
    parent="$(dirname "$mdir_abs")"
    base="$(basename "$mdir_abs")"
    merged="$parent/${base}-merged"

    if [ -f "$merged/config.json" ]; then
        echo "[INFO] 检测到已存在的 merged 目录，直接复用: $merged" 1>&2
        echo "$merged"
        return 0
    fi

    echo "[INFO] MODEL_PATH 是 LoRA adapter 目录，自动执行 swift export --merge_lora ..." 1>&2
    echo "[INFO]   adapter : $mdir_abs" 1>&2
    echo "[INFO]   output  : $merged" 1>&2
    if ! "${SWIFT_CMD[@]}" export \
            --adapters "$mdir_abs" \
            --merge_lora true \
            --output_dir "$merged" 1>&2; then
        echo "[FATAL] swift export --merge_lora 失败，请手动执行以下命令后重试：" 1>&2
        echo "        ${SWIFT_CMD[*]} export --adapters $mdir_abs --merge_lora true --output_dir $merged" 1>&2
        return 1
    fi

    if [ ! -f "$merged/config.json" ]; then
        echo "[FATAL] merge 完成但 $merged 内未找到 config.json，请检查 swift export 输出。" 1>&2
        return 1
    fi
    echo "$merged"
}

_MODEL_PATH_ORIG="$MODEL_PATH"
if _resolved=$(_ensure_full_model_dir "$MODEL_PATH"); then
    if [ "$_resolved" != "$MODEL_PATH" ]; then
        echo "[INFO] MODEL_PATH 已切换：$MODEL_PATH -> $_resolved"
        MODEL_PATH="$_resolved"
    fi
else
    echo "[FATAL] 处理 LoRA adapter checkpoint 失败：$_MODEL_PATH_ORIG"
    exit 15
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

# 稳定性 / 恢复参数：仅在环境变量非空时才追加，保证向后兼容
EXTRA_ARGS=()
if [ -n "$BETA" ]; then
    EXTRA_ARGS+=(--beta "$BETA")
fi
if [ -n "$TOP_P" ]; then
    EXTRA_ARGS+=(--top_p "$TOP_P")
fi
if [ -n "$TOP_K" ]; then
    EXTRA_ARGS+=(--top_k "$TOP_K")
fi
if [ -n "$DYNAMIC_SAMPLE" ]; then
    EXTRA_ARGS+=(--dynamic_sample "$DYNAMIC_SAMPLE")
fi
if [ -n "$LOG_ENTROPY" ]; then
    EXTRA_ARGS+=(--log_entropy "$LOG_ENTROPY")
fi
# ---- Sanity check：eval batch × world_size 必须能被 num_generations_eval 整除 ----
# TRL GRPOConfig.__post_init__ 会强制这个断言：
#   ValueError: The global eval batch size (per_device_eval_batch_size *
#               world_size) must be divisible by the number of generations
#               used for evaluation.
# 缺省 num_generations_eval 会跟随 --num_generations（本脚本的 NUM_GENERATIONS）。
# 例如 NUM_GENERATIONS=8 + 4 卡 + EVAL_BATCH_SIZE=1 → 4 %8 != 0 → 直接 crash。
# 此处自动求 gcd(global_eval_batch, NUM_GENERATIONS) 作为兜底 num_generations_eval，
# 显式注入 --num_generations_eval，保证任意 G / EVAL_BATCH_SIZE / NPROC 组合都能启动。
_gcd() {
    local a=$1 b=$2
    while [ "$b" -ne 0 ]; do
        local t=$b
        b=$(( a % b ))
        a=$t
    done
    echo "$a"
}
_global_eval_batch=$(( EVAL_BATCH_SIZE * NPROC_PER_NODE ))
if [ "$_global_eval_batch" -gt 0 ] && [ "$NUM_GENERATIONS" -gt 0 ]; then
    if [ $(( _global_eval_batch % NUM_GENERATIONS )) -ne 0 ]; then
        _num_gen_eval=$(_gcd "$_global_eval_batch" "$NUM_GENERATIONS")
        if [ "$_num_gen_eval" -lt 1 ]; then
            _num_gen_eval=1
        fi
        echo "[WARN] global eval batch ($EVAL_BATCH_SIZE × $NPROC_PER_NODE = $_global_eval_batch) 无法被 num_generations ($NUM_GENERATIONS) 整除。"
        echo "[WARN] 自动降低 num_generations_eval → $_num_gen_eval 以避免 GRPOConfig 断言失败。"
        echo "[WARN] 若想保持 eval G = train G ($NUM_GENERATIONS)，请调整 EVAL_BATCH_SIZE 使 EVAL_BATCH_SIZE × NPROC 能被 $NUM_GENERATIONS 整除。"
        EXTRA_ARGS+=(--num_generations_eval "$_num_gen_eval")
    fi
fi
if [ -n "$MAX_STEPS" ] && [ "$MAX_STEPS" -gt 0 ] 2>/dev/null; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -d "$RESUME_CHECKPOINT" ]; then
        echo "[FATAL] RESUME_CHECKPOINT 目录不存在: $RESUME_CHECKPOINT"
        exit 13
    fi
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
if [ -n "$RESUME_ONLY_MODEL" ]; then
    EXTRA_ARGS+=(--resume_only_model "$RESUME_ONLY_MODEL")
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
    "${EXTRA_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --add_version $ADD_VERSION \
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
