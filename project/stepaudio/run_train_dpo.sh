#!/usr/bin/env bash
# StepAudio2-mini offline DPO 训练脚本（基于 MS-SWIFT）
#
# 决策背景：v4/v6/v9/v10/v12 五次 GRPO 实验（详见 run_train_grpo.sh 头部
# 复盘）无一例外地把 SFT baseline (macro-F1 95.7%) 打崩到 macro-F1 60% 附近。
# 根本原因：base 模型 entropy 长期 <0.02，GRPO 的 rollout 严重塌缩到
# argmax（rare-class 从来不被 sample 到），group std 恒为 0，policy 只
# 从 speech/noise 边界的偶发错误里学习，把 rare-class 边界推歪。
#
# offline DPO 完全绕开 rollout：
#   * 训练数据 = (chosen=正确 label, rejected=错 label) 的成对样本；
#   * loss   = -log σ(β · Δ)，其中 Δ = (logπ_θ − logπ_ref)_chosen −
#              (logπ_θ − logπ_ref)_rejected；
#   * 不需要在训练时采样，因此不受 temperature/entropy 塌缩影响；
#   * ref = SFT ckpt，KL 项自然把 policy 锚定在 SFT 附近做局部纠错。
#
# 数据构造（见 tools/build_dpo_pairs.py）：
#   1) 拿 SFT ckpt 在 train.jsonl 上的推理 jsonl（已存在）；
#   2) 对每条样本：
#        - chosen   = 真实 label；
#        - rejected = SFT 实际预测错的 label（"hard negative"），
#                     若 SFT 已答对则从 confusion prior 采一个近似 hard neg；
#   3) 对 speech/noise 的"已答对样本"下采样（cap=800/类），保留全部
#      wrong-samples（真正的 DPO 学习信号），避免 pair 分布被 majority 样本淹没。
#
# 三步流程：
#   (I) 生成 SFT baseline 的推理结果（如已存在 checkpoint-1600-merged 的
#       推理 jsonl 可跳过）：
#         MODEL_PATH=./output/v16-20260629-162422/checkpoint-1600-merged \
#         VAL_JSONL=project/stepaudio/data/train.jsonl \
#         bash project/stepaudio/run_inference.sh
#       产出：project/stepaudio/infer_results/result_<sft_ckpt>_train.jsonl
#
#   (II) 构造 DPO 训练集（v5：token-flow 均衡 + 只保留 hard neg）：
#         python project/stepaudio/tools/build_dpo_pairs.py \
#             --infer-jsonl <上面的推理 jsonl> \
#             --output project/stepaudio/data/train.dpo.jsonl \
#             --classes speech,music,noise,porn,song \
#             --min-wrong-per-class 200 \
#             --max-audio-sec 90 \
#             --drop-right-samples \
#             --rejected-strategy mistake
#         该命令会：
#           * 丢弃 SFT 已答对的样本（DPO 梯度≈0，反而通过 nll 副损失把梯度
#             拉回多数类，是 v4 崩盘的直接放大器）；
#           * 对每个类的错样本 up-sample 到 200 条，得到 5×200 = 1000 条硬负
#             样本池；
#           * 强制 count(chosen==c) == count(rejected==c) 严格均衡（默认打开
#             --balance-token-flow=1），彻底消除 v4 里 porn/noise 因作为
#             rejected 出现远多于作为 chosen 而被 DPO 无脑压零的问题。
#
#   (III) 启动 DPO：
#         MODEL_PATH=./output/v16-20260629-162422/checkpoint-1600-merged \
#         bash project/stepaudio/run_train_dpo.sh
#
# 关键超参选择依据：
#   * BETA=0.1：DPO 的经典默认；base 已是 SFT 最优附近，太大的 β 会让 loss
#     被 KL 主导而学不动，太小又会 policy 漂离 SFT。0.1 是"一次微调"的稳妥值。
#   * RPO_ALPHA=0.5：TRL 的 RPO 变体，在 chosen 上加一个 SFT loss，进一步
#     锚定 policy 输出正确 label 的 log-prob（对本任务的 rare-class 特别关键：
#     它们只有 ~500 条样本，纯 DPO 容易只学"少出错误 class"而不是"多出正确
#     class"）。
#   * LR=5e-7：与 GRPO v11 同尺度；DPO 对 LR 极敏感，宁小勿大。
#   * NUM_EPOCHS=1 + MAX_STEPS=200：offline pair ≈ 6~8k，1 epoch 足以。
#   * per_device_batch=1 + grad_accum=8：与 GRPO 保持一致，effective batch=32。
#     DPO 每步实际前向 = 2×batch（chosen + rejected），显存约为 SFT 的 2×。
#
# 沿用 GRPO 脚本里所有踩过的坑：
#   - attn_impl eager（step_audio2_mini 唯一支持值）
#   - torch_dtype bfloat16
#   - truncation_strategy delete
#   - PYTORCH_CUDA_ALLOC_CONF expandable_segments
#   - GPU 占用前置检查（避免与其它训练抢卡）
#   - LoRA adapter 目录自动 merge_lora
#   - swift CLI 探测（多个 conda 环境兜底）
#
# 【与 GRPO 脚本的差异】
#   * gradient_checkpointing 默认 true（而不是 GRPO 的 false）：DPO 每步同时前
#     向 chosen + rejected 两份序列，激活显存约为 SFT 的 2×；MAX_LENGTH=2048 +
#     full-tuner + 80G A100 不开 GC 极易 OOM（v1 首跑就在 rank2 上 OOM，见头部
#     复盘）。GRPO 之所以关 GC 是为规避 rollout 兼容性问题，DPO 无此约束。
#     若显存充裕想加速可 GRADIENT_CHECKPOINTING=0 关闭。
#
# 【失败复盘 · v1-20260706-1912】
#   现象：MODEL_PATH=./output/v16-20260629-162422/checkpoint-1600-merged 启动
#   DPO，rank2 在 forward encoder 时 OOM：
#       torch.OutOfMemoryError: Tried to allocate 836.00 MiB.
#       GPU 2 has a total capacity of 79.11 GiB of which 253.38 MiB is free.
#       Process 3208526 has 78.31 GiB memory in use.
#   根因：卡上残留的其它训练进程（PID 3208526）占了 78.31 GiB，本训练只剩
#         ~500 MiB 就爆。脚本前置的 GPU 占用检查用的是
#         `nvidia-smi --query-compute-apps=used_memory`，该字段在容器/共享节点
#         下**只能看到本用户可见的进程**，返回 0 时阈值判断被绕过。
#   修复：
#     1) 前置检查改用 `--query-gpu=memory.used`（整卡真实已用显存，不受用户
#        可见性影响）作为主判据，compute-apps 作为辅助；
#     2) DPO 默认打开 gradient_checkpointing（见上）。
#
# 【失败复盘 · v2-20260707-1523】
#   现象：v1 修复后（GC=on + 前置检查加固）再次 OOM，位置从纯 forward 变成
#         torch.utils.checkpoint 的 recompute 路径（说明 GC 已生效），rank2
#         再次报：
#             Tried to allocate 836.00 MiB.
#             GPU 2 ... 233.38 MiB free.
#             Process 531697 has 78.33 GiB memory in use.
#   根因：Process 531697 不是本训练的进程（自己每个 rank 只用了 ~500 MiB
#         就崩了），是**别的用户/容器在本训练启动之后**抢卡进来的。上一版
#         前置检查是"启动瞬间一次性"，卡在 check→torchrun 拉起→模型加载
#         的几十秒窗口里被别人抢走，等 forward 才发现——太迟。
#   修复（v3）：
#     * 判据从"占用低于上限"改成"空闲高于下限"（GPU_MIN_FREE_MB，默认 60 GiB），
#       更贴近共享节点语义；
#     * 新增等待模式 GPU_WAIT_SECS，不满足时每 30s 复查一次，直到卡空出来或
#       达到超时。使用方式：
#           GPU_WAIT_SECS=1800 bash run_train_dpo.sh   # 最多等 30 分钟
#
# 【失败复盘 · v3-20260707-1531】
#   现象：v3 前置检查通过后成功进入训练，前 2 步 memory=23 GiB，第 3 步跳到
#         76.79 GiB，第 8 步 OOM。调用栈定位在 audio encoder：
#             modeling_step_audio_2.py:350  self.encoder(wavs, wav_lens)
#             modeling_step_audio_2.py:256  utils.checkpoint(block, ...)
#         且 is_ref_model=True（跑参考模型时炸）。
#   根因：数据集里音频时长跨度极大——
#             p50=7.9s  p90=52s  p95=168s  p99=358s  max=593s
#         audio encoder 的 attention 是 O(L²)，DPO 又要跑 4 次前向
#         (policy chosen/rejected + ref chosen/rejected)。抽到 >100s 的样本
#         就会把整卡打满，直到抽到更长的样本就 OOM。
#         注意 `--max_length 2048` 只截 LLM 侧 token 长度，管不到 audio
#         encoder 输入的原始 wav 长度。
#   修复（v4）：
#     * 在 tools/build_dpo_pairs.py 里加 --max-audio-sec（默认 45s）过滤长
#       尾样本，从数据构建阶段一次性消除风险。
#     * 重建 train.dpo.jsonl 的方法见 tools/build_dpo_pairs.py 头注释。
#
# 【失败复盘 · v4-20260707-1544（checkpoint-180 评测）】
#   现象：DPO 顺利跑完 200 步无 OOM，但 val 集评测结果比 SFT baseline 严重
#         倒退——
#             baseline (SFT ckpt-1600)  acc = 95.7%   macro-F1 = 92.4%
#             DPO v4  ckpt-180          acc = 56.5%   macro-F1 = 17.1%
#         逐类看：speech recall 从 ~99% 掉到 80.8%（F1 77.8%）；music/noise/
#         porn/song 的 recall 全部为 0！模型对 rare 三类几乎完全丧失了预测能力，
#         并且开始吐 <tts_end>/<audio_...> 之类的非法 token（2.07% 的样本）。
#   根因（三重叠加）：
#     A) chosen / rejected 的 token 流严重不平衡。旧构造脚本对 majority 类
#        (speech / noise) 做了 --keep-right-per-class=800 down-sample 后，
#        chosen 分布还是 speech≈3060 / noise≈1000 / porn≈220 / song≈160 /
#        music≈310；而 rejected 是根据 confusion prior 采的，porn/noise 作为
#        rejected 出现的次数远超作为 chosen。DPO loss 会系统性地把 porn/
#        noise 的 logits 压到 0，直到它们从 top-1 里彻底消失。
#     B) 保留 SFT 已答对的 4000+ 条样本毫无 DPO 梯度（policy≈ref → Δ≈0），
#        却仍然通过 rpo_alpha=0.5 的 nll_loss 加剧 chosen 分布向 speech 倾斜。
#     C) 长音频虽然被 45s 阈值挡住了 OOM，但把 rare-class（尤其是 song，本来
#        就 <500 样本）的可用数据又砍掉一半，pair 分布进一步倾斜。
#   修复（v5）：
#     * build_dpo_pairs.py 新增 --balance-token-flow（默认开启，v5 语义）：
#       对每个类严格 cap count(rejected==c) ≤ count(chosen==c)，从根源消除
#       token 流不平衡。
#     * build_dpo_pairs.py 新增 --drop-right-samples：丢弃 SFT 已答对样本，
#       让整份 pair 都是"真硬负"。5×min-wrong-per-class=200 → 5×200=1000 条
#       pair，速度反而更快、每步梯度信号更强。
#     * 本脚本 RPO_ALPHA 默认从 0.5 改为 0：既然只留硬负样本，nll 副损失
#       只会把 chosen 分布重新拉回多数类，得不偿失。若发现 chosen log-prob
#       在训练中一直下滑再考虑重新打开（0.1 起）。
#     * 本脚本新增 TRAIN_JSONL 的 balance 自检：读取 head 100 行统计 chosen
#       vs rejected 分布，若发现 |delta|/mean > 0.3 就报错拒绝启动，强制
#       用户重新构造。

set -ex

export LOG_LEVEL=INFO
export WANDB_DISABLED=true
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# ★ DPO 必须从 SFT ckpt 起步（policy 与 reference 都 = SFT）。
# 冷启动 DPO 不收敛（reference 也是 base，chosen/rejected 的初始 logp 差可
# 忽略，无梯度信号）。
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
if [ "$MODEL_PATH" = "/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini" ]; then
    echo ""
    echo "[WARN] ============================================================"
    echo "[WARN] MODEL_PATH 指向的是原始 base 模型（未 SFT）。"
    echo "[WARN] DPO 必须从 SFT 后的 ckpt 起步，冷启动 DPO 无梯度信号。"
    echo "[WARN] 强烈建议："
    echo "[WARN]     MODEL_PATH=./output/v16-20260629-162422/checkpoint-1600-merged \\"
    echo "[WARN]     bash project/stepaudio/run_train_dpo.sh"
    echo "[WARN] 若坚持从 base 冷启动，请显式设置：STRICT_SFT_WARMUP=0"
    echo "[WARN] ============================================================"
    if [ "${STRICT_SFT_WARMUP:-1}" = "1" ]; then
        echo "[FATAL] 拒绝冷启动 DPO（STRICT_SFT_WARMUP=1，默认）。"
        exit 16
    fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/stepaudio/dpo"}
# ms-swift 默认 add_version=True, 会在 OUTPUT_DIR 后自动追加 v<idx>-<timestamp>
# 子目录 (见 swift/arguments/sft_args.py 中 _add_version)。设 ADD_VERSION=false
# 可让 checkpoints 直接落到 $OUTPUT_DIR 下, 便于外部脚本按固定路径消费。
# 注意：关闭 add_version 后，若同一 OUTPUT_DIR 重复启动训练会覆盖之前的
# checkpoints/logs, 请自行确认是否是想要的行为。
ADD_VERSION=${ADD_VERSION:-true}

# ---------------- Tuner ----------------
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=1e-7   # DPO + full 必须比 lora 再小 1 个数量级
else
    # lora + DPO 的经典范围 5e-7 ~ 1e-6；base 已很强，取下限。
    DEFAULT_LR=5e-7
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（沿用 GRPO 脚本）
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- DPO 关键超参 ----------------
# BETA: DPO 的 KL 系数。0.1 = 经典默认；base 强场景可小幅上调 0.15~0.2 增强
# 拘束（避免 policy 漂离 SFT），但不要 <0.05 以免 loss 变成纯 SFT。
BETA=${BETA:-0.1}
# LOSS_TYPE: sigmoid（原始 DPO） / ipo / hinge / robust / apo_zero / apo_down /
# discopop / sft。分类任务上 sigmoid 稳定；如观察到过拟合可试 ipo（不用 β 缩放）。
LOSS_TYPE=${LOSS_TYPE:-sigmoid}
# LABEL_SMOOTHING: cDPO 的 conservative label smoothing，0.0 = 关闭；数据里
# 若有噪声（rejected 其实也可能对）可开 0.1 缓解。
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
# RPO_ALPHA: 在 chosen 上叠加 SFT NLL loss。0 = 纯 DPO；>0 就是 RPO。
# 【v5 语义变更】默认从 0.5 改为 0：v5 数据已丢弃 SFT-right 样本，全部都是
# 硬负样本，DPO 梯度信号本身就很集中；再加 nll_loss 只会把 chosen 分布重新
# 拉回 speech/noise 多数类（v4 塌缩的一个直接放大器）。若观察到 chosen
# log-prob 训练过程持续下滑，才考虑调回 0.1 起步。
RPO_ALPHA=${RPO_ALPHA:-0}
# 参考模型：为空 → 自动使用初始 policy 的深拷贝作为 ref。
# 若 MODEL_PATH 是 SFT-merged full 权重，这就是我们想要的 ref。
REF_MODEL=${REF_MODEL:-}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---------------- 训练规模 ----------------
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
# offline pair 通常 6k~8k；MAX_STEPS 200 = ~800 samples effective，够 1 epoch
# 里最有信号的部分。若 pair 更多可放大。
MAX_STEPS=${MAX_STEPS:-200}

SAVE_STEPS=${SAVE_STEPS:-20}
EVAL_STEPS=${EVAL_STEPS:-20}
save_total_limit=${save_total_limit:-10}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---------------- 序列 / 截断 ----------------
MAX_LENGTH=${MAX_LENGTH:-2048}
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

# ---------------- 显存优化 ----------------
# DPO 每步实际前向 = 2×batch（chosen + rejected），激活显存约为 SFT 的 2 倍。
# 默认打开 gradient_checkpointing 以避免 OOM；若显存富余可置 0 换取 ~1.3× 训练
# 加速。GRPO 脚本关 GC 是为规避 rollout 兼容性问题，DPO 无此约束。
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
case "$GRADIENT_CHECKPOINTING" in
    1|true|True|TRUE) GC_FLAG=true ;;
    0|false|False|FALSE) GC_FLAG=false ;;
    *) GC_FLAG="$GRADIENT_CHECKPOINTING" ;;
esac

# ---------------- DeepSpeed ----------------
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

# ---------------- 数据集 ----------------
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.dpo.jsonl"}
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[FATAL] TRAIN_JSONL 不存在: $TRAIN_JSONL"
    echo ""
    echo "        DPO 训练前必须先构造成对样本。请按以下步骤生成："
    echo ""
    echo "        1) 用 SFT ckpt 在 train.jsonl 上做推理（若已有可跳过）:"
    echo "             MODEL_PATH=<sft_merged_ckpt> \\"
    echo "             VAL_JSONL=$DATA_DIR/train.jsonl \\"
    echo "             bash project/stepaudio/run_inference.sh"
    echo ""
    echo "        2) 用推理 jsonl 构造 DPO pair:"
    echo "             python $SCRIPT_DIR/tools/build_dpo_pairs.py \\"
    echo "                 --infer-jsonl <上一步产出的 result_*.jsonl> \\"
    echo "                 --output $TRAIN_JSONL \\"
    echo "                 --classes speech,music,noise,porn,song \\"
    echo "                 --keep-right-per-class 800 \\"
    echo "                 --min-wrong-per-class 200 \\"
    echo "                 --rejected-strategy mistake"
    echo ""
    echo "        3) 重新执行本脚本。"
    exit 14
fi
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- token-flow balance 自检（v5） ----------------
# 上一轮 v4 训练塌缩的根因是 chosen / rejected 的 token 分布严重不均衡：
# 某些类作为 rejected 出现的次数远多于作为 chosen 出现的次数，DPO 会把它们
# 无脑压到 top-1 消失（porn/noise 在 v4 就这样归零）。build_dpo_pairs.py v5
# 默认打开 --balance-token-flow，但为了防止有人误用旧数据集重启训练，这里
# 再做一道运行时保险：读取整份 TRAIN_JSONL，统计 chosen/rejected 每类计数，
# 若最大 |delta|/mean > BALANCE_TOL（默认 0.30）就拒绝启动。
BALANCE_TOL=${BALANCE_TOL:-0.30}
BALANCE_CHECK=${BALANCE_CHECK:-1}
if [ "$BALANCE_CHECK" = "1" ]; then
    _balance_msg=$(python3 - "$TRAIN_JSONL" "$BALANCE_TOL" <<'PY'
import json, sys, collections
path, tol = sys.argv[1], float(sys.argv[2])
ch = collections.Counter(); rj = collections.Counter()
n = 0
with open(path, 'r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        n += 1
        gt = r.get('label')
        if gt is None:
            msgs = r.get('messages') or []
            asst = [m for m in msgs if m.get('role') == 'assistant']
            gt = asst[-1].get('content') if asst else None
        rej = r.get('rejected_response')
        if gt: ch[gt] += 1
        if rej: rj[rej] += 1
classes = sorted(set(ch) | set(rj))
worst_delta = 0.0
worst_c = None
print(f'[BAL] n={n}')
print(f'[BAL] {"class":<10}  {"chosen":>7}  {"rejected":>8}  {"delta":>6}')
for c in classes:
    a, b = ch.get(c, 0), rj.get(c, 0)
    mean = (a + b) / 2.0 if (a + b) else 1.0
    d = abs(a - b) / mean
    if d > worst_delta:
        worst_delta = d; worst_c = c
    print(f'[BAL] {c:<10}  {a:>7d}  {b:>8d}  {b - a:>+6d}')
print(f'[BAL] worst |delta|/mean = {worst_delta:.3f}  (class={worst_c})  tol={tol:.3f}')
sys.exit(0 if worst_delta <= tol else 42)
PY
)
    _balance_rc=$?
    printf '%s\n' "$_balance_msg"
    if [ "$_balance_rc" != "0" ]; then
        echo ""
        echo "[FATAL] TRAIN_JSONL 未通过 token-flow balance 自检 (worst delta > $BALANCE_TOL)。"
        echo "        该数据集大概率是 v4 及以前构造的旧版本，会导致 DPO 塌缩。"
        echo "        请用 v5 构造命令重新生成："
        echo "            python $SCRIPT_DIR/tools/build_dpo_pairs.py \\"
        echo "                --infer-jsonl <SFT 在 train.jsonl 上的推理 jsonl> \\"
        echo "                --output $TRAIN_JSONL \\"
        echo "                --classes speech,music,noise,porn,song \\"
        echo "                --min-wrong-per-class 200 \\"
        echo "                --max-audio-sec 90 \\"
        echo "                --drop-right-samples"
        echo "        若确认要跳过此检查（不推荐），可: BALANCE_CHECK=0 bash \$0"
        exit 17
    fi
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
echo "[INFO] gradient_checkpointing=$GC_FLAG"
echo "[INFO] DPO beta=$BETA loss_type=$LOSS_TYPE label_smoothing=$LABEL_SMOOTHING rpo_alpha=$RPO_ALPHA"
echo "[INFO] MAX_STEPS=$MAX_STEPS  SAVE_STEPS=$SAVE_STEPS  EVAL_STEPS=$EVAL_STEPS"

# ---------------- GPU 占用前置检查 ----------------
# 【判据改造 · v3】
# 共享/容器节点场景下更合理的判据不是"已用 < 上限"（阈值判上限），
# 而是"空闲 >= 下限"（阈值判下限）。参数：
#   - GPU_MIN_FREE_MB : 每张卡至少需要多少 MiB 空闲显存，默认 60000 (~60 GiB)
#                       依据：MAX_LENGTH=2048 + DPO(2×前向) + GC=on + zero2/3
#                       典型峰值 45~60 GiB；给到 60 GiB 留 20% 缓冲。
#   - GPU_WAIT_SECS   : 若不满足，最长等待多少秒（每 30s 复查一次）。
#                       默认 0 = 不等待直接失败；设置为 1800 表示最多等 30 分钟。
#   - GPU_PREALLOC_SKIP / FORCE : 显式跳过所有检查。
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] 当前 GPU 占用情况："
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader || true

    GPU_MIN_FREE_MB=${GPU_MIN_FREE_MB:-60000}
    GPU_WAIT_SECS=${GPU_WAIT_SECS:-0}
    GPU_POLL_INTERVAL=${GPU_POLL_INTERVAL:-30}
    GPU_PREALLOC_SKIP=${GPU_PREALLOC_SKIP:-${FORCE:-0}}

    if [ "$GPU_PREALLOC_SKIP" != "1" ]; then
        IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        _waited=0
        while :; do
            _busy=0
            _busy_report=""
            for _gid in "${_GPU_IDS[@]}"; do
                # memory.free 是驱动侧统计，不受用户可见性限制，能反映
                # "别的用户/容器"占用带来的真实剩余显存。
                _free_mb=$(nvidia-smi --id="$_gid" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{print $1+0}')
                _used_mb=$(nvidia-smi --id="$_gid" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1+0}')
                if [ "${_free_mb:-0}" -lt "$GPU_MIN_FREE_MB" ]; then
                    _busy=1
                    _busy_report="${_busy_report}    GPU ${_gid}: free=${_free_mb} MiB used=${_used_mb} MiB (需 free>=${GPU_MIN_FREE_MB} MiB)\n"
                fi
            done

            if [ "$_busy" = "0" ]; then
                echo "[INFO] GPU 占用检查通过 (每卡 free >= ${GPU_MIN_FREE_MB} MiB)"
                break
            fi

            echo "[WARN] 部分 GPU 空闲显存不足："
            printf "$_busy_report"

            if [ "$_waited" -ge "$GPU_WAIT_SECS" ]; then
                echo ""
                echo "[FATAL] 等待 ${_waited}s 后仍无足够空闲显存 (GPU_WAIT_SECS=${GPU_WAIT_SECS})。"
                echo "        处置建议："
                echo "          1) 如果占卡的是自己旧的训练进程："
                echo "               pgrep -af '(swift|torchrun|rlhf|deepspeed)' | awk '{print \$1}' | xargs -r kill -9"
                echo "          2) 显式换空闲卡：export CUDA_VISIBLE_DEVICES=<free_ids>"
                echo "          3) 让脚本自己等（例如最多等 30 分钟）："
                echo "               GPU_WAIT_SECS=1800 bash $(basename "$0")"
                echo "          4) 降低门槛（谨慎，可能后续 OOM）："
                echo "               GPU_MIN_FREE_MB=40000 bash $(basename "$0")"
                echo "          5) 强制跳过检查（最不推荐）：FORCE=1 bash $(basename "$0")"
                exit 11
            fi

            echo "[INFO] 已等待 ${_waited}s / ${GPU_WAIT_SECS}s，${GPU_POLL_INTERVAL}s 后重试..."
            sleep "$GPU_POLL_INTERVAL"
            _waited=$(( _waited + GPU_POLL_INTERVAL ))
        done
    else
        echo "[WARN] 已通过 GPU_PREALLOC_SKIP=1 / FORCE=1 跳过 GPU 占用检查。"
    fi
fi

# ---------------- swift 入口探测（与 GRPO 脚本一致） ----------------
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
        SWIFT_CMD=(swift)
    else
        SWIFT_CMD=(python -m swift.cli.main)
    fi
fi

# ---------------- LoRA adapter checkpoint 自动 merge（与 GRPO 脚本一致） ----------------
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
    if ! "${SWIFT_CMD[@]}" export \
            --adapters "$mdir_abs" \
            --merge_lora true \
            --output_dir "$merged" 1>&2; then
        echo "[FATAL] swift export --merge_lora 失败" 1>&2
        return 1
    fi
    if [ ! -f "$merged/config.json" ]; then
        echo "[FATAL] merge 完成但 $merged 内未找到 config.json" 1>&2
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

EXTRA_ARGS=()
if [ -n "$REF_MODEL" ]; then
    EXTRA_ARGS+=(--ref_model "$REF_MODEL")
fi
if [ -n "$RPO_ALPHA" ]; then
    EXTRA_ARGS+=(--rpo_alpha "$RPO_ALPHA")
fi
if [ -n "$MAX_STEPS" ] && [ "$MAX_STEPS" -gt 0 ] 2>/dev/null; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi

# ---------------- 启动 DPO 训练 ----------------
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" rlhf \
    --rlhf_type dpo \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    "${TUNER_ARGS[@]}" \
    "${DS_ARGS[@]}" \
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
    --truncation_strategy $TRUNCATION_STRATEGY \
    --beta $BETA \
    --loss_type $LOSS_TYPE \
    --label_smoothing $LABEL_SMOOTHING \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_checkpointing $GC_FLAG \
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
    --seed 42 \
    "$@"

echo "[INFO] DPO 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
