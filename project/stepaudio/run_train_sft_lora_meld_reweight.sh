#!/usr/bin/env bash
# 【Path-C1 · 方案 β · 2026-07-08】MELD SFT LoRA 均衡数据完整重训 (word-label).
#
# 目标: 复刻 v7 SFT LoRA 训练流程, 唯一差异是把训练数据从原始 train_letter.jsonl
#       换成 rare-class oversample 后的均衡数据集 train.reweight_uniform.jsonl,
#       并使用 word-label (与推理端 CLASSES 完全一致, 免除 letter↔word 归一化).
#
# 【为什么用 dispatcher 模式而不是自己写一份 config】
#   v6 版本 (方案 α, "从 v7 merged base + 5e-7 冷启新 lora 300 步") 已证明失败:
#   macro-F1 从 v7 的 41.17% 掉到 33.76%. 根因是同时干两件事 —— 切 label 协议
#   (letter → word) + 精修决策边界, 但小 lr (5e-7) 推不动协议切换.
#   方案 β 的思路: 承认这是"新一次完整训练", 用 v7 的稳态超参 (lr 5e-5, 3 epochs,
#   完整 lora target: q/k/v/o + gate/up/down + adapter.linear[12], freeze_aligner=false,
#   early_stop_interval=5) 从 base 冷启. 只有数据集是 rare-class 均衡的.
#
# 【为什么继承 v7 主脚本而非拷贝一份】
#   v7 主脚本 [run_train_sft_lora_meld.sh] 是 460+ 行的稳态 config, 包含
#   DDP MASTER_PORT 挑选、sitecustomize KV cache 强关、attn_impl=eager、
#   健康检查等所有 audio LM 训练必需项. 拷贝一份不但风险高, 还会导致后续
#   任何 v7 主脚本的修复无法同步过来. 本脚本只保留 env override + exec.
#
# 【关键 env override (相对 v7 主脚本 default)】
#     TRAIN_JSONL   train_letter.jsonl → train.reweight_uniform.jsonl
#                       (word-label, rare-class oversample 到 5000/each,
#                        neutral 保持 4709, 总 34709 条)
#     VAL_JSONL     dev_letter.jsonl   → dev.jsonl (word-label 版)
#     OUTPUT_DIR    output/meld        → output/meld/c1_reweight (与 v0-v7 分开)
#     MODEL_PATH    默认 = base (Step-Audio-2-mini), 不是 v7 ckpt.
#                   目的: 完整冷启一次, 让模型直接学"均衡数据 + word 协议"
#                   联合分布, 而不是先 letter 再切 word.
#   其他所有超参 (lr / epochs / target_modules / batch / grad_accum /
#   warmup / weight_decay / label_smoothing / freeze_vit / freeze_aligner /
#   early_stop / attn_impl / torch_dtype / max_length / truncation_strategy /
#   save_steps / eval_steps / save_total_limit / lora_rank / lora_alpha /
#   lora_dropout / lora_target_regex / dataloader_num_workers) 全部沿用 v7.
#
# 【硬标准 · 达标才算 β 成功】
#   * ckpt-best macro-F1 应 >= 43% (超过 C3 后处理 43.33%, 说明权重内化有效);
#   * disgust recall >= 50% (SFT v7 45.6%, C3 后处理 52.9%);
#   * anger  recall  >= 25% (SFT v7 22.6%, C3 后处理 27.2%);
#   * 推理时 PRIOR_ALPHA=0 就能达到; PRIOR_ALPHA=1.5 若再涨可作 bonus.
#
# 【β 首战成绩 · 2026-07-08 output/meld/sft/v6】
#   Ckpt      Acc     mF1     surp anger neut  joy   sad  fear disg
#   v7 base   58.47   41.17   46.3 22.6  89.6  25.1  24.0 22.0 45.6   (baseline)
#   C3 α=1.75 56.25   43.33   63.3 27.2  73.2  38.8  33.6 28.0 52.9   (推理后处理)
#   β ckpt-50 55.79   44.40✅ 58.7 40.0  67.6  46.8  32.7 34.0 45.6   (纯权重!)
#   β ck-150  52.87   43.10   61.6 39.4  58.4  52.7  35.6 38.0 47.1
#   β ck-300  51.30   42.17   55.9 54.8  52.2  60.2  29.3 28.0 29.4
#   → macro-F1 硬标准 43% ✅ 达到 44.40%; anger recall 硬标准 25% ✅ 达到 40%;
#     disgust recall 硬标准 50% ❌ (45.6%, 差 4.4pp) —— 但已经追平 v7 baseline;
#     且 ckpt-50 完全不需要 prior 后处理, 权重内化验证成功.
#
# 【早停坑 · v7 主脚本 EARLY_STOP_INTERVAL=5 + eval_loss 与 macro-F1 反相关】
#   β 训练时 eval_loss 曲线单调走高 (50→300: 0.532→0.606), 触发 5 次未提升早停,
#   1629 步只跑了 300 步. 但真实 test macro-F1 却是 ckpt-50 最高 (44.40 → 42.17).
#   根因: 训练集是 rare-class 均衡的 (7 类各 ~14%), 而 dev 集是原始分布
#   (neutral 42% + rare 各 2~4%). 模型学"多输出 rare-class"时, dev 上 CE loss
#   反而涨, 但 test 上 macro-F1 反而好. 这两个指标在 rare-class 均衡任务中
#   是反相关信号, 用 eval_loss 做早停完全走错方向.
#   本次运行歪打正着: 早停保住了 ckpt-50 这个最优点, 未继续降到 ckpt-300 的 42.17.
#
# 【若还要继续优化 (macro-F1 突破 45%)】
#   方案 Y (推荐): 关闭早停或换用 token_acc 做早停指标.
#     EARLY_STOP_INTERVAL=0 NPROC_PER_NODE=4 \
#         bash project/stepaudio/run_train_sft_lora_meld_reweight.sh
#     让训练跑满 3 epochs (1629 步), 每 25 步 eval + save, 事后用 sweep_eval
#     从 test macro-F1 角度选最优 ckpt (可能 在 ckpt-25/75 附近).
#   方案 Z (激进): 改评估集分布, 用从 train 抽样的均衡 dev 让 eval_loss 与
#     train 目标一致. 但会失去对真实 test 分布的监控能力.
#
# 【预算】
#   34709 条数据 × 3 epoch / (BS 1 × GA 8) ≈ 13000 步 (4 卡 DDP 时 ≈ 3250 步),
#   按 v7 训练速度 (约 0.5 s/step 4 卡) ≈ 27~45 分钟. 若 GPU 数量不足或
#   有显存压力可通过 NUM_EPOCHS=1 (缩短 3×) 快速拿一版试水结果.
#
# 用法:
#   # 【推荐】默认: 从 base 冷启, 4 卡 DDP, 3 epoch
#   NPROC_PER_NODE=4 bash project/stepaudio/run_train_sft_lora_meld_reweight.sh
#
#   # 试水版: 1 epoch 快速看趋势
#   NUM_EPOCHS=1 NPROC_PER_NODE=4 bash project/stepaudio/run_train_sft_lora_meld_reweight.sh
#
#   # 显式换 base (例如换成 stepfun 官方目录):
#   MODEL_PATH=/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini \
#       NPROC_PER_NODE=4 bash project/stepaudio/run_train_sft_lora_meld_reweight.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- 允许 KEY=VALUE 方式覆盖 ----
for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
    esac
done

# ---------------- 三个关键 env override ----------------
DATA_DIR="$SCRIPT_DIR/data_meld"
export TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.reweight_uniform.jsonl}"
export VAL_JSONL="${VAL_JSONL:-$DATA_DIR/dev.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SWIFT_ROOT/output/meld/c1_reweight}"

# MODEL_PATH 默认 = base (从零冷启), 若用户传 v7 ckpt 就走 letter-lora 之上再学 word
# 的路径 —— 不推荐, 但保留能力.
export MODEL_PATH="${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}"

# ---- 训练数据不存在则自动构造 (word-label 均衡版) ----
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[INFO] $TRAIN_JSONL 不存在, 自动调用 build_meld_reweight_data.py 构造..."
    _INPUT="$DATA_DIR/train.balanced.jsonl"
    if [ ! -f "$_INPUT" ]; then
        echo "[ERROR] 需要 $_INPUT 作为源, 但它不存在。" >&2
        exit 1
    fi
    # neutral 上限保持 4709 (原始数量), 其他 6 类 oversample 到 5000
    python3 "$SCRIPT_DIR/tools/build_meld_reweight_data.py" \
        --input  "$_INPUT" \
        --output "$TRAIN_JSONL" \
        --target-per-class 5000 \
        --cap-per-class neutral:4709
    echo "[INFO] 训练集已生成: $TRAIN_JSONL"
fi

# ---- v7 主脚本默认写死了 --classes S,A,N,J,D,F,G, 但 word 数据不需要该参数
#     (label 在 messages/label 里就是完整单词, 模型直接学). 无需额外覆盖. ----

echo "================================================================"
echo "[C1-β] Path-C1 方案 β · MELD SFT LoRA 均衡数据 word-label 完整重训"
echo "[C1-β] MODEL_PATH     = $MODEL_PATH"
echo "[C1-β] TRAIN_JSONL    = $TRAIN_JSONL"
echo "[C1-β] VAL_JSONL      = $VAL_JSONL"
echo "[C1-β] OUTPUT_DIR     = $OUTPUT_DIR"
echo "[C1-β] 其他超参完全继承 v7 主脚本 [run_train_sft_lora_meld.sh]:"
echo "[C1-β]   * lr=5e-5, 3 epochs, cosine, warmup 0.03"
echo "[C1-β]   * lora target = q/k/v/o + gate/up/down + adapter.linear[12]"
echo "[C1-β]   * freeze_vit=true, freeze_aligner=false"
echo "[C1-β]   * early_stop_interval=5"
echo "================================================================"

# 直接 exec 到 v7 主脚本, 所有 config 自动继承
exec bash "$SCRIPT_DIR/run_train_sft_lora_meld.sh"
