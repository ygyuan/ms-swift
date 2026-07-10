#!/usr/bin/env bash
# StepAudio2-mini OPSD（On-Policy Self-Distillation）后训练脚本 —— MELD 情感 7 分类版（基于 MS-SWIFT）
#
# ════════════════════════════════════════════════════════════════════════════
# 【任务差异 · 相对 ASC 5 分类的 run_train_opsd.sh】
# ════════════════════════════════════════════════════════════════════════════
# 类别 (7):     surprise, anger, neutral, joy, sadness, fear, disgust
# 数据目录:     project/stepaudio/data_meld/  (word 版, assistant label 是完
#               整单词; letter 版已被 SFT 脚本头部复盘证明会塌陷)
# 数据分布:     neutral 47.2% / joy 17.4% / anger 12.1% / surprise 11.9% /
#               sadness 7.2% / disgust 2.7% / fear 2.7%    (train 9989 条)
# SFT 起点:     MELD full SFT (v0-20260707-182357/checkpoint-234) test
#               acc=59.8%, macro-F1=43.0%; LoRA v7 checkpoint-150 acc=58.5%,
#               macro-F1=41.2%. 相比 ASC (SFT 已经 97.83%) MELD SFT 起点
#               弱得多, 但这也意味着教师能提供有价值的软分布信号.
# OPSD EV:      与 ASC 完全不同：ASC 上的 OPSD (v13/v14/v15) 都在行为
#               "从 97.8% 降到 50%" 的负向方向上抖, 因为教师本身已处
#               于最优点附近, hint / soft target / JSD 带来的都是噪声.
#               MELD 是相反的【正 EV】场景：
#                 - SFT 已学会情感分类的大方向 (>50% acc)，但造会现象
#                   严重 (neutral over-prediction: recall 84%, joy/sadness/
#                   fear 被大量预测为 neutral)；
#                 - 教师在看到 GT hint 后可以给出 "真正的情感 rank", 包含
#                   比 SFT one-hot label 更丰富的类间相似度信号 (例如
#                   surprise vs joy 相似, fear vs sadness 相似)；
#                 - 这些相似度矩阵正是 MELD 需要学的 "情感中组结构".
#
# 与 ASC 脚本的关键差异：
#   * MODEL_PATH 默认 = MELD SFT 后 ckpt (需用户刷新为自己的 run)；
#   * OUTPUT_DIR / DATA_DIR / TRAIN_JSONL 全部指向 data_meld / output/meld；
#   * TARGET_CLASS 从 "porn" 换成 "fear" (MELD 的美回归任务: recall 只有 20%)；
#   * OPSD_HINT_MODE 默认维持 rank 多选题式 (适合 7 类, 避免 direct hint
#     下教师软分布 ≈ one-hot(gt) 的退化问题)；
#   * LR 从 5e-7 抬到 1e-6（MELD 起点弱允许更大步长）；
#   * MAX_STEPS 从 1000 抬到 1500 (rare-class 需要多 epoch 学习)；
#   * SFT_ALPHA 从 2.0 降到 1.0（MELD one-hot NLL 会把分布拉回 neutral, 需要
#     释放更多权重给 JSD 去学习情感中组结构）；
#   * SYSTEM 强制注入 "You are a helpful assistant." (对齐 UltraEval-Audio
#     baseline; SFT / 推理 / OPSD 三方一致).
# ════════════════════════════════════════════════════════════════════════════
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
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# OPSD 既可"冷启动"（直接拿原始 mini 同时当教师+学生），也可在 SFT 后做精炼。
# 【重要】base 模型冷启动 OPSD 极易崩坏（教师=学生初始，两者都不会做分类，
#         KL 目标退化，学生很快漂移到 <audio_XXX> 音频码本 token 分支）。
#         *强烈建议* MODEL_PATH 指向已跑过 SFT 的 checkpoint（例如
#         $SWIFT_ROOT/output/stepaudio/sft/vX-.../checkpoint-XXX），
#         再用 OPSD 做软标签精修。
#
# 【v15 事故复盘 · MODEL_PATH 默认值调整】v15-20260702-150039 用了 checkpoint-400-merged
# 作为起点，训练 200 步后 checkpoint-20/40/80/200 accuracy 全部只有 ~57%，noise/porn F1=0，
# 且有 156~162 条 <tts_end>/<audio_XXXX> 无效响应。回溯发现根因是：
#   1) checkpoint-400 (adapter) 本身推理 accuracy=97.48%，是干净的 SFT ckpt；
#   2) 但 v15 脚本 _ensure_full_model_dir 用当前 transformers 环境自动 merge 出的
#      checkpoint-400-merged 目录**缺失 tokenizer 关键文件**（added_tokens.json /
#      merges.txt / special_tokens_map.json / vocab.json 都缺），config.json 里
#      transformers_version=5.9.0（异常），加载后 lm_head 对 label token 失效。
#   3) 而更早、正常状态下产出的 checkpoint-1600-merged 目录 tokenizer 完整，推理
#      accuracy=97.83% (noise F1=0.952 / porn F1=0.952)，是稳定可用的 OPSD 起点。
# 因此默认值调整为 checkpoint-1600-merged。**禁止再使用 checkpoint-400-merged**——
# 若发现旧目录仍存在，建议手动 rm -rf 后重跑。
MODEL_PATH=${MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
if [ "$MODEL_PATH" = "/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini" ]; then
    echo ""
    echo "[WARN] ============================================================"
    echo "[WARN] MODEL_PATH 指向的是原始 base 模型（未 MELD SFT）。"
    echo "[WARN] MELD OPSD 必须从 SFT 后的 ckpt 起步（冷启动 → 教师 == 学生 == base,"
    echo "[WARN]   无知识可蒸, JSD 退化为噪声, 会重现 ASC v13/v14 塌崩曲线）。"
    echo "[WARN] 推荐候选："
    echo "[WARN]   * MELD full SFT : ./output/meld/v0-20260707-182357/checkpoint-234"
    echo "[WARN]   * MELD LoRA SFT : ./output/meld/v7-20260707-162917/checkpoint-150 (自动 merge)"
    echo "[WARN] 例:  MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234 bash \$0"
    echo "[WARN] 若坚持从 base 冷启动，请显式设置：STRICT_SFT_WARMUP=0"
    echo "[WARN] ============================================================"
    if [ "${STRICT_SFT_WARMUP:-1}" = "1" ]; then
        echo "[FATAL] 拒绝冷启动 OPSD（STRICT_SFT_WARMUP=1，默认）。请指定 MELD SFT 后的 MODEL_PATH。"
        exit 16
    fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/meld/opsd/v3"}
# ms-swift 默认 add_version=True, 会在 OUTPUT_DIR 后自动追加 v<idx>-<timestamp>
# 子目录 (见 swift/arguments/sft_args.py 中 _add_version)。设 ADD_VERSION=false
# 可让 checkpoints 直接落到 $OUTPUT_DIR 下, 便于外部脚本按固定路径消费。
# 注意：关闭 add_version 后，若同一 OUTPUT_DIR 重复启动训练会覆盖之前的
# checkpoints/logs, 请自行确认是否是想要的行为。
ADD_VERSION=${ADD_VERSION:-false}

# 【MELD 关键 · SYSTEM prompt · 对齐 UltraEval-Audio baseline】
# UltraEval-Audio 官方推理器 (audio_evals/models/step_audio_2_mini.py:180) 强制
# 注入 system="You are a helpful assistant.". SFT / 推理 / OPSD 必须一致,
# 否则 policy 分布会因 prompt 前缀变化而漂移, 二次塌缩。
SYSTEM=${SYSTEM:-"You are a helpful assistant."}

# ---------------- 冻结策略（借鉴 run_train_sft_full_meld.sh v2 · 防类别塌陷）----------------
# 【SFT-full v0/v1/v6 事故教训 · MELD letter 版】
# 未冻结 embedding + lm_head 时, 即使 loss 只对 assistant 那 1 个分类 token 计梯度,
# 通过 embed / lm_head 反向传播依然会扰动整个词表分布, 400-800 步内 argmax 就塌陷到
# 多数类 (v0-full ckpt-50 → 99.9% 预测 neutral, acc=48%).
# OPSD 场景更危险：sft_alpha=3.0 让 CE 权重更高, 且 on-policy generate 每步都用当前
# 权重采样, 一旦 embed/lm_head 被推歪, 学生下一步就吐更歪的 completion, 教师被污染,
# 迅速形成正反馈崩溃。因此 OPSD 默认冻结 embed+lm_head, 让学习集中在 attention layers。
#
# 注意: MELD 词汇 (surprise/anger/neutral/joy/sadness/fear/disgust) 都是普通英文 token,
# 冻结 embedding 只是不让"这几个 token 的向量"被推来推去, 不影响 LLM 中间层学习。
# 【路径 A6 追加】默认 FREEZE_EMBED_LMHEAD=true; 若观察到严重欠拟合可 =false 手动打开。
FREEZE_EMBED_LMHEAD=${FREEZE_EMBED_LMHEAD:-true}
# 可追加冻结前若干 transformer 层进一步稳定 (SFT-full 建议 model.layers.0-3)。
# 默认空, 依靠 embed+lm_head 冻结即可。
EXTRA_FREEZE_PREFIXES=${EXTRA_FREEZE_PREFIXES:-""}
# vit (audio encoder) 冻结: 32 层 conformer 参数量大, 全参训练不稳且吃显存;
# OPSD 教师+学生双 forward 场景下更应冻结 vit 节省显存。
FREEZE_VIT=${FREEZE_VIT:-true}
# aligner (adapter, ~10M 参数) 是 audio → LLM 的关键桥梁, 保持可训, OPSD 才能
# 真正把音频情感语义蒸馏进桥接层; SFT-full 也是 aligner=false。
FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
# LLM 主体默认可训。LoRA 模式下 freeze_llm=false 但只有 lora 参数会更新, 语义无冲突。
FREEZE_LLM=${FREEZE_LLM:-false}

# ---------------- label_smoothing (借鉴 SFT-full 关键 bug 说明) ----------------
# 【swift + step_audio2 组合的 label_smoother shift bug 说明】
# swift/trainers/seq2seq_trainer.py 里对 label_smoother 的调用:
#   if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#       loss = self.label_smoother(outputs, labels, shift_labels=True)
#   else:
#       loss = self.label_smoother(outputs, labels)   # ← shift_labels=False!
# 'StepAudio2ForCausalLM' 不在 transformers 映射表, 走 else 分支 → logits 与 labels
# 未 shift 对齐, train_loss 可飙到 ~52 (SFT-full v4/v5 LoRA 实测)。
# OPSD 走 GKD trainer, 目前观察未复现该 bug (loss ~1.1 正常), 但为一致性显式设为 0.0,
# 避免"未来某次实验加了 smoothing 才触发"的隐患。
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}

# ---------------- OPSD 教师模式 ----------------
# OPSD_TEACHER_MODE:
#   dynamic       : 不传 --teacher_model；rlhf_args 会保持 teacher_model=None，
#                   但 *不会* 设置 _teacher_use_disable_adapter=True。
#                   实测结果是 gkd_trainer 中教师 forward 走 nullcontext()——
#                   直接使用启用了 LoRA 的学生本体，教师 == 学生，
#                   |student_logits - teacher_logits| ≡ 0，JSD ≡ 0，训练无梯度。
#                   ⚠️ 该模式在当前 swift 版本下等同于禁用 OPSD，不要使用。
#   fixed (默认)  : 传 --teacher_model = $MODEL_PATH。当 tuner_type=lora 且
#                   teacher_model==student model 时，rlhf_args 会规约成
#                   `_teacher_use_disable_adapter=True` + teacher_model=None：
#                   共用同一份 base 权重（无额外显存），但教师 forward 走
#                   disable_adapter()，与学生 LoRA 天然分离。这才是 OPSD 论文
#                   与官方示例真正生效的路径。
OPSD_TEACHER_MODE=${OPSD_TEACHER_MODE:-fixed}

# ---------------- Tuner ----------------
# OPSD 显存消耗 ≈ 学生 forward + 教师 forward + 学生 backward。
# dynamic 自蒸馏模式下：LoRA 训练时教师走 disable_adapter() 共享 base 权重，
# 不会真正多加载一份模型；这是 OPSD 与普通 GKD 最大的实用差异，强烈建议 LoRA。
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    # full + OPSD：dynamic 模式下教师=学生当前权重，教师 forward 不需要额外参数副本，
    # 但需要禁用 dropout/grad；fixed 模式下需要再加载一份完整 base 模型，显存翻倍，
    # 强烈建议配合 ZeRO-3。LR 也得给得很小，否则 KL 容易直接发散。
    DEFAULT_LR=1e-6
else
    # 【v13 事故复盘】v13-20260701-183236 (LoRA + lr=5e-6 + guided hint + beta=0.1)
    # 训练 4000 步后从 SFT 起点 97.83% 崩到 50.13%（song 类 precision=0.067），
    # 学生被拉向 "song 优先" 的偏置。根因：guided hint 列出了类别名 & 强调歌唱线索，
    # 教师软分布被 hint 误导；beta=0.1 的近 forward-KL 又强制学生覆盖教师全部（错向）模态。
    #
    # 【v14 事故复盘】v14-20260702-112351 (LoRA + lr=1e-6 + direct hint + beta=1.0 +
    # sft_alpha=1.0 + topk=8 + max_steps=500，起点同为 SFT 97.83%)：
    # ckpt-50/100/200 三点评测 accuracy 直接崩到 56.06/56.64/56.05%。noise 与 porn 两
    # 整类 P/R/F1 全部 = 0，songP≈0.013 但 FP≈1300 沦为兜底类；有 ~2.4% 样本吐出
    # <tts_end>/<tts_pad>/<audio_XXXX> 音频码本 token（漂移到音频分支）。
    # 但 eval_loss 却在 [0.0719 → 0.0666] 稳步下降——train/eval loss 完全无法反映
    # 真实分类准确率的又一次证据。
    #
    # 根因分析：
    #   1) direct hint 让教师和学生看到不同 prompt（学生看不到 hint，教师看到
    #      "...ground-truth label is 'porn'"），教师对答案 token 置信度 ≈ 1.0，其他
    #      候选类别的相对概率结构被 hint 抹平。学生本来 well-calibrated 的多分类头
    #      被 JSD 反向梯度拉向"任何一类都能给 100%"的塌陷解。
    #   2) topk=8 + reverse-KL(β=1.0)：stepaudio 词表 ~150K，输出 preamble 里
    #      <tts_end>/<tts_pad>/<audio_XXX> 排位很高，一旦被 top-8 卷入，reverse-KL
    #      的 mode-seeking 特性让学生锁到这些 token，直接漂移到音频码本分支。
    #   3) LR=1e-6 对 SFT 起点 97.83% 的模型仍然过大——精修最多允许 1e-7 量级。
    #   4) train.jsonl 真实分布 speech ≈ 69% / porn ≈ 5% / noise ≈ 17%，minority
    #      class 每 batch 出现频率极低，NLL 项梯度几乎全被 speech 主导，minority
    #      class 一旦被 JSD 拉歪就再无 NLL 反拉的机会。
    #
    # 路径 A2（v15）：温和精修
    #   LR 1e-7 / warmup_ratio 0.1 / max_steps 200 / save&eval steps 20
    #   beta 0.5 / topk 5 / sft_alpha 2.0 / max_grad_norm 0.5
    #   TRAIN_JSONL = train_softbal.jsonl / hint = direct(参考式)
    #   实测：train_loss 200 步基本震荡在 0.11~0.40（首步 0.16，末步 0.13），
    #        eval_loss 0.117 → 0.118 完全没下降，grad_norm 一直在 1.1~2.7；
    #        分类准确率与 SFT 起点持平（未恶化，也未提升）——LR 太小 + max_grad_norm
    #        严裁 + max_steps 太短，学生根本没吸收信号；蒸馏没有生效。
    #
    # 【路径 A3（v16 起）：结构化软分布 OPSD 研究方案】
    # 目标：让 OPSD 真正跑起来（train/eval loss 下降 & 分类准确率不塌陷），验证
    #      "hint 结构化软分布 + minority oversampling" 能否超过 SFT baseline。
    #
    # 三个核心改动：
    #   (1) hint 从 direct(参考式) 改为 rank(多选题式)：不告诉教师答案，让教师
    #       自主输出对 5 个 candidate class 的相对置信度分布。这样：
    #         - 教师输出天然就是有意义的软分布（top-5 各类都有非零概率）
    #         - 保留了 OPSD 论文所需的"类间相对结构"信息
    #         - 消除了 direct 模式下"教师软分布 ≈ one-hot(gt)"的塌陷
    #         - 学生和教师看到的 prompt 都不含 gt，无信息泄漏，符合 OPSD 论文原意
    #   (2) LR 5e-7 (5× v15) / max_steps 1000 (5× v15) / max_grad_norm 1.0 (2× v15)
    #       / save&eval steps 100 (5× v15)：v15 的曲线证明当前配置下学生根本没被
    #       蒸馏推动，需要显著更大的学习步长和 update 数量才能观察到蒸馏是否有效。
    #   (3) TRAIN_JSONL 支持 train_targeted.jsonl：把 target_class(porn) 上采样
    #       到 30%，其余 4 类各 17.5%——比 softbal 更陡峭的分布，为 target class
    #       提供更多 NLL/JSD gradient budget，正面攻击"porn recall = 0"顽症。
    #       与 loss_scale 加权效果等价（gkd_trainer 不消费 loss_scale，只能靠采样）。
    #
    # 兼容性：
    #   OPSD_HINT_MODE=direct/guided/rank 三模式共存，环境变量切换；
    #   TRAIN_JSONL 环境变量可覆盖数据集选择（默认 softbal，A3 手动指定 targeted）。
    #
    # 【路径 A4（当前默认 · 恢复真 on-policy self-distillation）】
    # 【A3 结果复盘】v3 (max_steps=1000, lmbda=0.0, beta=0.5, lr=5e-7) sweep 结果：
    #     ckpt    accuracy   macro-F1    vs SFT-v7-150 (41.17)
    #     100     0.5571     0.3256      -8.61 pp
    #     300     0.5621     0.3481      -6.36 pp
    #     500     0.5421     0.3555      -5.62 pp  (v3 best)
    #     1000    0.5103     0.3289      -8.28 pp  (塌回 neutral+joy)
    # 结论：
    #   * v3 无一 ckpt 超过 SFT baseline；
    #   * 500 步后 macro-F1 塌陷 (neutral R 90→70, surprise R 51→25)；
    #   * lmbda=0 + teacher=disable_adapter(self) 让 KL 项梯度近零，训练完全走 CE。
    #
    # A4 三项核心改动（本次 v4 默认）：
    #   (1) LMBDA 0.0 → 1.0：每步真 on-policy generate，让 student 从自己的采样里
    #       学习，才叫 On-Policy Self-Distillation；同时 KL 项因 student 和 teacher
    #       输出分布真实不同而产生非零梯度。
    #   (2) BETA 0.5 → 0.1：v3 证明 mode-covering 分量在 self-distillation 下
    #       几乎无信号，降到 0.1 让 CE (SFT_ALPHA=2.0) 更主导训练方向，KL 只做正则。
    #   (3) LR 5e-7 → 1e-7 / MAX_STEPS 1000 → 300：on-policy 对 LR 更敏感（learner
    #       每步都要 sample，噪声更大），配合早停避免 v3 那种 "500 步后塌陷"。
    #
    # 配套：
    #   * OPSD_TARGET_OVERSAMPLE=1 默认开启（fear 从 2.7% → 30%，见 v3 fear R 只有
    #     10-14%，不上采样几乎无梯度），配合 OPSD_TARGET_RATIO=0.10-0.15 抑制过拟合。
    #   * SAVE/EVAL_STEPS 100 → 25：v3 每 100 步一存 (共 10 个 ckpt) 太粗，500-1000
    #     step 之间可能有 sweet spot 被漏；A4 每 25 步一存 (共 12 个 ckpt) 密集观察。
    #   * MAX_COMPLETION_LENGTH 16 → 8：分类 3-4 token 就够，lmbda=1.0 每步都 generate，
    #     缩短能显著减少显存。
    #   * OPSD_ONPOLICY_DRIFT_ABORT_STEPS=50 (默认)：训练结束或 kill 后回读
    #     output_dir/completions.jsonl 前 50 步，若出现 <audio_*>/<tts_*> 立即报警。
    #
    # 【路径 A5（当前默认 · 修 teacher hint + 多类 minority 上采样）】
    # 【A4 结果复盘】v4 (max_steps=300, lmbda=1.0, beta=0.1, lr=1e-7, oversample fear@0.10)
    # sweep 结果：
    #     ckpt    accuracy   macro-F1    vs SFT-v7-150 (41.17)
    #     25      0.5575     0.3290      -8.27 pp
    #     150     0.5590     0.3482      -6.35 pp  (v4 best)
    #     300     0.5525     0.3426      -6.91 pp  (轻微退步)
    # 关键 minority recall: fear 14->16 / disgust 12->13 / anger 10->13——远低于 SFT。
    # 根因（v4 首次暴露的真凶）：
    #   train.opsd.rank.jsonl 里 teacher_prompt hint = "rank {speech, music, noise, porn, song}"，
    #   完全是 OPSD 五类版遗留，与 MELD 7 类冲突。教师被喂逻辑不通的 prompt，softmax
    #   分布紊乱，KL 分支几乎无有效蒸馏信号；训练实质仍是 CE-driven SFT (LR=1e-7 时几乎不动)。
    #
    # A5 三项核心改动（本次 v5 默认）：
    #   (1) hint 模板 7 类化：修正 rank/direct/guided 三个 template 里的类别列表；
    #       hash 变化自动重建 train.opsd.rank.jsonl。这是 A5 的 primary fix。
    #   (2) 多类 oversample：新增 OPSD_TARGET_CLASSES=fear,disgust,anger，把三个
    #       严重欠拟合的 minority 类同时抬到 ratio=0.12；避免 v4 只补 fear 单类，
    #       disgust/anger 依然"每 batch 0-1 样本"导致学不动。
    #   (3) LR 1e-7 → 3e-7 / BETA 0.1 → 0.2：hint 修好后教师有真信号，敢学快一点、
    #       给 KL 分支更大权重；MAX_COMPLETION_LENGTH 8 → 6 进一步压显存 / 防漂。
    #
    # 若 v5 仍未超过 SFT baseline，下一步 A6 应考虑：
    #   (a) 直接放弃 rank hint，改 direct hint（教师看到 gt）+ 强 CE alpha；
    #   (b) 或彻底放弃 OPSD 路线，改 SFT + class-balanced sampler / focal loss。
    DEFAULT_LR=3e-7
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
# 【v13 事故教训】v13 用 beta=0.1 (近 forward-KL)，学生被强制覆盖教师所有次优模态；
# 只要教师在 hint 影响下把 song 放在 top-2/top-3 一次，学生就会持续被拉向 song，
# 4000 步累积后就成了 "song 优先" 的偏置模型（song FP=3120）。
# 【v14 事故教训】v14 用 beta=1.0 (纯 reverse-KL) + topk=8：reverse-KL 的
# mode-seeking 特性让学生只锁 top-k 内的高概率模式，一旦教师在 top-8 里出现
# <tts_end>/<audio_XXX>（stepaudio 输出 preamble token 排位很高），学生就漂移到
# 音频码本分支——ckpt-50 已产出 2.4% 无效响应。
# 路径 A2 改为 beta=0.5 (标准 JSD)：mode-covering 与 mode-seeking 平衡，配合
# topk=5 物理过滤后已经不含 <tts_*>/<audio_*>，且不再对教师顶点 token 过度敏感。
# 【路径 A3】保持 beta=0.5：rank hint 下教师输出结构化软分布（top-7 各类都有非零
# 概率，非 one-hot），mode-covering 分量能让学生学到"完整的类间相对结构"（不只是
# top-1 峰值），mode-seeking 分量防止学生在低概率类别上过度覆盖噪声。这是 OPSD
# 论文推荐的默认值，也是路径 A3 相对 v13/v14 极端配置的一次矫正。
# 【路径 A4（v3 结果复盘）】v3 (macro-F1 500 步=35.55 vs SFT 41.17) 证明 beta=0.5 +
# lmbda=0.0 组合下 KL 项做的是"student ↔ disable_adapter(student)"，自蒸馏
# gradient 几乎为零，训练完全被剩下 50% × CE 主导，等价于 LR=2.5e-7 的 SFT。
# 结果就是 sweep_eval_summary 里所有 ckpt 都在 SFT baseline 之下。
# A4 把 beta 从 0.5 降到 0.1：CE 项主导（90%）保证学生朝"训练数据里的教师软标签"
# 前进，KL 项（10%）只做"轻正则，防止学生塌回 base neutral 塌陷解"。
# 【路径 A5（v4 结果复盘）】v4 macro-F1 best=34.82 (150 步) 仍低于 SFT 41.17，
# 且 fear R 只涨到 16.00 / disgust R 只涨到 13.24 / anger R 停在 10-12%。
# 根因：train.opsd.rank.jsonl 里 teacher_prompt 的 hint 模板是从 OPSD 五类版
# 直接 copy 过来的"rank {speech, music, noise, porn, song}"——与 MELD 7 类完全
# 无关，教师被喂了逻辑冲突的 prompt，softmax 输出紊乱，KL 分支几乎没信号。
# A5 修复：把 rank/direct/guided 三个 hint 模板全部改为 MELD 7 类；hash 变化
# 会自动触发 rank.jsonl 重建。同时把 beta 从 0.1 提到 0.2，让好 hint 的教师
# 信号影响力适度增加。
BETA=${BETA:-0.2}
# lmbda：on-policy 触发概率
#   1.0 = 每个 step 都用学生当前权重 generate 一段 completion 做训练（纯 on-policy）
#   0.0 = 不做 on-policy 采样，直接用数据集里的 assistant 答案作为目标序列
#         （等价于 offline distillation，速度最快）
# 【重要】base/未充分 SFT 的 stepaudio 直接 lmbda=1.0 会灾难性漂移：
#         第一步学生就可能生成 <audio_XXX> 音频码本 token，教师被喂错误 completion
#         做条件生成，软分布被污染，几百步内模型就完全崩坏。
#         必须使用 offline 自蒸馏 (lmbda=0)，学生 target 直接锚在数据集真实标签，
#         教师在同一位置给出软分布，才能稳定训练。
#         如果 MODEL_PATH 已是 SFT 后的 checkpoint 且分类准确率 >85%，
#         再考虑改回 1.0 做纯 on-policy 精修。
# 【路径 A4】MELD SFT 起点 acc=58% 远达不到 "85%" 门槛，但 v3 结果证明 lmbda=0.0
# 让 OPSD 退化成 KL-regularized SFT，方向由 train.opsd.rank.jsonl 的教师软标签
# 决定，不是"策略自我改进"。改成 1.0 试真 on-policy self-distillation——但必须
# 配合前 50 步 completions 漂移守护（见 OPSD_ONPOLICY_DRIFT_ABORT_STEPS），若
# 出现 <audio_XXX>/<tts_end> 立即 abort，防止 stepaudio 音频码本分支污染教师。
LMBDA=${LMBDA:-1.0}
# 采样温度（仅 lmbda>0 时影响学生 generate；分类任务不需要多样性）
TEMPERATURE=${TEMPERATURE:-1.0}
# Top-K logits 蒸馏：
# stepaudio 词表 ~150K，其中 2K+ 是 <audio_XXX> 音频码本 token；如果 K 太大，
# top-k 里会混入大量音频 token，噪声主导 KL 目标，加剧漂移到音频分支。
# 【v14 事故教训】K=8 时 <tts_end>/<tts_pad>/<audio_XXX> 会稳定进入 top-k，学生在
# reverse-KL 下锁到这些 token，直接漂到音频分支（ckpt-50 就有 2.4% invalid response）。
# 路径 A2 直接压到 K=5 = 类别数，物理上过滤掉所有 <tts_*>/<audio_*>，蒸馏目标
# 只保留 5 个类别 label token 之间的相对分布——这才是分类任务的真实软标签信号。
# 【路径 A5】保持 GKD_LOGITS_TOPK=5（OPSD 五类版遗留）——【路径 A6】改为 7
# 对齐 MELD 7 类：不然 KL 分布里总会有 2 个类被截断（取决于 teacher softmax top-k），
# minority 类很可能刚好被删，蒸馏信号丢失。
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-7}
# sft_alpha：在非 student 生成的样本上额外混入一份 SFT loss 比例
#   0.0 = 纯蒸馏；>0 = 蒸馏 + 部分 NLL，作为"硬锚"防止 KL 把学生推歪。
# 【v13 事故教训】v13 用 sft_alpha=0.5 依然不够——教师(guided)+topk=8 的错向 JSD
# 反向梯度太强，NLL 项被淹没。路径 A 直接把 sft_alpha 提到 1.0 让 NLL 与 JSD 等权，
# 由于 direct hint 下 JSD 目标已经接近 one-hot(gt)，两项方向一致，训练非常稳。
# 【v14 事故教训】v14 用 sft_alpha=1.0 依然崩坏：direct hint 让教师软分布抹平了
# 类间相对概率结构，学生的多分类头被 JSD 拉向塌陷解，minority class (noise/porn)
# 一旦被 JSD 拉歪，NLL 项由于 minority 每 batch 出现频率极低，反拉能力不足。
# 路径 A2 提到 2.0：让 NLL 硬锚权重 2× 于 JSD，压制 JSD 的错向梯度。同时配合
# balanced 数据集提高 minority class 每 batch 的 NLL 反拉信号。
# 【路径 A6】从 2.0 提到 3.0：v5 结果显示 KL 项 (self-distillation 下) 信号弱，
# CE 才是主信号源；加大 sft_alpha 让 CE 在 loss 里权重更高，配合 minority oversample
# 共同推进 tail 类学习。
SFT_ALPHA=${SFT_ALPHA:-3.0}
# 单条 completion 最长生成多少 token（仅 lmbda>0 时生效）。
# stepaudio 输出基本是 1 个标签词，MELD 最长标签 "surprise"/"sadness"/"disgust"
# 也在 3-4 token 内；A4 收紧到 8 减轻 on-policy generate 显存压力（lmbda=1.0 每步
# 都要 student.generate），同时防止学生生成过长序列漂移到音频分支。
# 【路径 A5】进一步收紧到 6：v4 completions 观察到全部 <= 4 token，8 是过度冗余；
# 6 依旧远大于最长标签，同时 batch 显存再省 ~25%（generate 阶段主要开销）。
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-6}
# 梯度裁剪
# 【v14 事故教训】max_grad_norm=1.0 下单个 high-loss batch 就能造成显著位移。
# 起点已 97.83% 的精修场景下任何大位移都是灾难。路径 A2 收紧到 0.5。
# 【路径 A3】v15 实测 grad_norm 一直在 1.1~2.7 但 loss 完全不动，说明 0.5 严裁 +
# LR 1e-7 的组合让每步 effective 更新几乎为 0（clip 后梯度只是原始梯度的 20~50%
# 再乘 1e-7）。放宽到 1.0（HF 默认值）+ LR 5e-7 让蒸馏信号真的能推动学生。
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---------------- 稳定性 / 恢复训练（可选，与 GRPO 脚本对齐） ----------------
# RESUME_CHECKPOINT : 从已有 ckpt 恢复训练；传入 checkpoint-XXXX 目录绝对路径。
# RESUME_ONLY_MODEL : 只加载模型权重，不恢复 optimizer/scheduler/RNG/DS 分片状态。
#                     本脚本默认 --save_only_model true，保存的 ckpt 里没有 optimizer 等
#                     完整训练状态；此时恢复训练必须设 RESUME_ONLY_MODEL=true，否则
#                     transformers 会报 "Can't find a valid checkpoint"。
#                     等价于把 ckpt 当作"新的初始化权重"，从 step 0 重新走 warmup。
#                     常见场景：先 lmbda=0 offline 蒸馏跑到较优 ckpt，再切 lmbda=1.0
#                     做纯 on-policy 精修——此时用 RESUME_CHECKPOINT 把上一阶段成果
#                     作为起点即可。
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-true}
else
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-}
fi

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
# 【v14 事故教训】warmup_ratio=0.03 + max_steps=500 只有 15 步 warmup。
# 前 15 步内 LR 就已经 ramp 到 peak，学生权重从起点被 direct-hint 的错向 JSD 大步位移。
# 路径 A2 提到 0.1（配合 max_steps=200，实际 warmup=20 步 + LR peak 已经缩到 1e-7），
# 更长的 warmup 让 LoRA 缓慢介入，防止起步就漂。
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
case "$GRADIENT_CHECKPOINTING" in
    1|true|True|TRUE) GC_FLAG=true ;;
    0|false|False|FALSE) GC_FLAG=false ;;
    *) GC_FLAG=true ;;
esac
# MAX_STEPS: 提前终止训练总步数（与 GRPO 脚本对齐）。
#   >0  : 显式限制步数，用于早停 / 节省时间；
#   0/负: 走完整个 epoch。
# 【v13 事故教训】v13 跑了 4000 步严重过训，从 97.83% 崩到 50.13%——
# 起点已经是 97.83% 的 SFT 模型，OPSD 的作用只是软标签精修，边际收益天生很小；
# 4000 步远远超过精修所需的规模。
# 路径 A：train.jsonl 56302 行，8 卡 / bs=1 / grad_accum=8 下 1 epoch ≈ 880 优化步，
# 精修目标只需 500 步（约 0.57 epoch），配合 SAVE_STEPS=50 每 50 步存一个 ckpt，
# 挑真实分类评测最好的那个即可。若发现 500 步内还没进入下降通道，再考虑加大。
# 【v14 事故教训】500 步已过训——ckpt-50/100/200 三点评测均已崩到 56%，越训越差
# 只是 song FP 略有波动。起点已 97.83% 的精修根本容不下 500 步的位移空间。
# 路径 A2 压到 200 步（约 0.23 epoch），配合 SAVE_STEPS=20 每 20 步一个 ckpt（共 10
# 个），保留在 model 完全没被搞坏之前挑到最优点的机会；即使 200 步内已崩，也能
# 从 ckpt-20/40 里挑到更接近 SFT 起点的版本。
# 【路径 A3】1000 步（约 1.14 epoch on softbal / 1.05 epoch on targeted）：
# rank hint 下教师软分布结构不再塌陷，学生每步吸收的有效蒸馏信号显著增加，
# 配合 LR 5e-7（v15 的 5×）让训练真正启动。SAVE_STEPS/EVAL_STEPS=100 每 100 步
# 一个 ckpt（共 10 个），仍便于外部真实评测挑最优点。若 300 步内 accuracy 掉头
# 立即 kill 训练。
# 【路径 A4】v3 (max_steps=1000) 结果证明 500 步后必塌，A4 直接截到 300 步，
# 减少 3.3× wall-clock。若 100 步内 macro-F1 就掉头，OPSD_SANITY_ONLY_WARN=1
# 让训练继续也可以，但更建议立即 kill 换 A5。
# 【路径 A6】300 → 600（~1.4 epoch）：v5 只跑 0.74 epoch，tail 类（fear/disgust）
# 重复样本才被看过不到 1 遍；A6 把 training budget 支回到 1.4 epoch 充分学习，
# 同时保留早停能力（若 macro-F1 在 200-400 step 已接近峰值可手动 kill）。
MAX_STEPS=${MAX_STEPS:-600}

# checkpointing
# 【路径 A2 短训练策略】200 步内每 20 步存一个 ckpt (共 10 个)，方便挑最优点。
# 训练指标 (train/eval loss) 对分类准确率**极不敏感**：
#   - v13 eval_loss 从 0.029→0.024 稳步下降，真实 accuracy 从 97.83% 崩到 50.13%
#   - v14 eval_loss 从 0.0719→0.0666 稳步下降，真实 accuracy 从 97.83% 崩到 56.06%
# 必须靠外部真实评测挑 ckpt——建议每存一个 ckpt 就跑一次 val.jsonl 分类评测，
# 一旦 accuracy 掉头（如从 97.83% 跌到 <97%）立即停止训练。
# 【路径 A3】1000 步 / 每 100 步存一个（共 10 个 ckpt），与 v13/v14 事故经验一致：
# 短时间内密集存 ckpt + 外部评测挑最优，不依赖 in-training loss。
# 【路径 A4】300 步 / 每 25 步存一个（共 12 个 ckpt），比 A3 密度更高，因为：
#   (a) v3 (A3) 显示 300→500 之间是 macro-F1 上升期，500 之后塌陷，需要在这
#       个区间密集打点；
#   (b) on-policy 训练每步真的 sample，个别 batch 可能引入较大噪声位移，密集
#       存 ckpt 能捕获瞬时最优点。
# 【路径 A6】SAVE/EVAL_STEPS 25 → 50：600 步 / 50 = 12 个 ckpt（与 A5 相等密度），
# 避免 600/25=24 个 ckpt 磁盘保存翻倍开销（lora ckpt 小，但训练内 eval 同时
# 发生会加剧开销）。
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-50}
save_total_limit=${save_total_limit:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}
# GKD trainer 支持 --log_completions（详见 swift/rlhf_trainers/gkd_trainer.py:727）：
# 每步把学生 on-policy 生成的 prompt/completion 写到 output_dir/completions.jsonl，
# 是最早发现"学生吐 <audio_XXX>"漂移的观测手段。lmbda=0 时无 completion，可关闭。
LOG_COMPLETIONS=${LOG_COMPLETIONS:-true}

# ---------------- 序列 / 截断 ----------------
# eager attention 下 attention softmax buffer ≈ B * H * L * L * 4 bytes (fp32)，
# Qwen2 的 stepaudio_mini 配置约 14 头，L=4096 时单层即 ~3.7GB，再叠加 OPSD 教师/学生
# 双 forward + on-policy generate，95G 卡几乎必爆。
# 【实测踩坑 · v6-20260701-171628】MAX_LENGTH=4096 时前 5 步 memory 就已 92.77 GiB，
# 长音频样本一来 attention softmax 直接 tried to allocate 1.72 GiB CUDA OOM。
# attention buffer 是 O(L^2)：4096 -> 3072 单层 attention 显存直降 ~44%，
# 全网络叠加节省约 15~20 GiB，是最有效的显存优化手段。
# stepaudio 音频 <audio_patch> 数量与 mel 帧数强绑定，超长音频（>~30s）会被
# truncation_strategy=delete 整条丢掉 -> resample_encode_failed_inputs 换一条，
# 数据侧代价可控。若发现 loss 长期在同批次 num_valid=0 抖动，再考虑升到 3584/4096。
# 默认压到 3072；教师侧走 total_length + 256 buffer（gkd_trainer._OPSD_TEACHER_LENGTH_BUFFER），
# hint 引入的额外 tokens 已被吸收，不需要为教师侧多留 max_length 空间。
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
# CUDA 分配器优化：
#   - expandable_segments:True    大块显存不用重新分配，减少碎片
#   - max_split_size_mb:128       避免大块被切碎，OPSD 长序列 attention 大分配友好
#   - garbage_collection_threshold:0.8  显存占用 > 80% 时主动触发 GC 回收未使用 block；
#                                       eager attention 下每步产生大量临时张量，激进 GC
#                                       能显著降低 fragmentation 导致的伪 OOM（实测有效）
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}
# 即使不用 vLLM 也屏蔽相关 warning，避免 transitive import 干扰日志
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"}

# ---------------- 数据集 ----------------
# 【重要 · 与 GRPO 脚本对齐】stepaudio train.jsonl 五类分布严重失衡
# （speech ≈ 69% / song ≈ 3.7%）。GRPO v0/v2 曾出现 minority class recall=0 的
# mode collapse，改用 tools/balance_train_jsonl.py 生成的 train_balanced.jsonl 后
# 稳定收敛。OPSD 有同样风险：
#   - 教师本身也是 stepaudio，在 majority class 上先验很强；
#   - hint 只提供"分类框架"（不泄漏答案）时，教师软分布依然会被 majority prior 拉向 speech；
#   - 学生在失衡数据上跑 KL，几乎只在 speech 位置有梯度信号 → minority class 学不到。
# 【v13 事故教训 · 数据源修正】v13 使用了 train_balanced.jsonl（各类 20%）依然崩坏，
# 证明 v13 的 song 偏置**不是数据分布问题**，而是 guided hint 引发的教师软分布错向。
# 路径 A 用 direct hint（教师软分布 ≈ one-hot(gt)），完全不受 hint 语义偏差影响，
# 因此可以直接用真实分布的 train.jsonl（56302 行）做精修——保留自然先验，
# 让模型对 majority class (speech) 也能小幅继续锐化。
# balanced 版本仍可通过 TRAIN_JSONL=... 覆盖使用。
#
# 【v14 事故教训 · 数据源修正】v14 用真实分布 train.jsonl 崩坏后确认：failure mode
# 是 noise/porn 两个 minority class 直接归零、song 沦为兜底。真实分布下 minority
# 每 batch 期望仅 ~0.05 条正样本，NLL 项对它们几乎没有反拉能力，一旦 direct-hint
# 的 JSD 把它们拉歪就再没有回头路。
#
# 【v14 数据源二次审计】重新审阅所有候选数据集的类别分布：
#   - train.jsonl (56302 行)         speech 69% / noise 16% / music 5.7% / porn 4.8% / song 3.7%
#     — 与 val 分布完全一致（val: 69/17/5.4/5.0/3.6%），但 minority 反拉不足。
#   - train_balanced.jsonl (195330 行) 各类精确 20%
#     — song 从 2063 → 39066 (19× 复制)、porn 从 2725 → 39066 (14× 复制)，
#       同一音频重复出现十几次，模型极易过拟合到具体音频指纹；且训练分布 speech=20%
#       与 val 分布 speech=69% 差 3 倍，即使学"对"了 balanced 分布，val 上也会大量误判。
#   - train_balanced_v2.jsonl (62000 行) speech 32% / porn 19% / noise 19% / music 16% / song 13%
#   - train_softbal.jsonl (45231 行)   speech 27% / noise 20% / song 18% / porn 18% / music 18%
#     — 【首选】既保留了"speech 是主类"的方向性先验（27% > val 分布外的其它 4 类合计），
#       又让每个 minority class 每 batch 至少期望 ~1.4 条正样本，NLL 反拉充分；
#       所有类别的 unique 音频数都在合理区间（不像 balanced 那样 song 复制 19×）。
# 路径 A2 默认改用 train_softbal.jsonl，兼顾 (1) 分布方向对齐 val (2) minority NLL 反拉
# (3) 避免过度重复上采样导致的音频指纹过拟合。若要 100% 均衡，仍可 TRAIN_JSONL=... 覆盖。
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
# MELD 临时还没有 train_softbal.jsonl (需 balance_train_jsonl.py 生成), 默认
# 回退到 train.jsonl (原始分布) —— OPSD 本身会通过 hint mode + SFT_ALPHA 平衡,
# 不一定需要事先均衡。
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.jsonl"}
if [ ! -f "$TRAIN_JSONL" ]; then
    # 【自动派生】train_softbal.jsonl 缺失时自动调用 balance_train_jsonl.py 生成，
    # 而不是直接 exit——用户无需手动预处理，脚本对新机器/新 checkout 自解释。
    # 只对默认路径（TRAIN_JSONL 未被环境变量覆盖）触发自动派生：显式指定的数据文件
    # 缺失仍走 fatal 分支（可能是路径拼错，不应该被自动创建）。
    _EXPECTED_SOFTBAL="$DATA_DIR/train_softbal.jsonl"
    _SRC_JSONL="$DATA_DIR/train.jsonl"
    if [ "$TRAIN_JSONL" = "$_EXPECTED_SOFTBAL" ] && [ -f "$_SRC_JSONL" ]; then
        echo "[INFO] $_EXPECTED_SOFTBAL 不存在，自动从 $_SRC_JSONL 派生软均衡数据集..."
        # softbal 目标分布（对应脚本注释里描述的 speech 27% / noise 20% / minority 各 18%）：
        #   speech: 12212 (原 39066 下采样 3.20x → 27.0%)
        #   noise:   9046 (原  9231 基本保留 0.98x → 19.9%)
        #   music:   8142 (原  3217 上采样  2.53x → 17.9%)
        #   porn:    8142 (原  2725 上采样  2.99x → 17.9%)
        #   song:    8142 (原  2063 上采样  3.95x → 17.9%)
        #   合计: 45684 行
        # 注意：balance_train_jsonl.py 的 --target 只支持 max/median/mean/<int>，
        # 不认 'soft'——之前 [FATAL] 分支给出的 --target soft 命令是**错的**，
        # 正确用法是 --custom-target 精确指定 5 类计数。
        _BALANCE_TOOL="$SCRIPT_DIR/tools/balance_train_jsonl.py"
        if [ ! -f "$_BALANCE_TOOL" ]; then
            echo "[FATAL] balance_train_jsonl.py 不存在: $_BALANCE_TOOL"
            exit 14
        fi
        # 使用与 SFT/GRPO 环境一致的 python：env-3.12.11
        _BALANCE_PY=${BALANCE_PY:-/data/miniconda3/envs/env-3.12.11/bin/python}
        if [ ! -x "$_BALANCE_PY" ]; then
            _BALANCE_PY=$(command -v python3 || command -v python)
        fi
        if ! "$_BALANCE_PY" "$_BALANCE_TOOL" \
                --input "$_SRC_JSONL" \
                --output "$_EXPECTED_SOFTBAL" \
                --custom-target 'speech=12212,noise=9046,music=8142,porn=8142,song=8142' \
                --seed 42; then
            echo "[FATAL] 自动派生 $_EXPECTED_SOFTBAL 失败。请手动执行："
            echo "        $_BALANCE_PY $_BALANCE_TOOL \\"
            echo "            -i $_SRC_JSONL \\"
            echo "            -o $_EXPECTED_SOFTBAL \\"
            echo "            --custom-target 'speech=12212,noise=9046,music=8142,porn=8142,song=8142' \\"
            echo "            --seed 42"
            exit 14
        fi
        if [ ! -f "$_EXPECTED_SOFTBAL" ]; then
            echo "[FATAL] balance_train_jsonl.py 执行返回 0，但输出文件仍不存在: $_EXPECTED_SOFTBAL"
            exit 14
        fi
        echo "[INFO] 派生完成: $_EXPECTED_SOFTBAL"
    else
        echo "[FATAL] TRAIN_JSONL 不存在: $TRAIN_JSONL"
        if [ ! -f "$_SRC_JSONL" ]; then
            echo "        源文件 $_SRC_JSONL 也不存在，无法自动派生。"
        fi
        echo "        显式指定其它数据集："
        echo "          TRAIN_JSONL=$_SRC_JSONL bash $(basename \"$0\")"
        echo "        或手动派生 train_softbal.jsonl（softbal ≈ 27/20/18/18/18%）："
        echo "          python $SCRIPT_DIR/tools/balance_train_jsonl.py \\"
        echo "              -i $_SRC_JSONL \\"
        echo "              -o $DATA_DIR/train_softbal.jsonl \\"
        echo "              --custom-target 'speech=12212,noise=9046,music=8142,porn=8142,song=8142' \\"
        echo "              --seed 42"
        exit 14
    fi
fi

# ---------------- Target-class oversampling（路径 A3 · 建议 5：per-class 加权替代方案）----------------
# 背景：stepaudio 分类任务 assistant response 只有 1 个 label token，OPSD 里 JSD/CE
# 都基于该 token 位置的分布计算。理论上应通过 per-class weighted CE（loss_scale）
# 让 target class（MELD 版默认 fear，OPSD 五类版默认 porn）在梯度里获得更大权重来
# 对抗 minority class 的欠学习问题；
# 但 gkd_trainer 的 self-distillation 分支不消费 loss_scale（详见
# swift/rlhf_trainers/gkd_trainer.py:_is_self_distillation 分支，只对 outputs.loss
# 加 sft_alpha 系数，未按样本 mask 加权），因此只能通过**采样加权**近似实现。
#
# 采样加权 = 让 target class 样本在训练里被"重复看到"若干次，等价于该类每 sample
# 的 loss 权重被放大。相比 loss_scale 有两点差异：
#   (a) 优点：不改代码，兼容 gkd_trainer / grpo_trainer 所有走 GKDLoss 的场景；
#   (b) 缺点：会引起同一音频指纹被多次看到（v13 事故观察到的过拟合模式），必须
#       控制 oversample ratio 上限，避免 balanced 版本那样 19× 复制的极端。
#
# 实现：从 SRC=$TRAIN_JSONL 出发对 OPSD_TARGET_CLASS 类做等比例复制，把它抬升到
# OPSD_TARGET_RATIO（默认 30%），其他类总量按现状保留，target 类多出来的部分通过
# 重复采样补齐。分布参考：
#   - MELD  train.jsonl (9989 条): neutral 48% / joy 17% / surprise 12% / anger 12% /
#                                  sadness 7.2% / disgust 2.7% / fear 2.7%
#   - OPSD  train_softbal.jsonl :  speech 27% / noise 20% / song 18% / porn 18% / music 18%
# MELD fear 从 2.7% → 30% 意味着 fear 样本要被复制约
#   (0.30 / 0.027) × (1 - 0.027) / (1 - 0.30) ≈ 15.4 倍，
# 与 balanced 版本 14× 复制同数量级；v13 事故（14×+ 复制导致音频指纹过拟合）就在
# 这个区间，务必配合 OPSD_TARGET_RATIO 更保守的值（比如 0.10-0.15）使用。
# 若沿用 OPSD 五类语境：porn 从 18% → 30% 只需约 0.67 倍额外拷贝，风险远低于 MELD。
#
# 触发方式：
#   OPSD_TARGET_OVERSAMPLE=1 bash run_train_opsd_meld.sh   # 自动派生 & 使用 targeted
#   OPSD_TARGET_CLASS=fear OPSD_TARGET_RATIO=0.30          # MELD 默认；可覆盖类别与目标占比
#   （旧 OPSD 五类版对应用例是 OPSD_TARGET_CLASS=porn，此处保留兼容）
# 关闭方式（默认）：
#   不设 OPSD_TARGET_OVERSAMPLE 或 =0，走 TRAIN_JSONL 原样。
# 【路径 A4】MELD fear/disgust 支持数 ~50-70，一个 epoch 出现 1-2 次，on-policy
# 采样时 target class 梯度信号几乎为零。默认开启 oversample，同时把 ratio 收紧
# 到 0.10（fear 从 2.7% → 10%, 约 3.7× 复制，远低于 v13 事故的 14× 阈值），
# 保证 target class 每 epoch 出现 ~3-4 次而不至于过拟合音频指纹。
OPSD_TARGET_OVERSAMPLE=${OPSD_TARGET_OVERSAMPLE:-1}
# 【MELD 版】默认关注 fear（美回归 minority 类, test recall ~0.22），
# OPSD 五类脚本里默认是 porn；两者语义不通用，请勿沿用旧值。
# 该变量同时影响 (1) target oversample 的类别; (2) sanity check 的 focus 默认;
# (3) sanity 内部调 run_eval_meld.sh 时透传的 TARGET_CLASS。
OPSD_TARGET_CLASS=${OPSD_TARGET_CLASS:-fear}
# 【路径 A5】多类 oversample。OPSD_TARGET_CLASSES 逗号分隔类名列表，每一个
# 类都会独立扩容到 OPSD_TARGET_RATIO（等比例复制），共存于同一个训练文件。
# v4 结果证明单类 oversample（fear）时 disgust/anger 完全没被补齐 → 三类 recall
# 全部远低于 SFT。默认列出三个 tail 类；若未设置则回退到旧的 OPSD_TARGET_CLASS 单类。
# 关闭方式：显式设为空 OPSD_TARGET_CLASSES="" 时脚本回退单类语义。
OPSD_TARGET_CLASSES=${OPSD_TARGET_CLASSES:-fear,disgust,anger}
# 【路径 A4】ratio 从 0.30 → 0.10：fear 从 2.7% → 10% 已经是 ~3.7× 复制，够每
# batch 能抽到 fear 样本；0.30 会到 15.4× (见 v13 事故红线 14×)，音频指纹过拟合
# 风险大。用户需要更激进可手动 OPSD_TARGET_RATIO=0.20 覆盖。
# 【路径 A5】0.10 → 0.12：v4 只上采 fear 单类，fear R 只涨 2pp，disgust/anger 完全没管；
# A5 用 OPSD_TARGET_CLASSES 同时上采 fear/disgust/anger 三类，每类目标占比 12%，
# 主类合计留 64% 空间（neutral 天然是主类，仍占大头）。三类各 ~4-6× 复制，远低
# 于 v13 事故红线 14×，音频指纹过拟合风险可控。
OPSD_TARGET_RATIO=${OPSD_TARGET_RATIO:-0.12}
# 【路径 A6】per-class 目标绝对数（覆盖 OPSD_TARGET_CLASSES/RATIO 的"统一 ratio"策略）。
# 格式: "cls1=N1,cls2=N2,..."，非空时按此绝对数扩容每个类；为空则回退 A5 逻辑。
# v5 复盘：A5 的"每类都到 12%"太粗——anger 原本已占 11.1%，扩到 12% 几乎没做上采样
# (1.3×)；disgust 原本 2.7% 扩到 12% 需要 5.8×，可能过拟合音频指纹；sadness 6.8%
# 被遗漏。A6 用绝对数，按每类"当前难度 + 支持数 + 音频过拟合红线"定制：
#   fear    : 268 → 800  (~3.0×  极稀有，需扩容但复制倍数收敛，防音频指纹)
#   disgust : 271 → 600  (~2.2×  比 A5 的 5.8× 降 62%，缓解 disgust R 上不去的过拟合)
#   sadness : 683 → 1000 (~1.5×  A5 遗漏，补齐)
#   anger   : 1109 → 1500 (~1.35×  略补 minority signal)
# 期望扩容后分布 ≈ neutral 42% / joy 15.6% / anger 13.4% / surprise 10.8% /
#                sadness 8.9% / fear 7.1% / disgust 5.4%（tail 类不再<3%）。
# 关闭方式：显式设为空 OPSD_TARGET_COUNTS="" 回退 A5 ratio 逻辑。
OPSD_TARGET_COUNTS=${OPSD_TARGET_COUNTS:-fear=800,disgust=600,sadness=1000,anger=1500}
# 【路径 A4】on-policy 漂移事后守护窗口（仅 LMBDA>0 时生效）。
# 训练结束后扫描 output_dir/completions.jsonl 前 N 行，如果出现 <audio_*> 或
# <tts_*>/<voice_*> 字符（stepaudio 音频码本分支的标志），报 [WARN][DRIFT]。
# 0 禁用扫描。方便事后定位漂移，不接管 kill 行为 (kill 需手动判断)。
OPSD_ONPOLICY_DRIFT_ABORT_STEPS=${OPSD_ONPOLICY_DRIFT_ABORT_STEPS:-50}
if [ "$OPSD_TARGET_OVERSAMPLE" = "1" ]; then
    _tgt_base=$(basename "$TRAIN_JSONL" .jsonl)
    # 【路径 A6】优先 OPSD_TARGET_COUNTS (per-class 绝对数)；退化到 A5 的 OPSD_TARGET_CLASSES；再退到 A4 单类
    if [ -n "$OPSD_TARGET_COUNTS" ]; then
        _tgt_classes_effective=$(printf '%s' "$OPSD_TARGET_COUNTS" | awk -F',' '{for(i=1;i<=NF;i++){split($i,a,"="); printf (i==1?"":",") a[1]}}')
        # 文件名 tag：把 "=" 换成 "@"（fs 安全），逗号换成 -
        _tgt_tag=$(printf '%s' "$OPSD_TARGET_COUNTS" | tr '=' '@' | tr ',' '-')
        TRAIN_JSONL_TARGETED=${TRAIN_JSONL_TARGETED:-"$DATA_DIR/${_tgt_base}.tgt-${_tgt_tag}.jsonl"}
        _oversample_mode="counts"
    else
        _tgt_classes_effective="${OPSD_TARGET_CLASSES:-$OPSD_TARGET_CLASS}"
        _tgt_tag=$(printf '%s' "$_tgt_classes_effective" | tr ',' '-')
        TRAIN_JSONL_TARGETED=${TRAIN_JSONL_TARGETED:-"$DATA_DIR/${_tgt_base}.tgt-${_tgt_tag}-${OPSD_TARGET_RATIO}.jsonl"}
        _oversample_mode="ratio"
    fi
    if [ ! -f "$TRAIN_JSONL_TARGETED" ]; then
        echo "[INFO] 派生 multi-target-class oversampled 训练集 (mode=$_oversample_mode): $TRAIN_JSONL_TARGETED"
        echo "[INFO]   target_classes = $_tgt_classes_effective"
        if [ "$_oversample_mode" = "counts" ]; then
            echo "[INFO]   target_counts  = $OPSD_TARGET_COUNTS"
        else
            echo "[INFO]   target_ratio   = $OPSD_TARGET_RATIO (each)"
        fi
        SRC_JSONL="$TRAIN_JSONL" \
        DST_JSONL="$TRAIN_JSONL_TARGETED" \
        TARGET_CLASSES="$_tgt_classes_effective" \
        TARGET_RATIO="$OPSD_TARGET_RATIO" \
        TARGET_COUNTS="$OPSD_TARGET_COUNTS" \
        /data/miniconda3/envs/env-3.12.11/bin/python - <<'PYEOF'
import json, os, random, sys
random.seed(42)

SRC = os.environ['SRC_JSONL']
DST = os.environ['DST_JSONL']
TARGET_CLASSES = [c.strip() for c in os.environ['TARGET_CLASSES'].split(',') if c.strip()]
TARGET_RATIO = float(os.environ['TARGET_RATIO'])
TARGET_COUNTS_RAW = os.environ.get('TARGET_COUNTS', '').strip()

# 【路径 A6】per-class 绝对数模式：解析 "cls=N,cls2=N2,..." 到 dict
target_counts = {}
if TARGET_COUNTS_RAW:
    for kv in TARGET_COUNTS_RAW.split(','):
        kv = kv.strip()
        if not kv:
            continue
        if '=' not in kv:
            sys.stderr.write(f'[FATAL] OPSD_TARGET_COUNTS entry missing "=": {kv}\n')
            sys.exit(1)
        k, v = kv.split('=', 1)
        try:
            target_counts[k.strip()] = int(v.strip())
        except ValueError:
            sys.stderr.write(f'[FATAL] OPSD_TARGET_COUNTS N not int: {kv}\n')
            sys.exit(1)

if not TARGET_CLASSES:
    sys.stderr.write('[FATAL] TARGET_CLASSES empty after parse\n')
    sys.exit(1)

def pick_label(d):
    for k in ('label', 'label_str_origin'):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    for m in reversed(d.get('messages', []) or []):
        if m.get('role') == 'assistant' and isinstance(m.get('content'), str) and m['content']:
            return m['content']
    return None

# Pass 1: 按类别装桶
buckets = {}
with open(SRC) as fi:
    for line in fi:
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        lbl = pick_label(d)
        if not lbl:
            continue
        buckets.setdefault(lbl, []).append(line)

n_orig = {k: len(v) for k, v in buckets.items()}

for tc in TARGET_CLASSES:
    if tc not in buckets:
        sys.stderr.write(f'[FATAL] TARGET_CLASS={tc} not found in dataset. Available: {list(buckets.keys())}\n')
        sys.exit(1)

# 计算每个 target 类扩容后目标数量
target_new_counts = {}
if target_counts:
    # 【A6】per-class 绝对数：直接读；未在 target_counts 里但在 TARGET_CLASSES 的类跳过
    for tc in TARGET_CLASSES:
        if tc in target_counts:
            target_new_counts[tc] = target_counts[tc]
    if not target_new_counts:
        sys.stderr.write(f'[FATAL] OPSD_TARGET_COUNTS 里没有任何 TARGET_CLASSES 里的类\n')
        sys.exit(1)
else:
    # 【A5 回退】"每类都到 TARGET_RATIO" 模式
    if TARGET_RATIO >= 0.999:
        sys.stderr.write(f'[FATAL] TARGET_RATIO={TARGET_RATIO} too high (must < 1.0)\n')
        sys.exit(1)
    if TARGET_RATIO * len(TARGET_CLASSES) >= 0.95:
        sys.stderr.write(f'[FATAL] TARGET_RATIO * K = {TARGET_RATIO}*{len(TARGET_CLASSES)} >= 0.95, no room for majority classes\n')
        sys.exit(1)
    n_other = sum(v for k, v in n_orig.items() if k not in TARGET_CLASSES)
    K = len(TARGET_CLASSES)
    n_new_each = int(round(TARGET_RATIO * n_other / (1.0 - K * TARGET_RATIO)))
    for tc in TARGET_CLASSES:
        target_new_counts[tc] = n_new_each

target_finals = {}
for tc, n_target in target_new_counts.items():
    pool = buckets[tc]
    n_orig_tc = len(pool)
    if n_target <= n_orig_tc:
        target_finals[tc] = list(pool)  # 目标数小于当前，原样保留（不 truncate）
    else:
        reps = n_target // n_orig_tc
        remainder = n_target - reps * n_orig_tc
        target_finals[tc] = pool * reps + random.sample(pool, remainder)

# 输出：非 target 类原样 + 每个 target 类扩容后池子，全部 shuffle
lines_out = []
for k, v in buckets.items():
    if k in target_finals:
        lines_out.extend(target_finals[k])
    else:
        lines_out.extend(v)
random.shuffle(lines_out)

with open(DST, 'w') as fo:
    for line in lines_out:
        fo.write(line + '\n')

# 打印分布信息
n_final = {k: (len(target_finals[k]) if k in target_finals else len(v)) for k, v in buckets.items()}
total = sum(n_final.values())
mode_label = 'counts (per-class abs)' if target_counts else f'ratio ({TARGET_RATIO} each)'
print(f'[OVERSAMPLE] src={SRC}')
print(f'[OVERSAMPLE] mode={mode_label}  target_classes={list(target_new_counts.keys())}')
print(f'[OVERSAMPLE] before: total={sum(n_orig.values())} ' + ', '.join(f'{k}={v}({v/sum(n_orig.values())*100:.1f}%)' for k, v in n_orig.items()))
print(f'[OVERSAMPLE] after : total={total} ' + ', '.join(f'{k}={v}({v/total*100:.1f}%)' for k, v in n_final.items()))
for tc, n_target in target_new_counts.items():
    n0 = len(buckets[tc]); n1 = len(target_finals[tc])
    print(f'[OVERSAMPLE]   {tc}: unique={n0} -> {n1} ({n1/n0:.2f}x, target={n_target})')
print(f'[OVERSAMPLE] -> {DST}')
PYEOF
    else
        echo "[INFO] 复用已存在的 multi-target-class oversampled 训练集: $TRAIN_JSONL_TARGETED"
    fi
    echo "[INFO] TRAIN_JSONL 切换：$TRAIN_JSONL -> $TRAIN_JSONL_TARGETED"
    TRAIN_JSONL="$TRAIN_JSONL_TARGETED"
fi
# 沿用 grpo 脚本：用 split_dataset_ratio 切 1% 作为 eval（OPSD 评估只看 ce loss/rouge，
# 不需要单独维护 val 文件）
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- 教师 hint 文案 ----------------
# 【v13 事故复盘 · hint 模式选择】
# v13 用 guided（"提供分类框架，不泄漏答案"）本意是让教师给出信息量更丰富的
# 软分布——但实测 stepaudio 上 guided 会引发严重的 song 偏置：
#   1) hint 显式列出 5 个类别名，song 位于末尾，教师注意力有末位锚定倾向；
#   2) hint 反复强调 "vocal characteristics (speech vs singing)" / "melodic content"，
#      从语义上把教师推向音乐/歌唱侧；
#   3) 教师被迫在 hint 影响下产出错向软分布，beta=0.1 (近 forward-KL) 又强制
#      学生覆盖教师所有模态 → 4000 步累积成 "song 优先" 灾难。
#   → v13-20260701-183236 checkpoint-1000/2000 accuracy 双双 ~50%，song FP > 3000。
#
# direct 模式（教师看到 ground-truth 标签）：教师软分布 ≈ one-hot(gt)，OPSD 退化为
# "软标签 SFT"。理论边际收益比 guided 小，但**方向绝对正确**、不受 hint 语义偏差
# 干扰、极其稳定。在 SFT 起点已经 97.83% 的场景下，这才是安全的精修方式。
# 【路径 A 决策】默认切 direct，配合 beta=1.0 (reverse-KL) + sft_alpha=1.0，风险最低。
#
# 如要重新尝试 guided，请务必：
#   a) 重写 hint（去掉类别名列表 & 消除 singing/melodic 等偏向词）
#   b) 挂真实分类 accuracy 评估回调，避免 eval_loss 假象；
#   c) 缩短训练步数，任何早期 accuracy 掉头立即停止。
OPSD_HINT_MODE=${OPSD_HINT_MODE:-rank}
if [ "$OPSD_HINT_MODE" = "rank" ]; then
    # 【路径 A3 默认】多选题排序式 hint：让教师自主输出对整个类别空间的相对置信度分布。
    # 与 direct 的核心差异：不告诉教师答案（不含 {LABEL}），教师和学生看到的信息量相当，
    # 只是教师被明确要求"按类别列表评估相对可能性并输出最合理的一个"，从而在 top-k
    # logits 里保留 5 个 candidate class 的结构化排序。这才是 OPSD 论文原意——
    # 教师提供"整个类别空间的软分布"作为蒸馏目标，而不是硬 one-hot(gt)。
    #
    # 与 guided 的核心差异：guided (v13) 强调"vocal characteristics / instrumental /
    # background noise / sexual audio cues"这些语义线索，会让教师软分布被 hint 语义
    # 偏差引导（如 song 类被强化）；rank 则是纯"要求教师给出排序"的结构性 prompt，
    # 不注入任何类别语义偏向，教师软分布只反映音频本身的声学特征在 5 类上的分布。
    #
    # 教师看到 hint 后的期望行为（论文预期，需要通过 completion log 验证）：
    #   - top-1 概率 ≈ 0.6~0.9（教师本身分类能力对应的置信度）
    #   - top-2/3 概率 ≈ 0.05~0.2（相邻 class 的合理不确定性）
    #   - top-4/5 概率 ≈ 0.001~0.03（明显不匹配的 class）
    #   - <tts_*>/<audio_*> 概率 ≈ 0（rank 明确要求输出 5 类之一）
    # 若实际观察到 top-1 ≈ 1.0（塌陷成 one-hot），则 rank 模式失败，回退 direct/guided。
    OPSD_TEACHER_HINT_TEMPLATE=${OPSD_TEACHER_HINT_TEMPLATE:-$'\n\n(Teacher-only ranking task: for the audio above, rank the 7 candidates {surprise, anger, neutral, joy, sadness, fear, disgust} by likelihood based purely on the vocal / emotional characteristics of the speaker — do NOT reveal the ranking, but pick the single most likely candidate as your answer following the required answer format. Assess each candidate holistically without preferring any category by name.)'}
elif [ "$OPSD_HINT_MODE" = "direct" ]; then
    # 【路径 A2 备选】把 ground-truth 作为参考告诉教师。
    # 【v14 事故教训 · hint 措辞修正】v14 使用的旧模板：
    #   'the ground-truth label ... is "{LABEL}". Now output exactly that label
    #    following the required answer format.'
    # 措辞过于"祈使"——"Now output exactly that label" 让教师软分布几乎变成硬 one-hot(gt)，
    # 其它 4 类概率被压到 ≈ 0，蒸馏信号退化为 hard-label SFT，OPSD 的类间相对结构
    # 完全丢失。学生在 val 边界样本上一旦不确定就直接切到 speech 兜底 → noise/porn recall=0。
    #
    # 路径 A2 改成"参考式"措辞：
    #   - 只陈述参考答案（reference answer），不使用祈使句；
    #   - 保留 "you MAY use it as reference but still assess based on the audio"，
    #     让教师保留自身对音频信号的判断，输出**软**分布而非硬 one-hot；
    #   - 教师最终仍会对 gt 类别给最高概率（因为它是 SFT 起点 97.83% 的模型 + 有参考），
    #     但相邻边界类别（如 song vs music, noise vs speech）会保留合理的 0.05-0.2 概率，
    #     这些正是 OPSD 想要蒸馏的"类间相对结构"。
    OPSD_TEACHER_HINT_TEMPLATE=${OPSD_TEACHER_HINT_TEMPLATE:-$'\n\n(Reference for teacher only, do NOT echo or mention: the reference label for this audio is "{LABEL}". You may use it as a reference, but still assess based on the audio itself and output the label you find most plausible in the required answer format.)'}
else
    # guided：只给"分类框架"，不泄漏答案。⚠ v13 已证实此模式在 stepaudio 上会引发
    # song 偏置崩坏，非有充分把握不要使用；如要使用请重写 hint 消除类别列表锚定。
    OPSD_TEACHER_HINT_TEMPLATE=${OPSD_TEACHER_HINT_TEMPLATE:-$'\n\n(Teacher hint, do NOT echo: focus on distinguishing among {surprise, anger, neutral, joy, sadness, fear, disgust}. Consider prosody, pitch variance, energy, speech rate, and paralinguistic cues (breathing / voice quality) to infer the speaker\'s emotional state. Output exactly one label following the required answer format.)'}
fi

# ---------------- 派生带 teacher_prompt 的训练 jsonl ----------------
# OPSD 要求每条样本携带 teacher_prompt 字段（详见 swift/rlhf_trainers/gkd_loss.py
# 中的 build_opsd_teacher_data）。原始 train.jsonl 里没有，这一步在内存里
# 扫一遍生成 train.opsd.jsonl。
#
# 派生逻辑：
#   - 取最后一条 user 的 content（含 <audio> 占位符），原样保留；
#   - 在末尾追加 hint，把 ground-truth 标签暴露给"教师视角"；
#   - 学生视角（messages.user.content）保持原文，不变。
# 派生文件名按 hint 模式隔离，避免"改了 hint 模式却复用旧派生 jsonl"的隐蔽 bug。
TRAIN_OPSD_JSONL=${TRAIN_OPSD_JSONL:-"$DATA_DIR/train.opsd.${OPSD_HINT_MODE}.jsonl"}
# 记录当前 hint 模板 hash，若变化则强制重建（避免只改环境变量不重建导致的静默错误）
_HINT_HASH=$(printf '%s' "$OPSD_TEACHER_HINT_TEMPLATE" | md5sum | awk '{print $1}')
_HINT_HASH_FILE="${TRAIN_OPSD_JSONL}.hint.md5"
OPSD_REBUILD=${OPSD_REBUILD:-auto}  # auto/1/0：auto = 输出不存在 / 源更新 / hint 变化时重建

_need_rebuild=0
case "$OPSD_REBUILD" in
    1) _need_rebuild=1 ;;
    0) _need_rebuild=0 ;;
    auto)
        if [ ! -f "$TRAIN_OPSD_JSONL" ]; then
            _need_rebuild=1
        elif [ "$TRAIN_JSONL" -nt "$TRAIN_OPSD_JSONL" ]; then
            _need_rebuild=1
        elif [ ! -f "$_HINT_HASH_FILE" ] || [ "$(cat "$_HINT_HASH_FILE" 2>/dev/null)" != "$_HINT_HASH" ]; then
            echo "[INFO] 检测到 OPSD hint 模板变更，强制重建派生 jsonl"
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
    # 写入 hint 指纹，供下次 auto 判断
    printf '%s' "$_HINT_HASH" > "$_HINT_HASH_FILE"
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
echo "[INFO] BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH MAX_STEPS=$MAX_STEPS"
echo "[INFO] gradient_checkpointing=$GC_FLAG log_completions=$LOG_COMPLETIONS"
echo "[INFO] OPSD beta=$BETA lmbda=$LMBDA temperature=$TEMPERATURE max_completion=$MAX_COMPLETION_LENGTH"
echo "[INFO] OPSD gkd_logits_topk=${GKD_LOGITS_TOPK:-<full vocab>} sft_alpha=$SFT_ALPHA"
echo "[INFO] OPSD hint_mode=$OPSD_HINT_MODE (rank=多选题式排序[A3默认], direct=直接给答案[A2], guided=只给分类框架[v13崩坏])"
echo "[INFO] OPSD target_oversample=$OPSD_TARGET_OVERSAMPLE target_classes=${OPSD_TARGET_CLASSES:-$OPSD_TARGET_CLASS} target_ratio=$OPSD_TARGET_RATIO"
if [ "$LMBDA" != "0.0" ] && [ "$LMBDA" != "0" ]; then
    echo "[INFO] 【路径 A6】lmbda=$LMBDA (on-policy) + beta=$BETA (中弱 KL) + lr=$LEARNING_RATE + max_steps=$MAX_STEPS + sft_alpha=$SFT_ALPHA"
    echo "[INFO]              hint 已修为 MELD 7 类；rank.jsonl hash 变化会自动重建"
    echo "[INFO]              per-class oversample counts=${OPSD_TARGET_COUNTS:-<disabled, using ratio>}；gkd_logits_topk=$GKD_LOGITS_TOPK 已对齐 7 类"
    echo "[INFO]              每步学生自采样，训练结束后会扫描前 $OPSD_ONPOLICY_DRIFT_ABORT_STEPS 行 completions.jsonl"
    echo "[INFO]              检查 <audio_*>/<tts_*> 漂移；若发现漂移已到 macro-F1 重陛系统性奇差阶段，建议中断"
else
    echo "[INFO] 【路径 A3 fallback】lmbda=0 (offline) —— KL 项自蒸馏梯度近零，训练等价于 (1-beta)× CE"
    echo "[INFO]              此时 completions.jsonl 不会产出，无法监控漂移；若想真跑 OPSD 请设 LMBDA=1.0"
fi
echo "[INFO] RESUME_CHECKPOINT='${RESUME_CHECKPOINT}'  RESUME_ONLY_MODEL='${RESUME_ONLY_MODEL}'"
if [ "$OPSD_HINT_MODE" = "rank" ]; then
    echo "[INFO] OPSD_HINT_MODE=rank (路径 A3 · 研究方案)：教师被要求对 5 类做排序，"
    echo "       输出结构化软分布（top-1 ≈ 0.6~0.9 / top-2/3 ≈ 0.05~0.2）而非 one-hot(gt)。"
    echo "       预期收益：类间相对结构信号真正被蒸馏；风险：教师若在 hint 影响下产出"
    echo "       错向排序，学生会被拉偏——需要每 100 步跑一次真实分类评测确认 accuracy 不崩。"
fi
if [ "$OPSD_HINT_MODE" = "guided" ]; then
    echo "[WARN] OPSD_HINT_MODE=guided：v13-20260701-183236 已证实此模式在 stepaudio 上"
    echo "       引发严重 song 偏置崩坏 (accuracy 97.83% -> 50.13%, song FP=3120)。"
    echo "       如需继续，请务必重写 hint 消除类别名列表 & singing/melodic 偏向词。"
fi
if [ "$OPSD_HINT_MODE" = "direct" ]; then
    echo "[INFO] OPSD_HINT_MODE=direct 复盘：v14-20260702-112351 (LR=1e-6/beta=1.0/topk=8/sft_alpha=1.0/steps=500)"
    echo "       从 SFT 起点 97.83% 崩到 56.06% (ckpt-50 已崩)，noise/porn 完全归零，"
    echo "       song 沦为兜底类，2.4% 样本吐 <tts_end>/<audio_XXX>。"
    echo "       路径 A2 已把默认超参压到 LR=1e-7 / beta=0.5 / topk=5 / sft_alpha=2.0 /"
    echo "       max_steps=200 / max_grad_norm=0.5 / warmup_ratio=0.1 / balanced 数据源。"
    echo "       仍强烈建议每个 ckpt 都跑一次真实分类评测，一旦 accuracy < 起点 97% 立即停止。"
fi
if [[ "$MODEL_PATH" == */Step-Audio-2-mini ]]; then
    echo "[WARN] MODEL_PATH 指向 base 模型：base 未做过分类 SFT，OPSD 精修意义不大。"
    echo "       路径 A 强烈建议指向 SFT 后的 checkpoint，例如："
    echo "         export MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234"
    if [ "$LMBDA" != "0.0" ] && [ "$LMBDA" != "0" ]; then
        echo "[WARN] 且 LMBDA>0：base 模型 on-policy 生成极易吐 <audio_XXX> 污染教师，"
        echo "       必须强制 LMBDA=0 或先做 SFT。"
    fi
fi
# 【v15 事故防线】显式黑名单：checkpoint-400-merged 是 v15-20260702-150039 事故里
# 由本脚本自动 merge 产出的**破损目录**（tokenizer 关键文件缺失，加载后 lm_head 对
# label token 失效）。此后再遇到该路径直接 abort，避免重蹈覆辙。
if [[ "$MODEL_PATH" == */checkpoint-400-merged ]]; then
    echo "[FATAL] 检测到 MODEL_PATH 指向 checkpoint-400-merged，该目录已被证实为破损"
    echo "        起点（v15-20260702-150039 事故复盘：val accuracy 仅 ~57%，"
    echo "        noise/porn F1=0，156+ 条无效响应）。"
    echo "        推荐替换为已验证的 checkpoint-1600-merged (accuracy=97.83%)："
    echo "          export MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234"
    echo "        或直接 unset MODEL_PATH 使用脚本默认值。"
    exit 16
fi
# 注：v13 事故已证实 stepaudio 分类任务的漂移根因是 hint 语义偏差，而非数据分布失衡；
# 因此路径 A 默认使用 train.jsonl (56302 行真实分布)，不再对失衡数据发出警告。

# ---------------- GPU 占用前置检查（与 GRPO 脚本对齐，双重判据） ----------------
# 目的：拦截"两个训练同时抢同一批 GPU"这种坑（GRPO v3-20260701-155217 事故复盘：
#      step_time 从 40s 恶化到 240s）。检查两层：
#   1) 目标 GPU 上的 already-used memory > GPU_PREALLOC_GUARD_MB 视为忙
#   2) 目标 GPU 上存在 python/swift/torchrun/deepspeed 训练进程 视为忙（更精准）
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
            echo "        OPSD 教师/学生双 forward + 可能的 on-policy generate 显存压力大，"
            echo "        与其它训练共卡会显著劣化 step_time（历史坑）。"
            echo "        处置建议："
            echo "          1) 找到旧训练进程后 kill：pgrep -af '(swift|torchrun).*rlhf' | awk '{print \$1}' | xargs -r kill"
            echo "          2) 或显式换空闲卡：export CUDA_VISIBLE_DEVICES=<free_ids>"
            echo "          3) 或明确要共享（不推荐）：FORCE=1 bash $(basename \"$0\")"
            exit 11
        fi
        echo "[INFO] GPU 占用检查通过 (阈值 ${GPU_PREALLOC_GUARD_MB} MiB / 卡, 且无其它 python/swift 训练进程)"
    else
        echo "[WARN] 已通过 GPU_PREALLOC_SKIP=1 / FORCE=1 跳过 GPU 占用检查，可能与其它训练抢卡。"
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

# ---------------- LoRA adapter checkpoint 自动 merge ----------------
# 场景：MODEL_PATH 指向 SFT 后保存的 checkpoint-XXXX 目录，但该目录只是 LoRA adapter
#       （只含 adapter_config.json / adapter_model.safetensors，没有 config.json /
#        tokenizer 文件 / base 权重）。此时直接 --model 传该目录会踩坑：
#          File "transformers/models/bert/tokenization_bert.py", line 114, in __init__
#              if not os.path.isfile(vocab_file):
#          TypeError: stat: path should be string, ..., not NoneType
#       原因是 AutoTokenizer 在 adapter 目录里找不到 tokenizer 相关文件。
#
# 处理策略：
#   1) 检测 MODEL_PATH 是否是纯 adapter 目录（存在 adapter_config.json 且不存在 config.json）；
#   2) 若已存在同级 <ckpt_name>-merged 目录 → 直接切换到它；
#   3) 否则调用 `swift export --adapters MODEL_PATH --merge_lora true`，
#      swift 会读取 adapter_config.json 中的 base_model_name_or_path 加载 base
#      并合并 LoRA 权重，输出到 <ckpt_dir>/<ckpt_name>-merged；
#   4) 切换 MODEL_PATH 指向 merged 目录（含完整 base 权重 + tokenizer + processor）。
#
# 触发条件被压到 SWIFT_CMD 探测之后：merge 需要 python 环境和 swift CLI。
# OPSD_TEACHER_MODE=fixed 场景下 --teacher_model 也会自动跟随新的 MODEL_PATH。
_ensure_full_model_dir() {
    local mdir="$1"
    if [ ! -d "$mdir" ]; then
        # 非本地目录（HF hub id 等），保持原样让 swift 自行处理
        echo "$mdir"
        return 0
    fi
    if [ -f "$mdir/config.json" ]; then
        # 已是完整模型目录
        echo "$mdir"
        return 0
    fi
    if [ ! -f "$mdir/adapter_config.json" ]; then
        # 既不是完整模型又不是 adapter，直接透传，让 swift 自己报错
        echo "$mdir"
        return 0
    fi

    # 到这里必然是 LoRA adapter 目录
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
    # 单卡 CPU/GPU 都可 merge，这里保留 CUDA_VISIBLE_DEVICES 让 swift 自行选择；
    # merge 过程只做一次前向权重加载 + 加法，10-20s 内完成。
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

    # 【v15 事故防线】v15-20260702-150039 事故里 checkpoint-400-merged 目录只包含
    # config.json + tokenizer.json + 权重文件，缺失 added_tokens.json / merges.txt /
    # special_tokens_map.json / vocab.json，加载后 lm_head 对 label token 失效，导致
    # 训练全程 accuracy 卡在 57%、noise/porn F1=0、156+ 条无效响应。这里对新 merge
    # 出的目录做完整性检查，任一 tokenizer 关键文件缺失就直接 abort。
    local _missing=()
    for _f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json; do
        if [ ! -f "$merged/$_f" ]; then
            _missing+=("$_f")
        fi
    done
    if [ ${#_missing[@]} -gt 0 ]; then
        echo "[FATAL] merge 输出目录 tokenizer 文件不完整，缺失: ${_missing[*]}" 1>&2
        echo "        目录: $merged" 1>&2
        echo "        这通常意味着 swift export 使用了不匹配的 transformers 版本，" 1>&2
        echo "        或 base_model_name_or_path 指向的模型缺失原始 tokenizer 文件。" 1>&2
        echo "        建议：" 1>&2
        echo "          1) rm -rf $merged" 1>&2
        echo "          2) 换用 conda env-3.12.11 (transformers==4.53.3) 重新 merge：" 1>&2
        echo "             /data/miniconda3/envs/env-3.12.11/bin/python -m swift.cli.main export \\" 1>&2
        echo "               --adapters $mdir_abs --merge_lora true --output_dir $merged" 1>&2
        echo "          3) 或直接换用已验证的完整 merged 目录（例如 checkpoint-1600-merged）。" 1>&2
        return 1
    fi
    echo "[INFO] merge 输出目录 tokenizer 完整性检查通过: $merged" 1>&2
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

# ---------------- 起点模型 sanity check（v15 事故防线） ----------------
# 【v15 事故复盘】v15-20260702-150039 训练 200 步后 checkpoint-20/40/80/200 accuracy
# 全部只有 ~57%，noise/porn F1=0，156+ 条 <tts_end>/<audio_XXXX> 无效响应；回溯发现
# 根因是**起点模型本身就已损坏**（自动 merge 出的 checkpoint-400-merged 目录 tokenizer
# 缺文件），OPSD 200 步只是"忠实维持"了这个损坏的起点，loss 从 0.16 起步且从未真正下降。
#
# 【MELD 版关键差异】
# 本脚本训练的是 MELD 7 分类 (surprise/anger/neutral/joy/sadness/fear/disgust)，
# 不是 OPSD 通用五类 (speech/music/noise/porn/song)。因此 v15 时代 sanity 用的
# OPSD 数据集 (data/val.small.jsonl) + noise/porn recall 阈值在 MELD 场景下会
# 100% 误报：任何 MELD 模型都不认识 noise/porn 这两个词，recall 必然为 0，且
# 五分类整体 accuracy 只会围绕 speech recall (~0.6) 打转，永远拿不到 0.85。
# 因此 MELD 版把 sanity 全部指向 MELD 数据 + MELD 关注类 + MELD 水位：
#   * VAL_JSONL 默认 data_meld/val.small.jsonl (1108 条, dev split)；
#   * OPSD_SANITY_FOCUS_CLASSES 关注类，逗号分隔；默认 fear (即 OPSD_TARGET_CLASS,
#     MELD 里 recall 最低 ~0.22 的美回归目标)；
#   * 阈值参考 v7-checkpoint-150 test 集: accuracy=0.585 / fear_recall=0.22，
#     并留 15pp 缓冲：soft acc>=0.50, hard acc>=0.35, focus recall>=0.10。
# 三级阈值：
#   HARD 阈值（低于此直接 abort）：accuracy 低到明显是"起点崩坏"级别
#     accuracy      >= OPSD_SANITY_HARD_ACCURACY      (MELD 默认 0.35)
#   SOFT 阈值（低于此仅 warn，不 abort）：期望的正常水位
#     accuracy      >= OPSD_MIN_ACCURACY              (MELD 默认 0.50)
#     focus recall  >= OPSD_MIN_FOCUS_RECALL          (MELD 默认 0.10, 每个关注类)
#     n_invalid_response <= OPSD_MAX_INVALID_RESPONSE (默认 50)
# 若 accuracy > HARD 但 SOFT 未达标 & 关注类 recall=0（其它类正常），判为"疑似
# 推理链路问题"（例如 MAX_NEW_TOKENS 截断、prompt 模板不一致），仅 warn 放行。
# 【注意】只对"关注类"触发 SUSPECT, 不再对"任意单类"触发——MELD 上 minority 类
# (fear/disgust/sadness support 只有几十到二百多) 起点 recall=0 是模型倾向问题,
# 与推理链路无关, 不该误报为链路 bug。
#
# 【历史优化 A+B+C+D】
#   A) 缓存：以 MODEL_PATH mtime + val.jsonl md5 做 cache key，命中直接读旧结果，
#      避免每次改超参重跑训练都要重新推理 2-3 分钟；OPSD_SANITY_FORCE_RERUN=1 强制重跑。
#   B) 诊断：FAIL 时输出 per-class precision/recall/f1 完整表 & 智能判定链路问题。
#   C) 三级阈值：HARD/SOFT/单类为 0 时的分级处置，避免单点极端值卡住训练。
#   D) MAX_NEW_TOKENS：默认 2048（与 run_inference_meld.sh 一致），避免 MELD 8 字符
#      长标签（surprise/sadness/disgust）被截；也对齐 UltraEval-Audio 容许
#      模型自由生成完整词后字面匹配的 55.47 baseline 口径。
#
# 关闭方式：
#   OPSD_SKIP_SANITY_CHECK=1  bash run_train_opsd_meld.sh   # 完全跳过
#   OPSD_SANITY_ONLY_WARN=1   bash run_train_opsd_meld.sh   # 只 warn 不 abort
#   OPSD_SANITY_FORCE_RERUN=1 bash run_train_opsd_meld.sh   # 忽略缓存重跑
#   FORCE=1                   bash run_train_opsd_meld.sh   # 等价于 SKIP=1
# 若确实希望跳过，务必知道 v15 事故就是因为跳过了这类验证，损失了数十分钟。
OPSD_SKIP_SANITY_CHECK=${OPSD_SKIP_SANITY_CHECK:-${FORCE:-0}}
OPSD_SANITY_ONLY_WARN=${OPSD_SANITY_ONLY_WARN:-0}
OPSD_SANITY_FORCE_RERUN=${OPSD_SANITY_FORCE_RERUN:-0}
# 【MELD 阈值】相比 OPSD 五类 (SFT 后 accuracy ~0.97) 大幅下调, 参考 v7-ckpt150 test:
# accuracy 0.585 / fear R 0.22 → soft 0.50, hard 0.35, focus R 0.10 (15pp 缓冲)。
OPSD_MIN_ACCURACY=${OPSD_MIN_ACCURACY:-0.50}
OPSD_SANITY_HARD_ACCURACY=${OPSD_SANITY_HARD_ACCURACY:-0.35}
# MELD 关注类: 逗号分隔, 每个类的 recall 都必须 >= OPSD_MIN_FOCUS_RECALL。
# 默认取 OPSD_TARGET_CLASS (fear), 与训练侧 target oversample 目标一致；
# 若想同时监控 disgust/sadness, 例如: OPSD_SANITY_FOCUS_CLASSES=fear,disgust
OPSD_SANITY_FOCUS_CLASSES=${OPSD_SANITY_FOCUS_CLASSES:-${OPSD_TARGET_CLASSES:-${OPSD_TARGET_CLASS:-fear}}}
OPSD_MIN_FOCUS_RECALL=${OPSD_MIN_FOCUS_RECALL:-0.10}
OPSD_MAX_INVALID_RESPONSE=${OPSD_MAX_INVALID_RESPONSE:-50}
OPSD_SANITY_MAX_NEW_TOKENS=${OPSD_SANITY_MAX_NEW_TOKENS:-2048}
OPSD_SANITY_VAL_JSONL=${OPSD_SANITY_VAL_JSONL:-"$SCRIPT_DIR/data_meld/val.small.jsonl"}
OPSD_SANITY_CACHE_DIR=${OPSD_SANITY_CACHE_DIR:-"$SCRIPT_DIR/infer_results/_sanity_cache"}

if [ "$OPSD_SKIP_SANITY_CHECK" = "1" ]; then
    echo "[WARN] 已通过 OPSD_SKIP_SANITY_CHECK=1 / FORCE=1 跳过起点 sanity check。"
    echo "       历史事故 v15-20260702-150039 就是因为起点模型本身已损坏而未察觉，"
    echo "       浪费了数十分钟训练时间；除非你已在其它渠道确认起点分类准确率 > 90%，"
    echo "       否则强烈建议不要跳过。"
elif [ ! -f "$OPSD_SANITY_VAL_JSONL" ]; then
    echo "[WARN] sanity check 数据文件不存在，跳过：$OPSD_SANITY_VAL_JSONL"
    echo "       如需启用，请先准备 val.small.jsonl 或指定 OPSD_SANITY_VAL_JSONL。"
else
    # ---- (A) 缓存机制：以 MODEL_PATH mtime + val md5 + 关键阈值参数为 key ----
    mkdir -p "$OPSD_SANITY_CACHE_DIR"
    _sanity_val_md5=$(md5sum "$OPSD_SANITY_VAL_JSONL" 2>/dev/null | awk '{print $1}')
    _sanity_val_md5=${_sanity_val_md5:-nomd5}
    # 用整个 MODEL_PATH 目录树最新 mtime 作为模型指纹（不依赖 stat 具体格式）
    _sanity_model_mtime=$(find "$MODEL_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    _sanity_model_mtime=${_sanity_model_mtime:-nomtime}
    # 【MELD 版】cache key 额外编入 "meld7_uea" 语义标记（UltraEval-Audio 对齐）：
    # 与旧 OPSD 五类 cache / 旧 MELD non-UEA cache 分离，修改类别列表 / target /
    # 调用的下游推理脚本（run_inference.sh -> run_inference_meld.sh）时，cache 会自动
    # 失效，避免命中错语义的旧结果。
    _sanity_classes_tag="meld7_uea"
    _sanity_key_raw="$MODEL_PATH|$_sanity_model_mtime|$_sanity_val_md5|mnt=$OPSD_SANITY_MAX_NEW_TOKENS|cls=$_sanity_classes_tag"
    _sanity_key=$(echo -n "$_sanity_key_raw" | md5sum | awk '{print $1}')
    _sanity_cache_summary="$OPSD_SANITY_CACHE_DIR/${_sanity_key}.json"
    _sanity_cache_meta="$OPSD_SANITY_CACHE_DIR/${_sanity_key}.meta"

    _sanity_gpus=${OPSD_SANITY_GPUS:-$CUDA_VISIBLE_DEVICES}
    _sanity_nproc=${OPSD_SANITY_NPROC:-$(echo "$_sanity_gpus" | awk -F',' '{print NF}')}
    [ -z "$_sanity_nproc" ] && _sanity_nproc=1

    _sanity_from_cache=0
    if [ "$OPSD_SANITY_FORCE_RERUN" != "1" ] && [ -f "$_sanity_cache_summary" ]; then
        echo ""
        echo "===================================================================="
        echo "[INFO] 起点模型 sanity check 命中缓存（跳过推理+评估）"
        echo "       cache_key   = $_sanity_key"
        echo "       cache_file  = $_sanity_cache_summary"
        [ -f "$_sanity_cache_meta" ] && echo "       cache_meta  = $(cat "$_sanity_cache_meta")"
        echo "       如需强制重跑：OPSD_SANITY_FORCE_RERUN=1 bash $(basename "$0")"
        echo "===================================================================="
        _sanity_summary="$_sanity_cache_summary"
        _sanity_eval_dir="$OPSD_SANITY_CACHE_DIR"
        _sanity_from_cache=1
    else
        _sanity_ts=$(date +%Y%m%d_%H%M%S)
        _sanity_tag="$(basename "$MODEL_PATH")_sanity_${_sanity_ts}"
        _sanity_result="$SCRIPT_DIR/infer_results/result_${_sanity_tag}.jsonl"
        _sanity_eval_dir="$SCRIPT_DIR/infer_results/eval_result_${_sanity_tag}"

        echo ""
        echo "===================================================================="
        echo "[INFO] 起点模型 sanity check 开始"
        echo "       MODEL_PATH        = $MODEL_PATH"
        echo "       VAL_JSONL         = $OPSD_SANITY_VAL_JSONL ($(wc -l < "$OPSD_SANITY_VAL_JSONL") 条)"
        echo "       GPUS              = $_sanity_gpus (nproc=$_sanity_nproc)"
        echo "       MAX_NEW_TOKENS    = $OPSD_SANITY_MAX_NEW_TOKENS  (默认 2048，对齐 UltraEval-Audio + 避免长标签截断)"
        echo "       cache_key         = $_sanity_key"
        echo "       hard threshold    : accuracy>=$OPSD_SANITY_HARD_ACCURACY (低于此 abort, MELD 起点崩坏线)"
        echo "       soft thresholds   : accuracy>=$OPSD_MIN_ACCURACY,"
        echo "                           focus_R>=$OPSD_MIN_FOCUS_RECALL (每个关注类, focus=$OPSD_SANITY_FOCUS_CLASSES),"
        echo "                           invalid<=$OPSD_MAX_INVALID_RESPONSE"
        echo "       如需跳过：OPSD_SKIP_SANITY_CHECK=1  bash $(basename "$0")"
        echo "       只警告  ：OPSD_SANITY_ONLY_WARN=1   bash $(basename "$0")"
        echo "===================================================================="

        # MELD 版 sanity 直接使用 MELD 专用下游脚本（run_inference_meld.sh / run_eval_meld.sh），
        # 它们内置 MELD 7 类默认 + UltraEval-Audio 对齐，无需再显式传 CLASSES / HEALTH_CLASSES。
        # focus 列表首个作为 TARGET_CLASS（threshold sweep 的正类），与 OPSD_TARGET_CLASS 通常一致（fear）。
        _sanity_target_class="${OPSD_SANITY_FOCUS_CLASSES%%,*}"
        _sanity_target_class="${_sanity_target_class:-$OPSD_TARGET_CLASS}"

        # 【MELD 版关键】下游必须用 MELD 专用推理/评估脚本，否则会塑陷为 100% neutral：
        #   run_inference_meld.sh 相比 run_inference.sh 多 3 项 UltraEval-Audio 对齐 (55.47 baseline 关键)：
        #     (A) 显式注入 --system "You are a helpful assistant." —— 缺失时 acc 会从 55.47 掉20pp；
        #     (B) MAX_NEW_TOKENS=2048 —— 避免 MELD 8 字符长标签（surprise/sadness/disgust）被截；
        #     (C) USE_CACHE=false —— 关掉 KV cache 导致的输出漂移（通过 sitecustomize shim monkey-patch）。
        #   上一版 sanity 调 run_inference.sh 拿到 accuracy=0.423 / 100% neutral，就是因为缺少这 3 项。
        #   run_eval_meld.sh 内置 MELD 7 类默认，使用 eval_classification_meld.py（直接兼容 threshold sweep）。
        # 因此下面不再显式传 HEALTH_CLASSES / CLASSES（MELD 专用脚本默认已是 MELD 7 类），
        # 只透传 TARGET_CLASS（从 focus 首类取）以对齐 threshold sweep 目标类。

        # 关闭 set -e 局部区间：sanity check 由我们自己判定退出码
        set +e
        MODEL_PATH="$MODEL_PATH" \
        VAL_JSONL="$OPSD_SANITY_VAL_JSONL" \
        RESULT_PATH="$_sanity_result" \
        NPROC_PER_NODE="$_sanity_nproc" \
        CUDA_VISIBLE_DEVICES="$_sanity_gpus" \
        MAX_NEW_TOKENS="$OPSD_SANITY_MAX_NEW_TOKENS" \
        bash "$SCRIPT_DIR/run_inference_meld.sh"
        _sanity_infer_rc=$?
        set -e
        if [ "$_sanity_infer_rc" -ne 0 ]; then
            echo "[FATAL] sanity check 推理失败 (rc=$_sanity_infer_rc)，请检查 MODEL_PATH 是否可正常加载。"
            exit 17
        fi

        set +e
        RESULT_PATH="$_sanity_result" \
        VAL_JSONL="$OPSD_SANITY_VAL_JSONL" \
        OUTPUT_DIR="$_sanity_eval_dir" \
        TARGET_CLASS="$_sanity_target_class" \
        bash "$SCRIPT_DIR/run_eval_meld.sh"
        _sanity_eval_rc=$?
        set -e
        if [ "$_sanity_eval_rc" -ne 0 ]; then
            echo "[FATAL] sanity check 评估失败 (rc=$_sanity_eval_rc)，请检查 eval_classification_meld.py 是否可用。"
            exit 17
        fi

        _sanity_summary="$_sanity_eval_dir/eval_summary.json"
        if [ ! -f "$_sanity_summary" ]; then
            echo "[FATAL] sanity check 未产出 eval_summary.json：$_sanity_summary"
            exit 17
        fi

        # 写入缓存
        cp -f "$_sanity_summary" "$_sanity_cache_summary" 2>/dev/null || true
        {
            echo "model=$MODEL_PATH"
            echo "model_mtime=$_sanity_model_mtime"
            echo "val=$OPSD_SANITY_VAL_JSONL"
            echo "val_md5=$_sanity_val_md5"
            echo "max_new_tokens=$OPSD_SANITY_MAX_NEW_TOKENS"
            echo "gpus=$_sanity_gpus"
            echo "eval_dir=$_sanity_eval_dir"
            echo "created_at=$(date -Iseconds)"
        } > "$_sanity_cache_meta" 2>/dev/null || true
    fi

    _PY_BIN=${SWIFT_CMD[0]}
    if [ ! -x "$_PY_BIN" ]; then
        _PY_BIN=$(command -v python3 || command -v python)
    fi

    # ---- (B)(C) 三级阈值判定 + 完整诊断信息 (MELD 版：关注类可配置) ----
    set +e
    "$_PY_BIN" - "$_sanity_summary" \
        "$OPSD_MIN_ACCURACY" "$OPSD_SANITY_FOCUS_CLASSES" \
        "$OPSD_MIN_FOCUS_RECALL" "$OPSD_MAX_INVALID_RESPONSE" \
        "$OPSD_SANITY_HARD_ACCURACY" "$OPSD_SANITY_ONLY_WARN" <<'PY'
import json, sys
summary_path  = sys.argv[1]
min_acc       = float(sys.argv[2])
# 关注类列表, 逗号分隔; 每个类的 recall 都必须 >= min_focus_recall
focus_classes = [c.strip() for c in sys.argv[3].split(',') if c.strip()]
min_focus_r   = float(sys.argv[4])
max_invalid   = int(sys.argv[5])
hard_acc      = float(sys.argv[6])
only_warn     = sys.argv[7] == "1"

with open(summary_path) as f:
    d = json.load(f)
report    = d.get('multiclass_report', {})
acc       = float(report.get('accuracy', 0.0))
per       = report.get('per_class', {})
n_invalid = int(d.get('n_invalid_response', 0))

# 关注类若在 per-class 报告里缺失 (例如拼写错或 val 集里没这个类), 直接告警
# 并按 recall=0 处理; 用户可通过 OPSD_SANITY_FOCUS_CLASSES 显式修正。
focus_recalls = {}
for c in focus_classes:
    if c not in per:
        print(f"[SANITY][WARN] focus class {c!r} 不在 per_class 报告里 "
              f"(可选: {sorted(per.keys())}); 请检查 OPSD_SANITY_FOCUS_CLASSES 拼写。",
              file=sys.stderr)
        focus_recalls[c] = 0.0
    else:
        focus_recalls[c] = float(per[c].get('recall', 0.0))

# ---- 打印核心指标 ----
print("[SANITY] ---- summary ----")
print(f"[SANITY] accuracy      = {acc:.4f}   (soft>={min_acc}, hard>={hard_acc})")
for c, r in focus_recalls.items():
    print(f"[SANITY] {c:<9s} recall= {r:.4f}   (soft>={min_focus_r}, focus class)")
print(f"[SANITY] invalid_resp  = {n_invalid}      (soft<={max_invalid})")

# ---- (B) 打印 per-class 完整表 ----
if per:
    print("[SANITY] ---- per-class report ----")
    print(f"[SANITY] {'class':<12s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}")
    for cls, m in per.items():
        p  = float(m.get('precision', 0.0))
        r  = float(m.get('recall', 0.0))
        f1 = float(m.get('f1', m.get('f1-score', 0.0)))
        sp = int(m.get('support', 0))
        marker = ' <-- focus' if cls in focus_classes else ''
        print(f"[SANITY] {cls:<12s} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sp:>10d}{marker}")

# ---- 判定 ----
soft_failed = []
if acc < min_acc:
    soft_failed.append(f"accuracy {acc:.4f} < {min_acc}")
for c, r in focus_recalls.items():
    if r < min_focus_r:
        soft_failed.append(f"{c}_recall {r:.4f} < {min_focus_r}")
if n_invalid > max_invalid:
    soft_failed.append(f"n_invalid_response {n_invalid} > {max_invalid} (词表异常? <tts_end>/<audio_*>)")

# 硬阈值：accuracy 极低 = 起点模型确实崩了
hard_failed = acc < hard_acc

# 智能诊断：仅当**关注类** recall=0 且**其它非关注类**明显在工作 (>0.5) 时，
# 才判为疑似推理链路问题。这样避免 MELD 上 fear/disgust 这类 minority 类天然
# recall=0 被误报为链路 bug (它们本来就是模型难分的类, 不是推理链路问题)。
zero_focus = [c for c, r in focus_recalls.items() if r == 0.0]
nonzero_others = [
    cls for cls, m in per.items()
    if cls not in focus_classes and float(m.get('recall', 0.0)) > 0.5
]
suspect_pipeline = (
    len(zero_focus) > 0
    and len(nonzero_others) > 0
    and acc >= hard_acc
)

if not soft_failed:
    print("[SANITY][PASS] 起点模型可作为 OPSD 训练起点。")
    sys.exit(0)

# soft 未通过：区分 hard fail vs 疑似链路 vs 只是不达 soft
print("", file=sys.stderr)
print("[SANITY] ---- diagnosis ----", file=sys.stderr)
if hard_failed:
    print(f"[SANITY][FAIL:HARD] accuracy {acc:.4f} < 硬阈值 {hard_acc}，起点模型确实**已损坏**。", file=sys.stderr)
    print("[SANITY] 处置：换 checkpoint / 重新 merge / 检查 tokenizer 完整性。", file=sys.stderr)
    print("[SANITY][FAIL] " + "; ".join(soft_failed), file=sys.stderr)
    sys.exit(2)  # rc=2: 硬失败

if suspect_pipeline:
    print(f"[SANITY][SUSPECT] 关注类 recall=0 (类={zero_focus}) 但非关注类正常 (类={nonzero_others})", file=sys.stderr)
    print("[SANITY]           且 accuracy {:.4f} >= 硬阈值 {}，判为**疑似推理链路问题**：".format(acc, hard_acc), file=sys.stderr)
    print("[SANITY]   (a) run_inference_meld.sh 的 MAX_NEW_TOKENS 是否被消费？（本次已设为", file=sys.stderr)
    print("[SANITY]       $OPSD_SANITY_MAX_NEW_TOKENS，若 script 内被硬编码为更小值，会截断长标签词）", file=sys.stderr)
    print("[SANITY]   (b) prompt 模板是否与训练时一致？UltraEval-Audio 对齐需要 --system \"You are a helpful assistant.\" + use_cache=False", file=sys.stderr)
    print("[SANITY]   (c) generation_config 是否引入了 do_sample / temperature 差异？", file=sys.stderr)
    print("[SANITY]   建议：先跑一次 run_inference_meld.sh + run_eval_meld.sh 手动核验；若确认起点已在", file=sys.stderr)
    print("[SANITY]         其它渠道验证过，可 OPSD_SKIP_SANITY_CHECK=1 或 OPSD_SANITY_ONLY_WARN=1。", file=sys.stderr)

print("[SANITY][FAIL:SOFT] " + "; ".join(soft_failed), file=sys.stderr)
if only_warn:
    print("[SANITY][WARN] OPSD_SANITY_ONLY_WARN=1，仅警告不 abort，继续进入训练。", file=sys.stderr)
    sys.exit(0)
sys.exit(1)  # rc=1: 软失败
PY
    _sanity_verdict_rc=$?
    set -e

    if [ "$_sanity_verdict_rc" -ne 0 ]; then
        echo ""
        if [ "$_sanity_verdict_rc" -eq 2 ]; then
            echo "[FATAL] 起点模型 sanity check **硬阈值未通过**（accuracy 低于 $OPSD_SANITY_HARD_ACCURACY），拒绝启动 OPSD 训练。"
        else
            echo "[FATAL] 起点模型 sanity check **软阈值未通过**，拒绝启动 OPSD 训练。"
        fi
        echo "        产出目录：$_sanity_eval_dir"
        [ "$_sanity_from_cache" = "1" ] && echo "        （本次结果来自缓存，如怀疑缓存过时：OPSD_SANITY_FORCE_RERUN=1）"
        echo "        常见处置："
        echo "          1) 换用已验证的 checkpoint（例如 checkpoint-1600-merged / checkpoint-2000-merged）："
        echo "             export MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234"
        echo "          2) 若怀疑 merge 出错，可 rm -rf 该 -merged 目录后重跑；本脚本会用当前"
        echo "             swift 环境重新 merge 并做完整性检查。"
        echo "          3) 若日志出现 [SANITY][SUSPECT]（疑似推理链路问题），先手动核验："
        echo "             MODEL_PATH=$MODEL_PATH VAL_JSONL=$OPSD_SANITY_VAL_JSONL \\"
        echo "               MAX_NEW_TOKENS=2048 bash \$SCRIPT_DIR/run_inference_meld.sh"
        echo "          4) 若确认起点已在其它推理链路验证过，直接跳过或只 warn："
        echo "             OPSD_SKIP_SANITY_CHECK=1 bash $(basename "$0")"
        echo "             OPSD_SANITY_ONLY_WARN=1  bash $(basename "$0")   # 保留检查但不阻塞"
        echo "          5) 若确实需要下调阈值（不推荐），MELD 版例如："
        echo "             OPSD_MIN_ACCURACY=0.45 OPSD_MIN_FOCUS_RECALL=0.05 bash $(basename "$0")"
        echo "             OPSD_SANITY_FOCUS_CLASSES=fear,disgust bash $(basename "$0")  # 换关注类"
        echo "          6) 强制忽略缓存重跑（若怀疑 MODEL_PATH 已更新但缓存未失效）："
        echo "             OPSD_SANITY_FORCE_RERUN=1 bash $(basename "$0")"
        exit 17
    fi
    if [ "$_sanity_from_cache" = "1" ]; then
        echo "[INFO] sanity check（缓存）通过，进入 OPSD 训练。"
    else
        echo "[INFO] sanity check 通过，进入 OPSD 训练。"
    fi
    echo "===================================================================="
    echo ""
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

# ---------------- 组装冻结参数 (借鉴 SFT-full v2) ----------------
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
    echo "[INFO] FREEZE_PARAMETERS = <none> (未冻结 embedding/lm_head, OPSD 训练易漂; 建议 FREEZE_EMBED_LMHEAD=true)"
fi
# 传入 vit / aligner / llm 三档冻结开关 (swift 原生支持)
FREEZE_ARGS+=(
    --freeze_vit "$FREEZE_VIT"
    --freeze_aligner "$FREEZE_ALIGNER"
    --freeze_llm "$FREEZE_LLM"
)
echo "[INFO] FREEZE_VIT=$FREEZE_VIT  FREEZE_ALIGNER=$FREEZE_ALIGNER  FREEZE_LLM=$FREEZE_LLM  LABEL_SMOOTHING=$LABEL_SMOOTHING"

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

# 可选早停 / 恢复参数：仅在环境变量非空时才追加，保证向后兼容（与 GRPO 脚本对齐）
EXTRA_ARGS=()
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

# ---------------- 启动 OPSD 训练 ----------------
# 【路径 A4】训练命令后额外捕获返回码 → 事后 completions.jsonl 漂移扫描。
# swift rlhf 里面本身会处理 SIGINT/SIGTERM，此处不拦截；只在它退出后做事后体检。
set +e
NPROC_PER_NODE=$NPROC_PER_NODE \
"${SWIFT_CMD[@]}" rlhf \
    --rlhf_type gkd \
    --model "$MODEL_PATH" \
    --model_type step_audio2_mini \
    --system "$SYSTEM" \
    "${TEACHER_ARGS[@]}" \
    "${TUNER_ARGS[@]}" \
    "${FREEZE_ARGS[@]}" \
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
    --label_smoothing_factor $LABEL_SMOOTHING \
    --beta $BETA \
    --lmbda $LMBDA \
    --temperature $TEMPERATURE \
    --sft_alpha $SFT_ALPHA \
    "${TOPK_ARGS[@]}" \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_checkpointing $GC_FLAG \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
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
    --log_completions $LOG_COMPLETIONS \
    --seed 42 \
    "$@"
_opsd_train_rc=$?

# ---------------- 【路径 A4】on-policy 漂移事后扫描 ----------------
# 目的：在训练行将结束时快速定位"前 N 步内学生就已漂到音频/tts 分支"的事故。
# 不会抠前中断训练（那需要侵入式的 trainer callback，本轮不做），仅做事后报警。
# 启动条件：OPSD_ONPOLICY_DRIFT_ABORT_STEPS>0 且 LMBDA 非零（只有 on-policy 才会产生 completions.jsonl）。
# 注意：整段包在外层 set +e 内（尚未恢复），扫描内命令报错不会 abort 主脚本。
if [ "${OPSD_ONPOLICY_DRIFT_ABORT_STEPS:-0}" -gt 0 ] 2>/dev/null \
    && [ "$LMBDA" != "0.0" ] && [ "$LMBDA" != "0" ]; then
    _comp_file="$OUTPUT_DIR/completions.jsonl"
    # swift 可能自动追加版本子目录 (add_version=true)；递归找第一个 completions.jsonl
    if [ ! -f "$_comp_file" ]; then
        _comp_file=$(find "$OUTPUT_DIR" -maxdepth 4 -name completions.jsonl -type f 2>/dev/null | head -1)
    fi
    if [ -n "$_comp_file" ] && [ -f "$_comp_file" ]; then
        _drift_hits=$(head -n "$OPSD_ONPOLICY_DRIFT_ABORT_STEPS" "$_comp_file" 2>/dev/null \
            | grep -cE '<audio_|<tts_|<voice_' || true)
        echo ""
        echo "================ [路径 A4 drift check] ================"
        echo "[INFO] completions.jsonl        = $_comp_file"
        echo "[INFO] 扫描前 $OPSD_ONPOLICY_DRIFT_ABORT_STEPS 行内 <audio_*>/<tts_*>/<voice_*> 命中 = $_drift_hits"
        if [ "${_drift_hits:-0}" -gt 0 ]; then
            echo "[WARN][DRIFT] 前期 on-policy 采样已漂向音频码本分支，训练结果大概率失效。"
            echo "[WARN][DRIFT] 前 3 条面目："
            grep -m 3 -E '<audio_|<tts_|<voice_' "$_comp_file" 2>/dev/null \
                | head -3 \
                | sed 's/^/[WARN][DRIFT]   /'
            echo "[WARN][DRIFT] 建议："
            echo "[WARN][DRIFT]   1) 下调 LMBDA (例如 0.3-0.5) 降低学生采样占比；"
            echo "[WARN][DRIFT]   2) 或回退 LMBDA=0.0 (路径 A3 fallback，失去真 on-policy)；"
            echo "[WARN][DRIFT]   3) 检查 MODEL_PATH 是否真的是 SFT 后 ckpt (base 直接 on-policy 必崩)。"
        else
            echo "[INFO] 前期采样干净，无 <audio_*>/<tts_*>/<voice_*> 漂移信号。"
        fi
        echo "==========================================================="
    else
        echo "[WARN][DRIFT] 未找到 completions.jsonl (可能 LOG_COMPLETIONS=false 或训练未完成)，跳过漂移扫描。"
    fi
    unset _comp_file _drift_hits
fi

if [ "${_opsd_train_rc:-0}" -ne 0 ]; then
    echo "[WARN] swift rlhf 退出码 non-zero: $_opsd_train_rc (训练可能未完成)"
fi

echo "[INFO] MELD OPSD 后训练完成，Checkpoint 保存在: $OUTPUT_DIR"
