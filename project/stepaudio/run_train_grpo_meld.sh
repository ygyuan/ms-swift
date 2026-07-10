#!/usr/bin/env bash
# StepAudio2-mini GRPO 训练脚本 —— MELD 情感 7 分类版（基于 MS-SWIFT）
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v6-self-asr · 引入 self-ASR reward, 2026-07-10】
# ═══════════════════════════════════════════════════════════════════════════
# 背景观察 (SFT 阶段):
#   * v9-sft-full (纯 label prompt):        test acc = 59%
#   * v10-sft-full (外部注入 ASR 文本):     test acc = 69% (+10pp)
#   * v11-sft-full (self-ASR: 让模型自己转写再决策, 数据集
#     data_meld/train.r1omni_self_asr.jsonl): test acc = 68%
#   -> ASR 内容本身承载 ~10pp 的情感信号, 且 self-ASR 已在 SFT 内化。
#
# v6 三点关键改动 (相对 v5):
#   (1) MODEL_PATH 默认 -> ./output/meld/v11-sft-full/checkpoint-100
#       (v11 就是 self-ASR SFT 的产物, 与新数据格式天然对齐;
#        v0/v7 的旧起点只会背离新数据格式而崩塌)
#   (2) 数据集默认 -> data_meld/train.r1omni_self_asr.jsonl
#       (assistant 结构: <transcript>...</transcript>\n<answer>xxx</answer>,
#        并携带 asr_text 列供 WER reward 使用)
#   (3) REWARD_FUNCS 新增 stepaudio_transcript_wer, 权重 0.3
#       reward = clip(1 - WER(pred_transcript, gold_asr), 0, 1)
#       * 连续 / 单调, 无阈值突变;
#       * 只做正向奖励, 不会惩罚 "直接给 label" (那由 accuracy_recall 管);
#       * 0.3 vs recall 的正 reward 均值 ~2 是小项, 但相对 lazy_penalty=0.8
#         同数量级, 足以给 policy 拉出 "好好转写有收益" 的梯度。
#
# 连锁调整:
#   * MAX_COMPLETION_LENGTH 16 -> 128
#     (v1~v5 只需吐 label 1~3 token; v6 要吐 <transcript>+<answer>,
#      MELD 单句 30~60 词 ~= 60~120 subword token, 16 会硬截 -> WER 恒 ~1)
#   * WER reward 走 word-level, lowercase, strip punctuation
#     (通过环境变量 STEPAUDIO_WER_TOKENIZATION/_LOWERCASE/_STRIP_PUNCT 可调)
#
# 若要退回 v5 的纯 label reward + 老数据:
#   REWARD_FUNCS="stepaudio_accuracy_recall" REWARD_WEIGHTS="1.0" \
#     TRAIN_JSONL=$SCRIPT_DIR/data_meld/train.jsonl \
#     MAX_COMPLETION_LENGTH=16 bash $0
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【任务差异 · 相对 ASC 5 分类的 run_train_grpo.sh】
# ═══════════════════════════════════════════════════════════════════════════
# 类别 (7):    surprise, anger, neutral, joy, sadness, fear, disgust
# 数据目录:    project/stepaudio/data_meld/
# 数据分布:    neutral 47.2% / joy 17.4% / anger 12.1% / surprise 11.9% /
#              sadness 7.2% / disgust 2.7% / fear 2.7%   (train 9989 条)
# SFT 起点:    MELD full SFT (v0-20260707-182357/checkpoint-234) test acc=59.8%,
#              macro-F1=43.0%; LoRA v7 checkpoint-150 test acc=58.5%,
#              macro-F1=41.2%. 相比 ASC (SFT 已经 macro-F1=95.7%) MELD 的 SFT
#              baseline 弱得多, rare-class (fear/disgust) F1 只有 20~30%.
# 后训练 EV:   与 ASC 的 "负 EV 精修" 相反, MELD GRPO 是 **正 EV**:
#              - rare-class 蒙对率虽低 (fear 20% / disgust 16%) 但远非 0,
#                group std>0 组容易凑齐, 不会像 ASC 冷启 v4/v6/v9 那样死锁;
#              - SFT 每类都还没到 90% 收敛平台, 大量提升空间.
#
# 与 ASC 脚本的关键超参差异（为什么 MELD 更"敢"跑）：
#   * BETA 0.2 → 0.05      : MELD 起点弱，KL 拘束不用像 ASC 那么严
#                            （0.2 会让 loss 被 KL 主导而学不动）；
#   * LR 5e-7 → 2e-6       : 弱起点可承受 4× 更大步长，加快收敛；
#   * MAX_STEPS 150 → 600  : rare-class 单 epoch 出现 ~270 次，要跑多个
#                            epoch 才能吃透（150 步只覆盖 ~1/5 epoch）；
#   * TEMPERATURE 0.7 → 1.0 : SFT 已给出 rare-class 的方向信号但不精，
#                            适度提升采样多样性帮助 group std>0；
#   * TOP_P 0.9 → 0.95 / TOP_K 20 → 40 : 同上；
#   * DYNAMIC_SAMPLE false → true : rare-class group 里 8/8 全错概率
#                            较高（fear 蒙对率仅 20% → 8 条全错概率 17%），
#                            丢弃 std=0 组能显著加速有效梯度累积；
#   * REWARD_FUNCS = stepaudio_accuracy_weighted stepaudio_format
#                            + STEPAUDIO_CLASS_WEIGHTS 给 rare 类加权：
#                            MELD 是"正 EV + rare 有信号但稀疏"的典型
#                            场景，加权能显著把 rare-class 的正 advantage
#                            拉出组内噪声；
#   * STEPAUDIO_CLASS_WEIGHTS = neutral=1.0,joy=1.5,anger=2.0,surprise=2.0,
#                               sadness=3.0,disgust=5.0,fear=5.0
#                            (inverse-freq clip 到 [1.0, 5.0])
#   * MODEL_PATH 默认告警：路径改为 MELD SFT ckpt；
#   * SYSTEM 强制注入 "You are a helpful assistant." (对齐 UltraEval-Audio
#     baseline 55.47% 的 prompt 前缀; run_inference_meld.sh 也是这么做的).
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v5 复盘 · GRPO v4-20260708-1614xx: STRICT 治 joy 泛滥过度成功，rare 类退步】
# ═══════════════════════════════════════════════════════════════════════════
# v4 结果 (仍从 LoRA v7 起步, SFT baseline macro-F1=41.17):
#   ckpt-25   acc=55.75  macro-F1=32.90
#   ckpt-150  acc=55.90  macro-F1=34.82  ← v4 最佳
#   ckpt-300  acc=55.25  macro-F1=34.26
# 五代趋势: v0=30.06 → v1=33.25 → v2=35.20 → v3=36.67 → v4=34.82 (首次回退!)。
# v4 比 v3-best 掉了 1.85pp, 五代来第一次负增长。
#
# 【v4 · 部分成功】
#   * joy 泛滥被治好: joy recall 48.5 → 38.1 (-10.4pp), STRICT 确实关闭了
#     joy 从 neutral 抢 mass 的零成本通道 ✓
#   * surprise 过度扩张收窄: recall 51.6 → 44.8, precision 从 36 回到 ~42 ✓
#
# 【v4 · 严重回退 · 三处问题】
#   (Q1) neutral 大幅反弹: recall 78.3 → 86.5 (+8.2pp), 回到接近 v1 水平。
#        原因: STRICT 罚了"猜 neutral", policy 的反应不是"改猜 rare",
#        而是"面对 neutral audio 时坚定猜 neutral 拿 +0.5 稳定收益, 面对
#        non-neutral audio 时也不敢猜 neutral 但改猜 anger/surprise (0 reward)
#        —— rare 类蒙对率低所以没收益"。结果 neutral 的 predicted-positive
#        变纯净但 recall 反而涨了。
#   (Q2) rare 3 类全面退步:
#         disgust: 16.2 → 13.2 (-3.0pp) 倒退到 v2 水平
#         anger:   15.4 → 12.5 (-2.9pp)
#         sadness: 19.2 → 15.9 (-3.3pp)
#        v3 好不容易撬动的 rare 类学习, v4 又打回去。根源: STRICT 惩罚在
#        非 disgust group 里也生效 (它们占 6/7 batch), 累积起来"惩罚 neutral"
#        的信号被稀释到全类别, rare 类的正激励没能相对突出。
#   (Q3) 三个 ckpt 动态范围只有 1.92pp (v3 是 3.8pp), 又进入 v1 那种
#        "低学习信号"高原, 学不动。
#
# 【v5 · 换思路: 前 4 代都在 reward 侧折腾, 边际收益递减, v4 甚至负收益】
# v4 的教训: 只用 reward shaping 无法把 rare 类彻底拉起来 —— 因为 SFT 起点
# (LoRA v7) 本身 disgust 蒙对率只有 ~13%, 16 条 rollout 里期望 2 条对,
# group std 极小。要让 rare 学起来, 必须从两个方向同时用力:
# **[B] 更温和的 joy 抑制 (关闭 STRICT, 改用 weight)** 恢复 v3 学习信号;
# **[C] 提高 rare 类的探索信号强度 (更大 gen_batch)**。
#
# 【v5 具体改动 · 6 条】
#   (1) 【关键 · B】关闭 STRICT: STEPAUDIO_LAZY_STRICT=1→0。回到 v3 non-strict
#       语义 (v3 曾 macro-F1=36.67 是历史最佳)。
#   (2) 【关键 · B】LAZY_LABELS 从 'neutral,joy' 缩回 'neutral': 只罚
#       gt=rare,pred=neutral (v1/v2/v3 老规则)。
#   (3) 【关键 · B】joy weight 从 1.0 降到 0.6: 用 weight 替代 STRICT 治 joy 泛滥。
#       joy=0.6 与 neutral=0.5 差距只有 0.1, policy 没有强烈动机猜 joy;
#       与 disgust=8 差 13×, joy 命中的相对激励弱到不足以形成 mode-seeking。
#   (4) 【关键 · C】GRAD_ACCUM 从 12 抬到 16: gen_batch 从 48 (3 prompt×G=16)
#       抬到 64 (4 prompt×G=16), rare 类每 step 曝光 +33%, 累计梯度更稳定。
#       TRL 约束 64%16=0 ✓, eval_batch=16 也满足。训练时间 +33%。
#   (5) surprise weight 从 2.0 回升到 2.2: v4 里 surprise recall 44.8 尚可,
#       离 SFT 46.3 仅 -1.5pp, 但 v3 用 2.5 时曾 51.6 过度扩张; 折中 2.2。
#   (6) disgust/fear 保持 8.0 (v3 证明有效, v5 继续加大 gen_batch 让它更稳定)。
#
# 【保留 v4 已生效 (对 joy 抑制)】: LAZY_PENALTY=0.8 (还罚 neutral) / MAX_STEPS=300
# (v4 曲线证明 150 附近最佳) / plugin 新增的 STRICT 参数 (v5 关闭但保留 API)。
#
# 【使用建议】v5 完全默认参数即可:
#   bash project/stepaudio/run_train_grpo_meld.sh   # 起点=full-SFT (推荐)
#   MODEL_PATH=./output/meld/v7-20260707-162917/checkpoint-150 \
#       ALLOW_LORA_V7_START=1 \
#       bash project/stepaudio/run_train_grpo_meld.sh  # 与 v4 同起点对照
#
# 【v5 预期】
#   * 最好情况: 恢复 v3 水平 (36.67) 并因 gen_batch 更大而进一步涨到 37-38;
#     joy 因 weight 极低 (0.6) 而 recall 回落到 30 附近;
#     rare 类因 exploration 更稳定而 disgust ≥ 18, sadness ≥ 22;
#     若用 full-SFT 起点, 预计 macro-F1 ≈ 39-40, 逼近 SFT baseline 41.2。
#
# 【若 v5 仍不达标, 备选】
#   (A) 混合 reward: REWARD_FUNCS="stepaudio_accuracy_recall
#       stepaudio_accuracy_weighted" REWARD_WEIGHTS="1.0 0.3", 用
#       weighted 提供绝对幅度差, recall 提供负 reward 治偷懒;
#   (B) Resume from v3 ckpt-250 (best): 保留 v3 已学到的 rare 类知识,
#       只叠加 v5 的温和调整, RESUME_CHECKPOINT=output/meld/grpo/v3/checkpoint-250;
#   (C) 数据侧: disgust/fear 更激进 oversample (1428 → 2500/类);
#   (D) 分阶段: 前 150 步 STRICT=1 治泛滥, 后 150 步 STRICT=0 微调。
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v4 复盘 · GRPO v3-20260708-1508xx: 继续缩窄差距但 joy 泛滥+起点未换是主因】(archive)
# ═══════════════════════════════════════════════════════════════════════════
# v3 结果 (⚠️ 用户显式 MODEL_PATH=./output/meld/v7... 覆盖了 v3 推荐的 full-SFT
# 默认起点, 所以 v3 实际仍从 LoRA v7 (macro-F1=41.2) 起步, 而非我预期的
# full-SFT ckpt-234 (macro-F1=43.0). 直接失去 +1.8pp 免费抬升):
#   ckpt-25   acc=55.75  macro-F1=32.87
#   ckpt-250  acc=54.98  macro-F1=36.67  ← v3 最佳
#   ckpt-500  acc=54.21  macro-F1=35.55  (过拟合回落)
# 四代趋势: v0=30.06 → v1=33.25 → v2=35.20 → v3=36.67, 累计 +6.61pp,
# 但仍差 SFT 4.5pp。
#
# 【v3 · 好消息】
#   * disgust recall 首次真正"动了": v0/v1/v2 = 11.8/13.2/13.2 (纹丝不动),
#     v3 从 ckpt-25 的 11.8 → ckpt-250 的 16.2 (+4.4pp). accuracy_recall
#     加大 class weight (5→8) 的组合终于让 disgust 有学习信号。
#   * neutral over-prediction 继续打压: recall 90.8 → 78.3 (ckpt-25→250)。
#   * surprise recall 35.6→51.6 (+16pp) 已超 SFT 46.3。
#   * sadness F1 30.2 vs SFT 35.8, 差距缩窄到 5.6pp。
#   * joy F1 46.0 已远超 SFT 36.2 (但见坏消息第 2 条)。
#
# 【v3 · 三处关键问题】
#   (P1) 起点未换 (最大机会成本): 用户命令行覆盖了 MODEL_PATH, 仍用 LoRA v7,
#        直接失去 +1.8pp 起点; 而且 LoRA v7 disgust 蒙对率极低, v3 ckpt-25
#        disgust recall 才 11.8 (跟 v1/v2 一样), 说明起点决定了 disgust 的
#        探索上限。若换成 full-SFT ckpt-234, 预计 v3 macro-F1 ≈ 38.5。
#   (P2) joy 泛滥 (LAZY 对称豁免 bug 未修复): joy recall 34.6 → 48.5 (+14pp!),
#        precision 43.8 (中等), joy 变成新的"避风港"。根源是 v2/v3 的
#        LAZY_LABELS='neutral,joy' 但非 STRICT 模式下 gt=neutral,pred=joy
#        因"gt 在 LAZY 里"而不罚, joy 无成本从 neutral 抢 mass。
#   (P3) surprise 过度扩张 (precision 塌陷): recall 35.6→51.6 好看, 但
#        precision 从 48.1 掉到 35.9 (-12pp), 411 次预测里 266 次错猜。
#        原因: surprise weight=2.5 高于 anger=2.0/neutral=0.5, policy 学到
#        "不确定就猜 surprise" —— 相当于把 neutral 的问题转移到 surprise 上。
#
# 【v4 优化 · 5 条主线】
#   (1) 【关键 · A】起点强制警告: 若用户传的 MODEL_PATH 是 LoRA SFT 版本,
#       打印显著警告推荐 full-SFT ckpt-234, 强制用户至少看一眼再决定。
#   (2) 【关键 · B】升级 plugin: 新增 STEPAUDIO_LAZY_STRICT=1 模式, 语义:
#         pred == gt                        -> +w[gt]
#         pred ∈ LAZY & pred != gt          -> -penalty  (无论 gt 是否在 LAZY)
#         其它 miss                          ->  0.0
#       这样 gt=neutral,pred=joy 也罚 -0.8, 关闭 joy 从 neutral 抢 mass 的
#       零成本通道。同时 gt=joy,pred=neutral 也罚, 抑制反向 hedging。
#       默认打开 STEPAUDIO_LAZY_STRICT=1, LAZY_LABELS='neutral,joy'。
#   (3) 【关键 · C】surprise weight 从 2.5 降到 2.0 (与 anger 持平), 避免
#       policy 把 surprise 当新的避风港。
#   (4) joy weight 从 1.2 降到 1.0 (跟 disgust weight=8/anger=2.0 相比,
#       joy 命中的正激励最弱), 配合 STRICT 模式压制 joy 泛滥。
#   (5) MAX_STEPS 500 → 300: v3 ckpt-500 (macro-F1=35.55) 已过拟合于 ckpt-250
#       (36.67), 早停节省 40% 训练时间, 且更容易挑到最佳 ckpt。
#
# 【保留 v3 已生效】: G=16 / TEMP=1.3 / EPSILON_HIGH=0.24 / BETA=0.15 /
# LR=1.5e-6 / disgust=8/fear=8 / accuracy_recall / LAZY_PENALTY=0.8 /
# balanced 数据 / scale_rewards=none / dynamic_sample=false / GRAD_ACCUM=12。
#
# 【重要】使用建议:
#   # 强烈推荐 (v4 默认组合, 起点=full-SFT):
#   bash project/stepaudio/run_train_grpo_meld.sh
#
#   # 若必须用 LoRA v7 起点 (会有 -1.8pp 起点惩罚):
#   MODEL_PATH=./output/meld/v7-20260707-162917/checkpoint-150 \
#       bash project/stepaudio/run_train_grpo_meld.sh
#
# 若 v4 仍不达标, 备选路线 (按优先级):
#   (A) 恢复混合 reward 兜底 (weighted_accuracy 提供绝对幅度差):
#       REWARD_FUNCS="stepaudio_accuracy_recall stepaudio_accuracy_weighted"
#       REWARD_WEIGHTS="1.0 0.3";
#   (B) LAZY_PENALTY 抬到 1.2 (更激进抑制 hedging), 但注意 rare 类
#       precision 可能大幅塌陷;
#   (C) 数据侧: disgust/fear 更激进 oversample (1428 → 2500/类);
#   (D) 分阶段训练: 前 150 步用 STRICT=1 治泛滥, 后 150 步关掉 STRICT 只留
#       正激励做微调 (需 resume from ckpt)。
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v3 复盘 · GRPO v2-20260708-112xxx 继续改善但 disgust 未突破】(archive)
# v2 结果 (SFT baseline macro-F1=41.17 / acc=58.47):
#   ckpt-25   acc=55.79  macro-F1=32.53
#   ckpt-250  acc=55.40  macro-F1=35.20  ← v2 最佳
#   ckpt-500  acc=55.36  macro-F1=33.95  (回落, 出现过拟合)
# 三代趋势: v0=30.06 → v1=33.25 → v2=35.20, 方向对, 累计 +5.14pp,
# 但仍差 SFT 约 6pp。
#
# 【v2 · 好消息】
#   * 训练不再"高原": 三个 ckpt macro-F1 动态范围 2.7pp (v1 只有 0.7pp),
#     说明 accuracy_recall 负 reward 确实让梯度信号变强了;
#   * surprise recall 36.3 → 47.7, **首次超越 SFT baseline 46.3** (+1.4pp);
#   * joy recall 34.6 → 39.8, joy F1 从 ~41 涨到 43.4, F1 已超 SFT (36.2);
#   * anger recall 9.3 → 13.0, sadness recall 13.5 → 16.4, fear 14.0 → 18.0
#     (均有 +3~4pp 提升, 但仍距 SFT 差 4~10pp);
#   * neutral over-prediction 打压继续: neutral recall 90.6 → 84.0.
#
# 【v2 · 坏消息 · 一处关键问题】
#   * disgust recall 全程停在 13.2%: ckpt-25/250/500 分别是 11.8/13.2/13.2
#     (SFT 45.6%, 差 32.4pp). 三个 ckpt **数值一模一样**说明 disgust 在训练
#     中完全没被学习到。
#
# 【v2 · disgust 根因量化】
#   - SFT disgust 是靠"高 recall 45.6% 但低 precision 25%" 换来的 —— 大量猜 disgust
#     但很多是猜错。GRPO 天然趋向高 precision, disgust 一旦不确定就放弃。
#   - MELD balanced 里每 1/7 = 14% group gt=disgust。12 条 rollout 里出现
#     disgust 概率约 86%, 但每 group 内只 1~2 条命中 → signal-to-noise 极低。
#   - 与此同时: joy 从 neutral 抢 mass 更凶 (recall 34.6→39.8), 因为我把
#     joy 加入 LAZY_LABELS 时出现了对称豁免 bug: gt=neutral,pred=joy 因
#     "gt 在 LAZY 里"而不罚, joy 无成本地吸收 neutral 的 mass。
#
# 【v3 优化 · 3 条主线】
#   (1) 【关键 · A】起点换到 full-SFT (v0-20260707-182357/ckpt-234, macro-F1=43.0):
#       比 LoRA v7 (41.2) 直接抬高 1.8pp, 而且 full-SFT 的 rare-class mass
#       更完整, disgust 出现在 rollout 的频率会自然更高。
#   (2) 【关键 · B】修复 LAZY 对称豁免 bug:
#       STEPAUDIO_LAZY_LABELS 从 'neutral,joy' 缩回 'neutral' (只惩罚 neutral),
#       同时 joy weight 从 1.5 降到 1.2 (降低猜 joy 的正激励);
#       disgust/fear weight 5.0 → 8.0 (加强稀有类正激励)。
#   (3) 【关键 · C】提高 disgust 出现在 rollout 的概率:
#       NUM_GENERATIONS 12 → 16 (每 group 多 33% 命中机会),
#       TEMPERATURE 1.2 → 1.3 (更宽采样, disgust 低置信预测更易入 rollout)。
#   (4) 【辅助】BETA 0.1 → 0.15: 拉紧 KL 抑制 joy 侧 mass 漂移。
#   (5) 【连锁】GRAD_ACCUM 9 → 12 (gen_batch=1×4×12=48=3×16 ✓ 满足 TRL 约束),
#       EVAL_BATCH_SIZE 3 → 4 (4×4=16 与 num_generations=16 整除)。
#   (6) 【保留 v2 已生效】: MAX_STEPS 500 / SAVE_STEPS 25 / balanced 数据 /
#       scale_rewards=none / dynamic_sample=false / TOP_P=1.0 / TOP_K=0 /
#       LR=1.5e-6 / EPSILON_HIGH=0.24 / accuracy_recall / LAZY_PENALTY=0.8 /
#       neutral 权重=0.5 / surprise=2.5 / sadness=3.5 / anger=2.0。
#
# 若 v3 仍不达标, 备选路线 (按优先级):
#   (A) 升级 plugin, 加"group-level distributional recall bonus": 检测 group
#       里 gt-class rollout ≥ 1 条时给额外 +bonus (专治 disgust 稀疏);
#   (B) 混合 reward: REWARD_FUNCS="stepaudio_accuracy_recall
#       stepaudio_accuracy_weighted", weights="1.0 0.3" (加大 disgust 命中
#       时的 5×+8× 双重奖励);
#   (C) LAZY_PENALTY 抬到 1.2 (加大"猜错 neutral"的代价, 更激进);
#   (D) 数据侧: 对 disgust/fear 做更激进的 oversample (from 1428 到 2500),
#       让每 batch disgust 出现频率进一步提升。
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v2 复盘 · GRPO v1-20260708-104405 部分改善但仍未超越 SFT baseline】(archive)
# ═══════════════════════════════════════════════════════════════════════════
# v1 结果 (起点仍是 SFT v7 ckpt-150, SFT baseline macro-F1=41.17 / acc=58.47):
#   ckpt-25   acc=55.79  macro-F1=32.79
#   ckpt-150  acc=55.75  macro-F1=33.25  ← v1 最佳
#   ckpt-300  acc=55.63  macro-F1=32.53
# 相对 v0 (macro-F1 30.06→33.25, +3.2pp) 有进步但仍差 SFT 约 8pp。
#
# 【好消息 · v1 部分策略生效】
#   * neutral over-prediction 已从 v0 的 82.3% 回落 (balanced 数据 + neutral 权
#     重打折 + scale_rewards=none + dynamic_sample=false 组合起作用);
#   * surprise recall 22.4→36.3 (+14pp), 说明打压 neutral 后 mass 部分回流;
#   * joy recall 33.3→34.6 (维持), 依然高于 SFT (25.1);
#
# 【坏消息 · 三个 rare 类仍严重掉 recall】
#   - disgust recall : SFT 45.6% → v1 13.2%  (-32.4pp !!! 关键短板)
#   - anger   recall : SFT 22.6% → v1  9.3%  (-13.3pp)
#   - sadness recall : SFT 24.0% → v1 13.5%  (-10.5pp)
#   - fear    recall : SFT 22.0% → v1 14.0%  (-8.0pp)
#   - surprise recall: SFT 46.3% → v1 36.3%  (-10.0pp)
#
# 【三个 ckpt 差异只有 0.7pp】说明训练在 step 25 就已高原, 学习信号
# 本身不够强, 不是"训得不够"的问题。
#
# 【v1 深层根因】
#   (a) SFT 的 disgust 是靠"高 recall 低 precision" (recall 45.6% / precision
#       25%) 换来的 —— SFT 会"乱猜 disgust", 把 neutral/surprise 的 93 条误判
#       成 disgust, 因此 recall 高。GRPO 只在"猜对"给正 reward, policy gradient
#       会自然趋向"高 precision 低 recall"的保守策略 —— 一旦不确定就退回猜
#       majority。这是 policy gradient 的本性 —— 不会主动冒险探索。
#   (b) 对 disgust 这种 SFT-recall 只有 45% 的类, 8 条 rollout 期望 3.6 条对,
#       但很多样本 SFT 本身也猜错 → rollout 8/8 全错或 1/8 对, group std 极小,
#       weighted advantage 接近 0, policy 不会往 disgust 方向探索。
#   (c) 换句话说, v1 的 reward 只有"胡萝卜"没有"棍子": 猜错给 0.0, 猜对给
#       weight[gt]. rare-class 组"全错"时全组都是 0, 没有梯度信号。
#   (d) 三个 ckpt 差异 0.7pp: 梯度信号弱 + 保守超参 (LR=1e-6, epsilon_high=0.22)
#       组合导致 policy 几乎不动。
#
# 【v2 优化 · 核心是引入负 reward + 更强探索】
#   (1) 【关键】REWARD_FUNCS 切换到 stepaudio_accuracy_recall (新 ORM,
#       已在 stepaudio_plugin.py 注册)。语义:
#         pred == gt          -> +w[gt]      (与 accuracy_weighted 同)
#         pred ∈ LAZY & gt ∉  -> -penalty    (惩罚偷懒猜 majority)  ← 新
#         其它 miss           ->  0.0
#       解决"胡萝卜无棍子"问题: rare-class 组即使 8/8 全错也能得到 -penalty,
#       group std 变大, advantage 明显, policy 被迫探索。
#   (2) 【关键】STEPAUDIO_LAZY_LABELS='neutral,joy': 不只惩罚 neutral,
#       连 joy 也纳入 (v1 里 joy recall 34.6% > SFT 25.1%, 已开始被当"避风港"),
#       防止 policy 把 mass 从 neutral 挪到 joy 而不去 rare 类。
#   (3) STEPAUDIO_LAZY_PENALTY=0.8 (默认 0.5→0.8): 与 disgust weight=5 比,
#       -0.8 的惩罚只占 16%, 不至于压垮 rare 探索的信号; 但比 0.5 强 60%,
#       能显著加大 group std, 让梯度真正流动。
#   (4) NUM_GENERATIONS 8 → 12: disgust 蒙对率 16%, 8 条 rollout 期望 1.3 条对,
#       12 条期望 2 条, 显著降低 std=0 概率。
#   (5) TEMPERATURE 1.1 → 1.2: 更宽采样, disgust/fear 低置信预测更容易入 rollout。
#   (6) LR 1e-6 → 1.5e-6, EPSILON_HIGH 0.22 → 0.24: 引入负 reward 后 advantage
#       变大且稳定, 可以放开一点步长; 与 v0 的 2e-6 相比仍保守 25%。
#   (7) MAX_STEPS 300 → 500: 强化学习信号后有梯度可学, 让训练走完 balanced
#       数据 ~1.5 个 epoch (balanced ~10000/8 ≈ 1250 steps/epoch × 4gpu?
#       实际 ~312/epoch, 500 步 = 1.6 epoch)。
#   (8) 保留 v1 已生效的组合: balanced 数据 / neutral 权重 0.5 /
#       scale_rewards=none / dynamic_sample=false / TOP_P=1.0 / TOP_K=0。
#
# 若 v2 仍不达标, 备选路线 (按优先级):
#   (A) 增大 LAZY_PENALTY 到 1.5, 让 rare 类学习信号进一步放大;
#   (B) 起点从 v7 (LoRA SFT, macro-F1=41.2) 换到 v0 full-SFT (macro-F1=43.0),
#       起点更强, 后续每一步的 KL 约束更容易保住 disgust 边界;
#   (C) 混合 recall + weighted (REWARD_FUNCS="stepaudio_accuracy_recall
#       stepaudio_accuracy_weighted", weights="1.0 0.3"), 让 recall 主导
#       但保留 weighted 的绝对幅度差。
# ═══════════════════════════════════════════════════════════════════════════
#
# ═══════════════════════════════════════════════════════════════════════════
# 【v1 复盘 · GRPO v0-20260708-094519 未超越 SFT baseline】(archive)
# ═══════════════════════════════════════════════════════════════════════════
# v0 结果（起点 = SFT v7 ckpt-150, test macro-F1=0.412 / acc=0.585）：
#   ckpt-200  acc=0.5487  macro-F1=0.3006
#   ckpt-400  acc=0.5494  macro-F1=0.3031
#   ckpt-600  acc=0.5490  macro-F1=0.3063
# 三个 ckpt 差异微乎其微 (macro-F1 差 0.6pp) 说明训练 step 100 后已高原,
# 且比 SFT baseline 全面倒退 (accuracy -3.6pp, macro-F1 -11pp)。
#
# 【症状】GRPO 把预测 mass 大幅转移到 majority (neutral):
#   - neutral recall  : SFT 89.6% → GRPO 93.7% (↑ 4pp)
#   - joy     recall  : SFT 25.1% → GRPO 33.3% (↑ 8pp)
#   - surprise recall : SFT 46.3% → GRPO 22.4% (↓ 24pp !!)
#   - anger   recall  : SFT 22.6% → GRPO  7.5% (↓ 15pp)
#   - sadness recall  : SFT 24.0% → GRPO 11.5% (↓ 13pp)
#   - fear    recall  : SFT 22.0% → GRPO 14.0% (↓  8pp)
#   - disgust recall  : SFT 45.6% → GRPO 11.8% (↓ 34pp !!!)
# neutral 占预测总数从 70.8% 拉高到 82.3% —— 典型 mode-collapse 到 majority。
#
# 【根因】
#   (a) 训练 reward 骗人: eval reward 用同款 weighted-accuracy, 但 val
#       是从 train 切 1% (分布同 47% neutral 偏置), reward 上涨其实来自
#       neutral recall 涨的贡献 (weight=1.0 的 neutral 命中数增加压过了
#       weight=5 的 disgust 命中数减少)。
#   (b) stepaudio_format 是死项: MELD 上 invalid_response=0, format 恒=1,
#       方差=0, 完全不产生梯度, 只是稀释 group std, 让 policy 更"听话"于
#       accuracy_weighted 的 majority 主导信号。
#   (c) scale_rewards=group 抹平了 class weight: rare-class 组内均值/std
#       归一后, weight=5 vs weight=1 的绝对差被压平, class weight 名存实亡。
#   (d) dynamic_sample=true 反向作用: MELD 场景 std=0 组恰恰是"neutral 全
#       对"或"rare-class 全错"这类需要"打压 majority 自信度 / 保留 rare 类
#       负 anchor"的样本, 丢掉它们等于只留 majority 争议样本, 反而放大 majority 学习。
#   (e) epsilon_high=0.28 (不对称 clip): 让 majority 的正 advantage 被过度放大,
#       在长训练下累积成 neutral logit 的单侧漂移。
#   (f) 数据源仍是原始 47% neutral 偏置的 train.jsonl。
#
# 【v1 优化 · 一次到位对照实验】
#   (1) 数据: 默认切换到 train.balanced.jsonl (oversample 到均值 ~1428/类),
#       并在文件不存在时自动 python3 data_meld/build_meld_train_balanced.py
#       生成; 消除 47% neutral 采样偏置。
#   (2) Reward: 移除 stepaudio_format (方差=0 死项), 只留
#       stepaudio_accuracy_weighted; REWARD_WEIGHTS 相应从 "1.0 0.1" → "1.0"。
#   (3) Class weight: neutral 1.0 → 0.5 (打压 majority 正 reward),
#       surprise 2.0 → 2.5 / sadness 3.0 → 3.5, 让 policy 有动力恢复 rare 类。
#   (4) SCALE_REWARDS group → none: 让 class weight 真正作用到 advantage 上。
#   (5) DYNAMIC_SAMPLE true → false: 保留 std=0 组的"打压 majority 自信度"信号。
#   (6) BETA 0.05 → 0.1: KL 拉紧一档, 抑制 rare 类被吞噬。
#   (7) EPSILON_HIGH 0.28 → 0.22: 收窄不对称 clip, 抑制 majority 单侧漂移。
#   (8) LR 2e-6 → 1e-6: 对已有 SFT 起点小步走。
#   (9) MAX_STEPS 600 → 300: v0 曲线 step 100 后已高原, 早停避免继续偏 majority。
#   (10) TOP_P 0.95 → 1.0, TEMPERATURE 1.0 → 1.1: 更宽采样, 增加 rare 类
#       进入 8 条 rollout 的概率。
# ═══════════════════════════════════════════════════════════════════════════
#
# 关键设计（沿用 ASC 脚本）：
#   1. 使用 swift rlhf --rlhf_type grpo；
#   2. 自定义 reward 插件（外置 plugin）：
#        examples/train/grpo/plugin/stepaudio/stepaudio_plugin.py
#      注册 stepaudio_accuracy / stepaudio_format / stepaudio_accuracy_weighted。
#      **重要**：plugin 会从 prompt 里动态解析 label 列表（正则匹配
#      "one of [a, b, c]"），所以 MELD 7 类无需改 plugin 代码，只要
#      MELD 的 prompt 里包含 "[surprise,anger,neutral,joy,sadness,fear,disgust]"
#      枚举（已在 build_meld_jsonl.py 里生成）就能自动适配。
#   3. **不**启用 vLLM：vLLM 暂不支持 step_audio2_mini 的音频多模态输入；
#      使用 transformers native generate 走 GRPO sampling，G=NUM_GENERATIONS。
#   4. 沿用 SFT 脚本里所有踩过的坑：
#        - attn_impl eager（step_audio2_mini 当前唯一支持值）
#        - torch_dtype bfloat16
#        - truncation_strategy delete
#        - PYTORCH_CUDA_ALLOC_CONF expandable_segments
#        - 训练前的 GPU 占用前置检查
# GRPO 显存压力极大（一步要生成 G 条 completion + 计算 ref/old logp），
# 默认走 LoRA + DeepSpeed ZeRO-2。
#   5. GRPO 不需要单独的 val_dataset，使用 --split_dataset_ratio 从 train 切 1%。

set -ex

export LOG_LEVEL=INFO
export WANDB_DISABLED=true
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# ---------------- 模型 / 输出路径 ----------------
# ★ GRPO 必须从 MELD SFT 后的 ckpt 起步（冷启动会重现 ASC v4/v6/v9 崩塌曲线）。
# 推荐候选（按 macro-F1 排序，v4 强烈建议 full-SFT）：
#   * MELD full SFT: ./output/meld/v0-20260707-182357/checkpoint-234
#     (test acc=59.8% / macro-F1=43.0%, rare 类都有正蒙对率)  ← v4 强烈推荐
#   * MELD LoRA SFT: ./output/meld/v7-20260707-162917/checkpoint-150
#     (test acc=58.5% / macro-F1=41.2%, 脚本会自动 merge_lora, v0/v1/v2/v3 用过)
# 【v4 复盘 · 起点选择的量化影响】v3 用户命令行覆盖了 MODEL_PATH 到 LoRA v7,
# 结果 v3 ckpt-25 的 disgust recall 只有 11.8% (与 v1/v2 完全一样), 说明起点
# 决定了 disgust 的"可探索范围"。若换成 full-SFT ckpt-234 (rare-class mass
# 更完整), 起点直接 +1.8pp, 加上 disgust 探索上限提升, 预计整体 macro-F1
# 可达 38-40 (v3 的 36.67 → 38.5)。
# 【v6 复盘 · 默认起点切换到 v11-sft-full】v11 是 self-ASR 数据集下 SFT 的产物
# (test acc=68%, assistant 已学会吐 <transcript>...</transcript>+<answer>),
# 与新 GRPO 的数据格式/prompt 天然对齐; 若继续用 v0/v7 老起点, assistant 的
# 输出分布仍是单 token label, WER reward 会恒 0, 训练 collapse。
MODEL_PATH=${MODEL_PATH:-$SWIFT_ROOT/output/meld/v11-sft-full/checkpoint-100}
if [ "$MODEL_PATH" = "/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini" ]; then
    echo ""
    echo "[WARN] ============================================================"
    echo "[WARN] MODEL_PATH 指向的是原始 base 模型（未 SFT）。"
    echo "[WARN] MELD GRPO 必须从 SFT 后的 ckpt 起步："
    echo "[WARN]   base 未 SFT 时 rare-class (fear/disgust) 蒙对率接近 0,"
    echo "[WARN]   group std 恒为 0，policy 只从 majority (neutral) 学到"
    echo "[WARN]   \"永远输出 neutral\" 的塌缩解 (ASC v4/v6/v9 曲线)。"
    echo "[WARN] 强烈建议："
    echo "[WARN]     MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234 \\  # full"
    echo "[WARN]     bash project/stepaudio/run_train_grpo_meld.sh"
    echo "[WARN] 若坚持从 base 冷启动，请显式设置：STRICT_SFT_WARMUP=0"
    echo "[WARN] ============================================================"
    if [ "${STRICT_SFT_WARMUP:-1}" = "1" ]; then
        echo "[FATAL] 拒绝冷启动 GRPO（STRICT_SFT_WARMUP=1，默认）。请指定 MELD SFT 后的 MODEL_PATH。"
        exit 16
    fi
    echo "[WARN] 已通过 STRICT_SFT_WARMUP=0 允许冷启动，请注意 rare-class 崩塌风险。"
fi
# 【v4 新增 · LoRA v7 起点告警】v0/v1/v2/v3 均从 LoRA v7 起步, 但 full-SFT
# ckpt-234 (macro-F1=43.0) 比 LoRA v7 (macro-F1=41.2) 直接高 1.8pp,
# 而且 full-SFT 的 rare-class mass 更完整, disgust/fear/anger 的探索上限更高。
# 检测到 LoRA v7 起点时给出显著警告 (非致命, 用户仍可继续)。
case "$MODEL_PATH" in
    *v7-20260707-162917/checkpoint-150*|*v7-20260707-162917/checkpoint-150-merged*)
        echo ""
        echo "[WARN] --------------------------------------------------------"
        echo "[WARN] 你正在使用 LoRA v7 ckpt-150 作为 GRPO 起点。"
        echo "[WARN] 但 full-SFT v0 ckpt-234 起点通常显著更优:"
        echo "[WARN]   * SFT macro-F1: LoRA v7 = 41.2  vs  full-SFT v0 = 43.0 (+1.8pp)"
        echo "[WARN]   * rare-class mass: full-SFT 保留更完整, disgust 探索上限更高"
        echo "[WARN] 强烈建议改用 full-SFT 起点:"
        echo "[WARN]   MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234 \\"
        echo "[WARN]     bash project/stepaudio/run_train_grpo_meld.sh"
        echo "[WARN] 若仍希望继续用 LoRA v7 (对照实验等场景), 显式设置"
        echo "[WARN]   ALLOW_LORA_V7_START=1  跳过本告警。"
        echo "[WARN] --------------------------------------------------------"
        if [ "${ALLOW_LORA_V7_START:-0}" != "1" ]; then
            echo "[INFO] 5 秒后继续 (Ctrl+C 中止, 或 ALLOW_LORA_V7_START=1 消除本告警)..."
            sleep 5
        fi
        ;;
esac
# v2 默认输出目录: output/meld/grpo/v2 (ADD_VERSION=false 时不追加 timestamp)。
# 若需要多组 v2 对照实验, 显式 OUTPUT_DIR=xxx 覆盖。
# 【v6】默认输出改到 grpo/v6-self-asr 以与 v1~v5 严格隔离 (语义完全不同)。
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/meld/grpo/v6-self-asr"}
# ms-swift 默认 add_version=True, 会在 OUTPUT_DIR 后自动追加 v<idx>-<timestamp>
# 子目录 (见 swift/arguments/sft_args.py 中 _add_version)。设 ADD_VERSION=false
# 可让 checkpoints 直接落到 $OUTPUT_DIR 下, 便于外部脚本按固定路径消费。
# 注意：关闭 add_version 后，若同一 OUTPUT_DIR 重复启动训练会覆盖之前的
# checkpoints/logs, 请自行确认是否是想要的行为。
ADD_VERSION=${ADD_VERSION:-false}

# 【MELD 关键 · SYSTEM prompt · 对齐 UltraEval-Audio baseline】
# UltraEval-Audio 官方推理器 (audio_evals/models/step_audio_2_mini.py:180) 强制
# 注入 system="You are a helpful assistant."。SFT / 推理 / GRPO 都必须保持一致,
# 否则 policy 分布会因 prompt 前缀变化而漂移, 二次塌缩。
SYSTEM=${SYSTEM:-"You are a helpful assistant."}

# ---------------- Tuner ----------------
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=5e-6   # GRPO + full 建议比 lora 小 2~3× 防 KL 爆炸
else
    # 【v2 复盘】v1 用 1e-6 太保守, 三个 ckpt macro-F1 差异仅 0.7pp 说明梯度
    # 信号太弱走不动。v2 引入 accuracy_recall 后 group std 明显变大 (负 reward
    # 让 rare-class 组不再全组=0), advantage 稳定, 可回升到 1.5e-6 (仍比 v0 的
    # 2e-6 保守 25%)。
    DEFAULT_LR=1.5e-6
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（沿用 SFT/ASC 脚本）
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- GRPO 关键超参 ----------------
# NUM_GENERATIONS: 每个 prompt 采样多少 completion 做 group-relative advantage。
# 【v3 复盘】v2 用 G=12, disgust recall 仍卡在 13.2% (三个 ckpt 完全相同),
# 说明 disgust 出现在 group 里的样本数仍不足。G=16 再提升 33% 命中机会,
# 让 disgust rollout 至少 2~3 条/组能形成非退化 group std。
# 代价: 每 step 生成量再 +33%, 训练时间从 ~60s/step 到 ~80s/step。
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
# 单条 completion 最长生成多少 token。
# 【v6】self-ASR 格式下 assistant 需要吐:
#     <transcript>...MELD 单句 30~60 词...</transcript>\n<answer>xxx</answer>
# MELD 单句 subword ~ 60~120 token + XML 标签 ~ 10 token, 128 是安全上限。
# 若 16 (v1~v5 值) 会硬截 <transcript>, WER reward 恒 ~1, 训练无收益。
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
# 【v3 复盘】v2 用 TEMP=1.2, disgust recall 仍卡在 13.2% (完全不动),
# 说明 disgust 在 rollout 里出现的机会仍不够。抬到 1.3 进一步扩大采样池,
# disgust/fear 这种极低置信预测更容易被采到。
# 注意: TEMP 过高会退化到近似均匀采样, group std 反而下降, 边际衰减明显;
# 1.3 是 SFT 训练温度 (通常 1.0) 的 30%, 仍属安全区。
TEMPERATURE=${TEMPERATURE:-1.3}
# 【v2 复盘】v1 用 epsilon_high=0.22 已成功抑制 neutral 单侧漂移 (predicted
# neutral 从 82% 回落), 但 v2 引入负 reward 后 rare 类的正 advantage 需要
# 更充分反传 ("发现猜对 disgust 的价值"), 略放宽到 0.24; 与 LR=1.5e-6 匹配。
EPSILON=${EPSILON:-0.2}
EPSILON_HIGH=${EPSILON_HIGH:-0.24}
# 【v1 复盘】scale_rewards=group 会把每组 mean/std 归一, 5× vs 1× 的
# class weight 差被抹平 —— class weight 变成"名存实亡". 改用 none 让
# weighted reward 的绝对量差直接传到 advantage, 让 rare 类 5× 权重真正起作用。
SCALE_REWARDS=${SCALE_REWARDS:-none}
# 【v2 复盘 · REWARD 核心变更】v1 只用 stepaudio_accuracy_weighted (纯胡萝卜),
# rare-class 组 8/8 全错时 reward 全组=0, 无梯度信号, policy 不会往 rare 探索。
# v2 切换到 stepaudio_accuracy_recall (胡萝卜+棍子):
#   pred == gt              -> +w[gt]      (与 v1 同)
#   pred ∈ LAZY & gt ∉ LAZY -> -penalty    (惩罚偷懒猜 majority) ← 新增
#   其它 miss               ->  0.0
# 好处: rare-class 组即使 8/8 全错 (全都猜 neutral), 也会得到 -penalty,
# 组内引入非零方差, advantage 有梯度, policy 被迫尝试猜 rare 类。
# 若观察到副作用 (format 泄漏 / recall 过度冲高但 precision 崩), 可回退:
#   REWARD_FUNCS="stepaudio_accuracy_weighted" bash $0
#
# 【v6 · 新增 WER 辅助 reward】
# stepaudio_transcript_wer:
#     r = clip(1 - WER(pred_transcript_in_<transcript>, gold_asr_text), 0, 1)
# 只在 completion 含 <transcript>...</transcript> 且 gold asr_text 非空时给
# 正 reward; 缺一即 0.0 (不引负 reward, 不与 accuracy_recall 冲突)。
# 权重 0.3: 相对 recall 正 reward 均值 ~2 是小项, 相对 lazy_penalty=0.8
# 同数量级, 足以让 policy 感受到 "好好转写有额外收益"。
# 关闭方式: REWARD_FUNCS="stepaudio_accuracy_recall" REWARD_WEIGHTS="1.0" bash $0
REWARD_FUNCS=${REWARD_FUNCS:-"stepaudio_accuracy_recall stepaudio_transcript_wer"}
REWARD_WEIGHTS=${REWARD_WEIGHTS:-"1.0 0.3"}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# 【v6 · WER reward 环境变量 (可选覆盖)】
#   STEPAUDIO_WER_TOKENIZATION = word (默认) | char
#   STEPAUDIO_WER_LOWERCASE    = 1 (默认) | 0
#   STEPAUDIO_WER_STRIP_PUNCT  = 1 (默认) | 0
# 情感任务对拼写/标点不敏感, 默认 word + lowercase + strip_punct 最宽松,
# 给 policy 最容易拿到 WER 正 reward 的空间。
export STEPAUDIO_WER_TOKENIZATION=${STEPAUDIO_WER_TOKENIZATION:-word}
export STEPAUDIO_WER_LOWERCASE=${STEPAUDIO_WER_LOWERCASE:-1}
export STEPAUDIO_WER_STRIP_PUNCT=${STEPAUDIO_WER_STRIP_PUNCT:-1}

# 【v3 · LAZY 集合 · 修复 v2 对称豁免 bug】
# 定义"偷懒预测"标签集合: 当 gt 不在这里但 pred 在这里时给 -penalty。
# 【v5 复盘】v4 打开 STRICT=1 + LAZY={neutral,joy} 治 joy 泛滥"过度成功":
# joy recall 从 48.5 掉到 38.1 (方向对), 但 rare 3 类反而全面退步 (disgust
# 16.2→13.2, anger 15.4→12.5, sadness 19.2→15.9), macro-F1 从 v3 的 36.67
# 跌到 v4 的 34.82, 首次负增长。原因: STRICT 罚"猜 neutral"信号被稀释到
# 非 disgust group (占 6/7 batch), rare 类的正激励没能相对突出。
# v5 回到 v3 的 non-strict + 单 LAZY, 从 weight 侧治 joy 泛滥 (见下方 joy=0.6)。
export STEPAUDIO_LAZY_LABELS=${STEPAUDIO_LAZY_LABELS:-"neutral"}

# 【v4 新增 · LAZY_STRICT 模式】(v5 默认关闭, API 保留)
# STRICT=0 (v1/v2/v3/v5 默认, backwards-compatible):
#   pred ∈ LAZY & gt ∉ LAZY  -> -penalty  (只罚"猜 neutral 但真值是 rare")
#   pred ∈ LAZY & gt ∈ LAZY (mismatch, 如 gt=neutral,pred=joy) -> 0.0
# STRICT=1 (v4 曾用, v5 关闭):
#   pred ∈ LAZY & pred != gt  -> -penalty  (无论 gt 是否在 LAZY)
# v4 教训: STRICT=1 表面能治 joy 泛滥, 但会把 policy 推向"面对 non-neutral
# audio 就改猜 anger/surprise, 而非猜 rare 类" 的保守解, rare 类学习信号被
# 削弱。v5 关掉 STRICT, 从 class weight 侧 (joy=0.6) 治 joy 泛滥。
export STEPAUDIO_LAZY_STRICT=${STEPAUDIO_LAZY_STRICT:-"0"}

# 【v2 · LAZY_PENALTY】
# 惩罚强度。plugin 默认 0.5, v2 抬到 0.8:
#   * 与 disgust weight=5 相比, -0.8 只占 16%, 不会压垮 rare 探索的正激励;
#   * 但比 0.5 强 60%, 组内方差显著变大, 梯度真正流动;
#   * 与 neutral weight=0.5 差 1.6× (惩罚>猜对 majority 的正激励),
#     policy 会主动避免"猜 neutral 但 gt 非 neutral"。
# 若 rare recall 过度冲高但 precision 崩 (神经质乱猜 rare), 回落到 0.5。
export STEPAUDIO_LAZY_PENALTY=${STEPAUDIO_LAZY_PENALTY:-"0.8"}

# 【v1 复盘 · CLASS_WEIGHTS】v0 用 neutral=1.0 名义上不放大, 但由于
# scale_rewards=group 会把每组的 advantage 归一化, majority 组内一次
# 正确得 w=1.0 与 rare 组内一次正确得 w=5.0 归一后差距被抹平; 再加上
# neutral 训练样本量是 rare 类的 4×~18×, 累积梯度 majority 完胜。
#
# v1 双管齐下:
#   - scale_rewards=none 让 weight 绝对量差直接进 advantage;
#   - neutral 权重 1.0 → 0.5 (显式打折 majority 正 reward, 让"猜对 neutral"
#     不再等值于"猜对其他类"的一半);
#   - surprise 2.0 → 2.5, sadness 3.0 → 3.5 略提, 加大与 majority 的对比。
# 【v4 复盘 · CLASS_WEIGHTS 调整】v3 用 disgust=8/fear=8/joy=1.2/surprise=2.5,
# disgust 首次真正学到 (recall 11.8→16.2), 但同时暴露两个副作用:
#   * joy weight=1.2 + LAZY 对称豁免 → joy recall 34.6→48.5 (泛滥)
#   * surprise weight=2.5 > anger=2.0 → surprise recall 35.6→51.6 但 precision
#     从 48 掉到 36 (policy 用 surprise 当新避风港)
# v4 试图打开 STRICT 从 reward 侧堵 joy 泛滥 + 降 surprise weight, 结果
# STRICT 副作用把 rare 3 类打回 v2 水平 (见上方 v5 复盘)。
# 【v5 复盘】改用 class weight 单一手段治 joy 泛滥, 不再依赖 STRICT:
#   * joy weight 1.0 → 0.6 (与 neutral=0.5 差距只有 0.1, policy 猜 joy 没动机;
#     与 disgust=8 差 13×, joy 命中相对激励弱到不足以形成 mode-seeking);
#   * surprise weight 2.0 → 2.2 (v4 的 2.0 让 surprise recall 44.8 尚可但
#     离 SFT 46.3 差 -1.5pp, 略微回升);
#   * disgust/fear 保持 8.0 (v3 见效, v5 继续加大 gen_batch 让它更稳定);
#   * neutral=0.5 / anger=2.0 / sadness=3.5 保持 v3/v4 水平。
# 也可通过环境变量覆盖: STEPAUDIO_CLASS_WEIGHTS='neutral=1.0,...' bash $0
export STEPAUDIO_CLASS_WEIGHTS=${STEPAUDIO_CLASS_WEIGHTS:-"neutral=0.5,joy=0.6,anger=2.0,surprise=2.2,sadness=3.5,disgust=8.0,fear=8.0"}

# 抗 mode-collapse 开关
# 【v3 复盘 · BETA】v2 用 0.1, 但 joy 仍出现明显泛滥 (recall 34.6→39.8),
# 说明 KL 约束还不够压得住 policy 从 neutral→joy 的 mass 迁移。抬到
# 0.15 让 policy 变化更保守, 保护 rare 类 mass 不被 joy 抢走。
# 若 KL 变得过大导致 rare 类学习速度下降, 可回落到 0.12。
BETA=${BETA:-0.15}
# 【v1 复盘 · TOP_P/TOP_K】v0 用 0.95/40 但 8 条 rollout 里 rare 类仍常常
# 缺席, 放开到 1.0/0 (相当于 disable top-p/top-k, 完全依赖 temperature) 
# 增加 exploration。
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-0}
# 【v1 复盘 · DYNAMIC_SAMPLE】v0 用 true, 目的是丢弃 rare-class 全错组
# 加快梯度累积; 但反作用是同时也丢弃了大量"neutral 全对"的 std=0 组
# (这些组的价值恰恰是提供"neutral 也不能再自信"的 KL 约束). MELD 的
# neutral 47% + accuracy 89%, neutral 组约 30%+ 是 std=0, 一起丢掉后
# policy 更加"敢"往 neutral 偏。v1 改 false 保留全部组。
DYNAMIC_SAMPLE=${DYNAMIC_SAMPLE:-false}
LOG_ENTROPY=${LOG_ENTROPY:-true}

# ---------------- 稳定性 / 恢复训练（可选） ----------------
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-true}
else
    RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-}
fi

# ---------------- 训练规模 ----------------
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
# EVAL_BATCH_SIZE 必须满足：per_device_eval_batch_size × world_size 能被
# num_generations_eval（默认 = NUM_GENERATIONS）整除；下方 sanity check 会自动兜底。
# 【v3】NUM_GENERATIONS=16, 4 卡 × EVAL_BATCH_SIZE=4 = 16 刚好整除。
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
# 【v3 关键约束 · GRAD_ACCUM】TRL GRPOConfig.__post_init__ 强制要求:
#   generation_batch_size = per_device_train_batch_size × world_size × grad_accum
#                        必须能被 num_generations 整除
# 否则报错: "generation_batch_size (X) must be divisible by num_generations (Y)".
# 当前默认 (v5): BATCH_SIZE=1, world_size=4 (4 卡), NUM_GENERATIONS=16
#   → gen_batch = 1 × 4 × 16 = 64 = 4 × 16  ✓  (v5, 每 step 覆盖 4 个 prompt)
# 若改为 8 卡: gen_batch = 1 × 8 × 16 = 128 = 8 × 16  ✓
# 若改为 2 卡: gen_batch = 1 × 2 × 16 = 32 = 2 × 16  ✓
# 若改为 1 卡: gen_batch = 1 × 1 × 16 = 16 = 1 × 16  ✓
# 历史:
#   v1 (G=8)  gen_batch=32 (GRAD_ACCUM=8),  4×8;
#   v2 (G=12) gen_batch=36 (GRAD_ACCUM=9),  3×12;
#   v3 (G=16) gen_batch=48 (GRAD_ACCUM=12), 3×16;
#   v4 (G=16) gen_batch=48 (GRAD_ACCUM=12), 3×16;
#   v5 (G=16) gen_batch=64 (GRAD_ACCUM=16), 4×16.  ← 每 step 覆盖 4 个 prompt
# v5 复盘: v4 rare 类退步的核心原因之一是 gen_batch 只有 48, rare 类每 step
# 只有 ~2 条 rollout, group std 极小。抬到 64 (+33%) 让 rare 类曝光更稳定,
# 累计梯度更准。训练时间 +33%, 300 步约 6.5 小时 (v4 是 5 小时)。
# 下方 sanity check 会在启动前显式校验并给出人类可读的报错。
GRAD_ACCUM=${GRAD_ACCUM:-16}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
# 【v4 复盘 · MAX_STEPS】v3 跑 500 步, ckpt-250 最佳 (36.67), ckpt-500 已明
# 显过拟合回落 (35.55)。v4 早停到 300 步节省 40% 训练时间, 且减少挑"过拟合
# ckpt"的风险。密采样仍保持每 25 步一个 ckpt (300/25=12 个 ckpt)。
MAX_STEPS=${MAX_STEPS:-300}

# checkpointing：300 步 / 25 步 = 12 个 ckpt 密采样挑最佳
SAVE_STEPS=${SAVE_STEPS:-25}
EVAL_STEPS=${EVAL_STEPS:-25}
save_total_limit=${save_total_limit:-15}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---------------- 序列 / 截断 ----------------
# 【对齐 SFT-full 经验 · 2026-07-09】MAX_LENGTH 2048 → 4096。
# SFT-full ([run_train_sft_full_meld.sh]) 一直用 4096, MELD 单条对话
# p99 音频 ~15s + prompt/history token 边界接近 2048, 2048 会偶发截断
# rare-class 的关键 token; 抬到 4096 与 SFT-full 完全对齐, 也让 GRPO
# rollout 阶段的 prompt token 不会因截断而丢弃 rare-class 样本。
MAX_LENGTH=${MAX_LENGTH:-4096}
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-delete}

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
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore:TRL currently supports vLLM versions:ignore:You have version 0.17.1 installed"}

# ---------------- 数据集 (MELD word 版, 7 类) ----------------
# 【v1 复盘 · 数据源】v0 用原始 train.jsonl (47% neutral / 2.7% fear/disgust)
# 仅靠 weighted reward 不足以抵消采样偏置: majority 类每 batch 被采样次数
# 是 rare 类的 17×, 累积梯度仍然是 majority 主导. v1 默认切换到
# train.balanced.jsonl (oversample 到均值 ~1428/类, 各类占比拉平到 ~10-15%),
# 让"每 step 的 group 平均结构"就已经均衡, weighted reward 只做二次微调。
#
# 自动生成: 若 balanced 版不存在, 调用 data_meld/build_meld_train_balanced.py
# (默认 --strategy oversample --target mean --seed 42) 生成。
# 如需退回原始不均衡: TRAIN_JSONL=$DATA_DIR/train.jsonl bash $0
#
# 【v6 · 默认切换到 self-ASR 数据集】
# assistant 输出结构:
#     <transcript>...gold ASR from MELD CSV...</transcript>\n<answer>xxx</answer>
# 顶级 asr_text 字段供 stepaudio_transcript_wer reward 消费。
# 若需要旧的 label-only 数据: TRAIN_JSONL=$DATA_DIR/train.jsonl bash $0
# 若需要 balanced 版 self-ASR (需先自行造):
#     TRAIN_JSONL=$DATA_DIR/train.r1omni_self_asr.balanced.jsonl bash $0
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.r1omni_self_asr.jsonl"}
if [ ! -f "$TRAIN_JSONL" ]; then
    # 仅当默认路径 (balanced) 不存在时尝试自动生成; 如果用户显式指定了
    # 一个不存在的路径则直接报错。
    if [ "$TRAIN_JSONL" = "$DATA_DIR/train.balanced.jsonl" ] && [ -f "$DATA_DIR/train.jsonl" ] \
       && [ -f "$DATA_DIR/build_meld_train_balanced.py" ]; then
        echo "[INFO] $TRAIN_JSONL 不存在, 自动调用 build_meld_train_balanced.py 生成..."
        _BAL_PY="${CONDA_SWIFT_PY:-python3}"
        # 若上面探测到的 python 尚未初始化, 就先用 python3 兜底
        if ! command -v "$_BAL_PY" >/dev/null 2>&1; then
            _BAL_PY=python3
        fi
        if ! "$_BAL_PY" "$DATA_DIR/build_meld_train_balanced.py" \
                --in_jsonl "$DATA_DIR/train.jsonl" \
                --out_jsonl "$TRAIN_JSONL" \
                --strategy oversample \
                --target mean \
                --seed 42; then
            echo "[FATAL] 自动生成 balanced 训练集失败, 请手动运行:"
            echo "        $_BAL_PY $DATA_DIR/build_meld_train_balanced.py \\"
            echo "            --in_jsonl $DATA_DIR/train.jsonl \\"
            echo "            --out_jsonl $TRAIN_JSONL --strategy oversample --target mean"
            exit 14
        fi
    fi
fi
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[FATAL] TRAIN_JSONL 不存在: $TRAIN_JSONL"
    echo "        MELD 训练数据应位于 $DATA_DIR/train.jsonl (word 版, 7 类)。"
    echo "        若被误删，请先跑 build_meld_jsonl.py 生成。"
    echo "        也可显式:  TRAIN_JSONL=$DATA_DIR/train.jsonl bash \$0  # 退回原始不均衡"
    exit 14
fi
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0.01}

# ---------------- Reward / Plugin ----------------
PLUGIN_PATH=${PLUGIN_PATH:-"$SWIFT_ROOT/examples/train/grpo/plugin/stepaudio/stepaudio_plugin.py"}
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "[FATAL] reward plugin 不存在: $PLUGIN_PATH"
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
echo "[INFO] STEPAUDIO_CLASS_WEIGHTS=$STEPAUDIO_CLASS_WEIGHTS"
echo "[INFO] STEPAUDIO_LAZY_LABELS=${STEPAUDIO_LAZY_LABELS:-<unset>}  STEPAUDIO_LAZY_PENALTY=${STEPAUDIO_LAZY_PENALTY:-<unset>}  STEPAUDIO_LAZY_STRICT=${STEPAUDIO_LAZY_STRICT:-<unset>}"
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
# TOP_K=0 表示禁用 top-k (纯 temperature 采样), 有些 trainer 会对 0 报错,
# 所以只在 TOP_K>0 时传给 swift。
if [ -n "$TOP_K" ] && [ "$TOP_K" != "0" ]; then
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
# ---- Sanity check（v2 新增）：generation_batch_size 必须能被 num_generations 整除 ----
# TRL GRPOConfig.__post_init__ 会强制这个断言（trl/trainer/grpo_config.py:856）：
#   ValueError: generation_batch_size (X) must be divisible by num_generations (Y).
# 其中:
#   generation_batch_size = per_device_train_batch_size × world_size × grad_accum
# 与 eval 侧不同, TRL 目前没有"降级 train num_generations"的参数放行途径,
# 所以我们只能在启动前显式失败, 并算出最近的可行 GRAD_ACCUM 建议给用户。
# 之前 v1 用 G=8 时 gen_batch=1×4×8=32 恰好整除, v2 改成 G=12 后 32 % 12 ≠ 0
# 直接 crash (见 rank1/rank3 报错). 现在把 GRAD_ACCUM 默认改成 9 (gen_batch=36),
# 但用户如果通过 BATCH_SIZE / GRAD_ACCUM / NPROC 覆盖了默认值, 这里再兜底一次。
_gen_batch=$(( BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM ))
if [ "$_gen_batch" -gt 0 ] && [ "$NUM_GENERATIONS" -gt 0 ]; then
    if [ $(( _gen_batch % NUM_GENERATIONS )) -ne 0 ]; then
        # 找一个"最接近当前 GRAD_ACCUM 的可行 GRAD_ACCUM": 让 gen_batch 是 NUM_GENERATIONS 的倍数
        # 需要 GRAD_ACCUM × (BATCH_SIZE × NPROC) 能被 NUM_GENERATIONS 整除
        # 即 GRAD_ACCUM 需能被 (NUM_GENERATIONS / gcd(BATCH_SIZE × NPROC, NUM_GENERATIONS)) 整除
        _bs_world=$(( BATCH_SIZE * NPROC_PER_NODE ))
        _g=$(_gcd "$_bs_world" "$NUM_GENERATIONS")
        _step=$(( NUM_GENERATIONS / _g ))
        # 向下取整的可行 GRAD_ACCUM (若为 0 则向上取一档)
        _down=$(( (GRAD_ACCUM / _step) * _step ))
        [ "$_down" -lt "$_step" ] && _down=$_step
        _up=$(( _down + _step ))
        echo ""
        echo "[FATAL] ============================================================"
        echo "[FATAL] TRL 约束不满足: generation_batch_size ($_gen_batch) 无法被 num_generations ($NUM_GENERATIONS) 整除"
        echo "[FATAL] "
        echo "[FATAL] 计算: BATCH_SIZE ($BATCH_SIZE) × NPROC ($NPROC_PER_NODE) × GRAD_ACCUM ($GRAD_ACCUM) = $_gen_batch"
        echo "[FATAL] "
        echo "[FATAL] 请从以下几种方式选一个修复:"
        echo "[FATAL]   (推荐) GRAD_ACCUM=$_down  bash \$0   # gen_batch=$((_bs_world * _down))"
        echo "[FATAL]   (或者) GRAD_ACCUM=$_up    bash \$0   # gen_batch=$((_bs_world * _up))"
        echo "[FATAL]   (或者) NUM_GENERATIONS 选一个能整除 $_gen_batch 的值 (如 $(_gcd "$_gen_batch" "$NUM_GENERATIONS"))"
        echo "[FATAL] ============================================================"
        exit 17
    fi
    echo "[INFO] gen_batch check OK: gen_batch=$_gen_batch (= $BATCH_SIZE × $NPROC_PER_NODE × $GRAD_ACCUM), num_generations=$NUM_GENERATIONS, prompts/step=$((_gen_batch / NUM_GENERATIONS))"
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
    --system "$SYSTEM" \
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
    --label_smoothing_factor 0.0 \
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

echo "[INFO] MELD GRPO 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
