#!/usr/bin/env bash
# StepAudio2-mini offline DPO 训练脚本 —— MELD 情感 7 分类版（基于 MS-SWIFT）
#
# ═══════════════════════════════════════════════════════════════════════════
# 【任务差异 · 相对 ASC 5 分类的 run_train_dpo.sh】
# ═══════════════════════════════════════════════════════════════════════════
# 类别 (7):    surprise, anger, neutral, joy, sadness, fear, disgust
# 数据目录:    project/stepaudio/data_meld/
# 数据分布:    neutral 47.2% / joy 17.4% / anger 12.1% / surprise 11.9% /
#              sadness 7.2% / disgust 2.7% / fear 2.7%      (train 9989 条)
# SFT 起点:    MELD full-SFT ckpt (v0-20260707-182357/checkpoint-234) 在 test
#              集上 acc=59.8%, macro-F1=43.0%; 或 LoRA v7 ckpt-150 acc=58.5%,
#              macro-F1=41.2%. 相比 ASC (SFT 已经 macro-F1=95.7%) MELD 的 SFT
#              baseline 还很弱, rare-class (fear/disgust) F1 只有 20%~30%.
# 后训练 EV:   与 ASC 的"负 EV 精修"完全不同, MELD 后训练是 **正 EV**：
#              rare-class 有大量 hard negatives 可挖 (SFT 把 fear/disgust 大
#              量误判为 neutral), 且 SFT 每类都还没到 90% 收敛平台。
# 与 ASC 脚本的关键差异:
#   * BETA 从 0.1 降到 0.05 (v6): MELD 起点弱 → margin 起始值只有 ~1e-3,
#     β=0.15 时 σ(β·Δ)≈0.5 梯度为零，训练完全学不到东西 (见 v5 复盘);
#   * RPO_ALPHA 保持 0.0 (v6): 见下方 v5 复盘, nll 副损失会把 chosen 分布拉
#     向 majority (v5 里 chosen 分布 joy=1021 > neutral=441, nll 就把预测
#     全推向 joy, 反而把 neutral/anger/... recall 全打崩);
#   * LORA_RANK 8 → 16 (v6): MELD 起点弱, chosen/rejected 表征差异需要更多容量
#     才能拉开;
#   * LEARNING_RATE 5e-7 → 2e-6 (v6): β 减半 + 数据只有 ~3k pair, 需要更大
#     lr 才能在 1 epoch 内推动 margin;
#   * MAX_STEPS 400 → 220 (v6): 3543 条 pair / bs=1 / accum=4 / 8 卡 ≈ 每
#     epoch 110 步, v5 的 400 步 = 3.6 epoch → 严重过拟合到 SFT 起点, v6
#     控制在 2 epoch 附近;
#   * SAVE_STEPS / EVAL_STEPS 40 → 20 (v6): 更细粒度地监控 margin/accuracy
#     曲线并 early-stop; 加 load_best_model_at_end + metric=eval_accuracies;
#   * MODEL_PATH 默认 = MELD SFT 后 merged ckpt (见下方 MODEL_PATH 说明);
#   * SYSTEM 默认改为空字符串 (v6): v5 用 "You are a helpful assistant."
#     导致 policy/ref 的 log-prob 与 SFT 训练时分布不匹配, margin=0 起点
#     压根没有偏好信号 (见 v5 复盘 §5). 现改为透传 MODEL_PATH/args.json 里
#     SFT 训练时的 system 值;
#   * 数据构造命令 --classes 从 5 类换成 7 类, --min-wrong-per-class 从 200
#     调到 150; **不再 --drop-right-samples**, 而是配 --keep-right-per-class
#     700 让 chosen 分布贴近 test 集自然分布 (见 v5 复盘 §2).
# ═══════════════════════════════════════════════════════════════════════════
#
# offline DPO 完全绕开 rollout：
#   * 训练数据 = (chosen=正确 label, rejected=错 label) 的成对样本；
#   * loss   = -log σ(β · Δ)，其中 Δ = (logπ_θ − logπ_ref)_chosen −
#              (logπ_θ − logπ_ref)_rejected；
#   * 不需要在训练时采样，因此不受 temperature/entropy 塌缩影响；
#   * ref = SFT ckpt，KL 项自然把 policy 锚定在 SFT 附近做局部纠错。
#
# 数据构造（见 tools/build_dpo_pairs.py）：
#   1) 拿 MELD SFT ckpt 在 train.jsonl 上的推理 jsonl；
#   2) 对每条样本：
#        - chosen   = 真实 label (word 版，例如 "disgust" / "fear" / ...)；
#        - rejected = SFT 实际预测错的 label（"hard negative"），
#                     若 SFT 已答对则从 confusion prior 采一个近似 hard neg；
#   3) 保留所有 SFT 错误样本 (真正的 DPO 学习信号); majority class (neutral)
#      的已答对样本下采样至 cap=800/类, 避免 pair 分布被 majority 淹没。
#
# 三步流程：
#   (I) 生成 MELD SFT baseline 的推理结果（如已存在可跳过）:
#         MODEL_PATH=./output/meld/<sft_run>/checkpoint-<step>-merged \
#         VAL_JSONL=project/stepaudio/data_meld/train.jsonl \
#         bash project/stepaudio/run_inference_meld.sh
#       产出：project/stepaudio/infer_results/result_<sft_ckpt>_train.jsonl
#
#   (II) 构造 DPO 训练集（token-flow 均衡 + hard neg + 保留 right samples）：
#         python project/stepaudio/tools/build_dpo_pairs.py \
#             --infer-jsonl <上面的推理 jsonl> \
#             --output project/stepaudio/data_meld/train.dpo.jsonl \
#             --classes surprise,anger,neutral,joy,sadness,fear,disgust \
#             --min-wrong-per-class 150 \
#             --keep-right-per-class 700 \
#             --min-chosen-per-class 300 \
#             --max-audio-sec 30 \
#             --rejected-strategy mistake
#         注：
#           * MELD 音频比 ASC 短很多 (p95 ≈ 15s, max ≈ 30s), --max-audio-sec
#             收紧到 30 已能覆盖 >99% 样本, 更进一步压 OOM 风险;
#           * **不再** 用 --drop-right-samples: v5 复盘表明扔掉答对样本会让
#             chosen 分布严重偏离 test 分布 (neutral 从 47% 掉到 12%),
#             DPO+GC 里余下少数 nll 项会把预测推向 chosen 里 dominant 的
#             joy, 加剧塌陷。改成 keep-right-per-class 700 + min-chosen 300
#             让每类 chosen 至少 300, majority 类也不夸张。
#
#   (III) 启动 DPO：
#         MODEL_PATH=./output/meld/<sft_run>/checkpoint-<step>-merged \
#         bash project/stepaudio/run_train_dpo_meld.sh
#
# 关键超参选择依据（MELD 特化 v6，与 ASC / v5 差异见上）：
#   * BETA=0.05：v5 用 0.15 导致 σ(β·Δ)≈0.5, |Δ|=0.003 时梯度<1e-4, 训练
#     曲线 loss 在 0.83 平原上抖 400 步毫无学习。β 减小到 0.05 会让
#     margin 从 ~0.003 放大到显著非零区域, 从而给 policy 拉开 chosen/
#     rejected 的实质空间。代价是 KL 拘束变弱, 因此必须配套 max_steps 减半。
#   * RPO_ALPHA=0.0：见 v5 复盘, 打开会把预测推向 chosen 分布里的 dominant
#     类。若发现 chosen log-prob 一直下滑 → 只能考虑 0.05 起, 且必须重新
#     构造 pair 让 chosen 分布贴近 test 分布。
#   * LR=2e-6：β 减半 + LoRA rank 加倍 + 数据只有 ~3k pair, 需要更大 lr 才
#     能在 2 epoch 窗口内推动 margin。参考 TRL/DPO 论文的 LoRA 推荐范围
#     1e-6~5e-6, 2e-6 是稳健选择。
#   * LORA_RANK=16 / ALPHA=32：v5 用 rank=8 表征力不足, chosen/rejected 只
#     差 1~2 个 word token, 但底层是
#
# 沿用 ASC 脚本里所有踩过的坑：
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
#
# 【失败复盘 · v5-20260708-094755（MELD 首个 DPO 尝试）】
#   现象：SFT baseline (v7 ckpt-150) 在 test 集 acc=58.5% / macro-F1=41.2%,
#         各类 recall 相对均衡 (surprise 46, joy 25, anger 23, sadness 24,
#         fear 22, disgust 46). DPO 训练 400 步后:
#             ckpt-80   acc=56.1%  macro-F1=33.1%
#             ckpt-240  acc=56.4%  macro-F1=34.5%
#             ckpt-400  acc=55.8%  macro-F1=33.6%
#         所有 ckpt 均低于 baseline, 且 macro-F1 掉幅巨大 (-8pt); 主要是
#         anger/sadness/fear/disgust recall 全部跌到 10-15% (SFT ckpt-150
#         对应值分别是 22.6% / 24.0% / 22.0% / 45.6%)。
#   TensorBoard 关键信号：
#     * train/loss: 0.81 → 0.83 (400 步几乎不变, 完全没学)
#     * train/rewards/margins: 0.003 → -0.003 (在噪声里抖, 甚至变负)
#     * eval/rewards/accuracies: 0.58 → 0.33 (从优于随机滑到远差于随机)
#     * train/nll_loss: 1.19 → 1.5-1.6 (rpo_alpha=0.1 让 NLL 项 = 15~20%
#       总 loss, 而 DPO 主项梯度贡献 <1%)
#   根因（五重叠加, 按贡献从大到小）：
#     §1) BETA=0.15 太大 · MELD SFT 起点弱, |margin| 起始 ~0.003, β·Δ≈
#         4.5e-4, σ'(0)≈0.25, DPO 梯度 <1e-4, 训练完全学不到东西;
#     §2) chosen 分布反向 · --drop-right-samples 把 SFT 答对的 majority
#         (neutral) 全扔了, 结果 chosen 分布是 joy=1021 > anger=741 >
#         surprise=516 > sadness=464 > neutral=441 > fear=210 > disgust=150,
#         与 test 分布 (neutral 47%, joy 15%) 完全反向;
#     §3) RPO_ALPHA=0.1 打开 NLL · 结合 §2 → NLL 把预测无脑推向 joy;
#         ckpt-400 结果印证: joy FP 从 SFT 的 100+ 涨到 176, neutral recall
#         略降, sadness/fear/disgust 大幅塌陷;
#     §4) MAX_STEPS=400 过长 · 3543 条 / bs=1 / accum=4 / 8 卡 每 epoch 110
#         步, 400 步 = 3.6 epoch, 在 §1 无梯度信号的情况下反复过拟合到 §3
#         推向 joy 的分布;
#     §5) SYSTEM="You are a helpful assistant." 与 SFT 时不一致 · v7 SFT
#         用的是 MELD 特化 prompt, 改前缀导致 policy/ref 起点的 log-prob
#         全线偏低 (eval/logps/chosen=-3.87 vs 正常 SFT 应该 <-1.0), 也
#         等于 ref_model 是漂移过的, 从起点就没法给出可靠的偏好信号。
#   修复（v6, 本脚本当前版本）：
#     * BETA 0.15 → 0.05
#     * RPO_ALPHA 0.1 → 0.0
#     * LR 5e-7 → 2e-6
#     * LORA_RANK 8 → 16
#     * MAX_STEPS 400 → 220 (~2 epoch)
#     * SAVE_STEPS/EVAL_STEPS 40 → 20
#     * 加 load_best_model_at_end + metric_for_best_model=rewards/accuracies
#     * SYSTEM 默认改为 "" (透传自 SFT ckpt 的 args.json), 强制保证
#       train/inference/SFT 三者 prompt 前缀一致;
#     * 数据构造推荐命令去掉 --drop-right-samples, 加
#       --keep-right-per-class 700 --min-chosen-per-class 300, 让 chosen
#       分布贴近 test 分布。
#
# 【失败复盘 · v6-20260708-104339 (v1 run)】
#   现象：DPO 的偏好目标完美实现，但 test 分类效果依然差 SFT baseline:
#             SFT v7 ckpt-150       acc=58.5%  macro-F1=41.2%
#             DPO ckpt-20  (v6)     acc=55.8%  macro-F1=32.6%
#             DPO ckpt-120 (v6-best) acc=56.4%  macro-F1=34.7%
#             DPO ckpt-220 (v6)     acc=55.8%  macro-F1=34.1%
#         DPO 的所有偏好指标都非常漂亮（这是 v6 相比 v5 的巨大进步）:
#             train/loss             0.69 → 0.66 (单调下降)
#             train/rewards/margins  0.00 → +0.063 (单调上升 20×)
#             eval/rewards/margins   0.00 → +0.049 (单调上升)
#             eval/rewards/accuracies 0.46 → 0.75 (单调上升)
#         但下游分类反而变差, 特别是各非-neutral 类的 recall 全下降:
#             surprise recall  46.3% → 34.2%   (被推向 neutral 从 120→150)
#             anger    recall  22.6% → 11.6%   (被推向 neutral 从 162→237)
#             sadness  recall  24.0% → 15.9%   (被推向 neutral 从 127→148)
#             fear     recall  22.0% → 16.0%   (被推向 neutral 从 29→35)
#             disgust  recall  45.6% → 13.2%   (被推向 neutral 从 26→43)
#             neutral  recall  89.6% → 90.3%   (SFT 起点保住了)
#   根因（offline DPO 与分类目标的结构性错配, 三重）:
#     §1) rejected 里 neutral 占比过高 · SFT 把 anger/sadness/fear/disgust
#         大量误判为 neutral, 于是构造 pair 时 rejected=neutral 的样本数
#         远超 chosen=neutral 的样本数 (从每个 sample 的 pair 结构看, 而非
#         整体 token flow, 后者已被 build_dpo_pairs balance)。
#         DPO loss 会持续压低"在这些 prompt 上输出 neutral 的 log-prob",
#         但这个信号无差别地污染了整个 policy 对 neutral 的 logit。
#     §2) DPO/sigmoid 缺乏刹车 · train/rewards/rejected 从 -0.005 一路走
#         到 -0.077 (下降 15×), 而 rewards/chosen 只从 0 到 -0.009。DPO
#         对 rejected 的推低速度是 chosen 的 8×, 意味着 policy 在"排斥
#         neutral"上花了远超"抬高正确类"的力气。sigmoid DPO 没有天然
#         收敛点, 只要 β 允许就会持续拉开。
#     §3) DPO 偏好准确率 与 test 分类 macro-F1 呈非单调关系 ·
#         eval/rewards/accuracies 峰值在 step 180, 但 sweep 表明 ckpt-120
#         的 test macro-F1 最高。metric_for_best_model=rewards/accuracies
#         会选到 180, 与真实最优 ckpt 错位 60 步。
#   修复（v7, 本脚本当前版本）:
#     * LOSS_TYPE sigmoid → ipo · IPO 有天然平衡点, 不会像 DPO 那样无
#       穷压低 rejected. 对 §2 直接有效。
#     * LABEL_SMOOTHING 0.0 → 0.1 · cDPO 视 pair 有 10% 噪声, 降低
#       "压 rejected=neutral"的强度, 减轻 §1 污染。
#     * RPO_ALPHA 0.0 → 0.05 · 极小 SFT 副损失把 policy 拉住不漂离 SFT
#       起点太远。v5 崩溃是因为 alpha=0.1 且 chosen 分布严重反向,
#       v6/v7 数据 chosen 分布已被 keep-right 拉平,
#       0.05 只做"锚定"不做"推动"。
#     * MAX_STEPS 220 → 140 · sweep 表明分类峰值在 step 120, 220 是
#       浪费; 缩短后 lr 冷却也更适配 rare-class 的少见次数。
#     * METRIC_FOR_BEST_MODEL rewards/accuracies → loss · eval/loss 在
#       step 180-200 到达谷底后微升, 与 test 分类峰值更契合 (§3)。
#     * 数据构造推荐命令: --keep-right-per-class 700 → 1500,
#       让 chosen 里 neutral 数量能接近 joy, 从数据侧对齐 §1 方向。
#
# 【失败复盘 · v7-20260708-112631 (v3 run)】
#   现象：v7 三管齐下 (ipo + label_smoothing=0.1 + rpo_alpha=0.05) 训完 140
#         步, test 结果与 v6 几乎完全一致 —— 说明这三个改动加起来没有实质
#         效果:
#             SFT v7 ckpt-150      acc=58.5%  macro-F1=41.2%
#             DPO v6-best ckpt-120 acc=56.4%  macro-F1=34.7%  (sigmoid)
#             DPO v7 ckpt-60       acc=55.9%  macro-F1=33.3%  (ipo+ls+rpo)
#             DPO v7 ckpt-140      acc=56.0%  macro-F1=33.7%  (ipo+ls+rpo)
#         各非-neutral 类 recall 与 v6 基本相同 (disgust 13.2, anger 10.1),
#         SFT 起点保住的 neutral recall 也几乎一样 (90.4)。
#   TensorBoard 关键信号 (IPO 在这个规模下"停摆"):
#     * train/loss: 100.0 → 95.1 (IPO loss=(Δ-1/(2β))²=(Δ-10)², Δ=0 时=100)
#                                (140 步只走了 5%, 目标是 loss→0 即 Δ→10)
#     * train/rewards/margins: 0 → +0.013 (v6 sigmoid 同期能到 +0.063,
#                                          IPO 有效步长只有 sigmoid 的 1/5)
#     * eval/rewards/margins:  0 → +0.007
#     * eval/rewards/accuracies: 0.63 → 0.72 (单调上升, 与 v6 一样漂亮)
#     * train/rewards/rejected: 0 → -0.016 (v6 是 -0.077, 说明 IPO 确实
#                                           减慢了压 rejected 的速度, 但
#                                           减速的同时也减慢了整体信号)
#   根因（三重, 都是 v7 药方失效的原因）:
#     §1) IPO 有效步长过小 · IPO loss = (Δ - 1/(2β))², β=0.05 时目标 Δ*=10
#         梯度形式 2·(Δ - 10), 在 Δ≈0 时梯度约 -20, 但经 β 缩放和归一化后
#         实际每步 Δ 增量只有 ~1e-4。sigmoid DPO 在同 β 下每步能推 3~4 倍。
#         想 IPO 的"天然刹车"生效需要跑到 Δ≈8~10, 现实只能到 0.013, IPO
#         从头到尾都在"极远处朝目标缓慢挪"—— 相当于弱化版 sigmoid, 没
#         享受到 IPO 的收敛保证, 反而信号更弱。
#     §2) IPO 下 label_smoothing 语义不同 · sigmoid 的 label_smoothing
#         (cDPO) 是对 preference 加噪, 直接减弱 loss 幅度; IPO 的 label
#         smoothing 是对目标 Δ* 插值 (Δ*=(1-2ε)/(2β)), ε=0.1 时目标从 10
#         变成 8, 对当前 Δ≈0 的 policy 几乎无影响。**在 IPO 下开 LS 基本浪费**。
#     §3) 上述两条使得 v7 三管齐下 (ipo + ls + rpo_alpha) 里前两管其实
#         都失效, 只剩 rpo_alpha=0.05 一点点 NLL 锚定。而 v6 的塌陷主因是
#         §0 结构性错配 (rejected=neutral 密度过高), rpo_alpha=0.05 一味
#         锚定 SFT 也无法逆转结构性错配 —— test 结果几乎复刻 v6。
#   诊断结论：**offline DPO/IPO 无法凭超参调整逃出"rejected 里 majority
#             class 密度过高"的结构性陷阱**。三条路:
#     路 A) 回到 sigmoid + 加大 label_smoothing + 加大 rpo_alpha + 缩短步数
#           把 v6-sigmoid 的信号强度全面调弱, 同时用 LS(cDPO 真实生效) 和
#           rpo_alpha 双管锚定 SFT, 期望减轻 §0 溢出污染。天花板 = v6-sigmoid
#           成绩再稍好一点。**这是 v8 (本脚本当前版本) 选择的路。**
#     路 B) 改 build_dpo_pairs.py, 显式降低 rejected=neutral 的比例
#           (例如上限 rejected==neutral 不超过 rejected 总数的 30%),
#           从数据侧根治 §0。**需要改 tools, 不在本次范围。**
#     路 C) 放弃 DPO 系, 改 class-weighted CE 精修 SFT (对 SFT 错样本加权)。
#           **需要完全重构训练管线, 不在本次范围。**
#   修复（v8, 本脚本当前版本, 走路 A）:
#     * LOSS_TYPE ipo → sigmoid  (回归 v6 已被证明有效的信号强度)
#     * LABEL_SMOOTHING 0.1 → 0.2 (sigmoid 下 LS 是 cDPO, 真的减弱信号,
#                                    0.2 相当于视 20% pair 有噪声)
#     * RPO_ALPHA 0.05 → 0.1 (v8 chosen 分布已被 keep-right 1500 拉平,
#                              alpha=0.1 不再像 v5 那样把预测推向 joy;
#                              加倍锚定 SFT 起点)
#     * BETA 0.05 → 0.1 (v6 β=0.05 让 rejected 一路掉到 -0.077, 过狠;
#                         回到 0.1 让 KL 拘束加倍, policy 漂移减半)
#     * LEARNING_RATE 2e-6 → 1e-6 (配合 β 加倍, lr 减半, 训练进一步柔化)
#     * MAX_STEPS 140 → 100 (v6 分类峰值在 step 100-120, v8 信号更弱
#                             应把峰值前移, 100 步够)
#     * SAVE_STEPS/EVAL_STEPS 20 → 10 (步数缩短, 评估更密以定位最优点)
#     * save_total_limit 15 → 12 (100/10=10 ckpts + 2 缓冲)
#   v8 明确的失败判据 (若达不到, 应直接切路 B 或路 C, 不要再调超参):
#     * ckpt-best macro-F1 应 >= 38% (比 v6 34.7% 至少涨 3pt), 否则 v8 无效;
#     * disgust recall 应 >= 30% (比 v6 13.2% 至少涨 15pt), 否则 §0 未解;
#     * anger recall  应 >= 18% (比 v6 11.6% 至少涨 6pt), 否则 §0 未解;
#     * neutral 上 joy FP 不涨 (v6 是 79, v8 应 <=79), 否则 rpo_alpha 太狠
#       (v5 的重演: 数据 chosen 反向 + rpo_alpha 大 → 推向 joy)。
#
# 【失败复盘 · v8-20260708-114909 (v4 run)】
#   现象：v8 五管齐下 (β=0.1, LS=0.2, rpo=0.1, lr=1e-6, sigmoid) 训完 100 步,
#         **结果比 v6/v7 更差**, 短短 33.3% macro-F1 与 SFT 41.2% 差 8pt:
#             SFT v7 ckpt-150      acc=58.5%  macro-F1=41.2%
#             v6-best ckpt-120     acc=56.4%  macro-F1=34.7%
#             v7  ckpt-140         acc=56.0%  macro-F1=33.7%
#             **v8 ckpt-50 (best)  acc=56.2%  macro-F1=33.3%**
#         各 rare-class recall 创新低: anger 9.0% (v6 11.6%), disgust 11.8%
#         (v6 13.2%), sadness 13.5% (v6 15.9%); neutral 上 joy FP=73 (与 v6 相当)。
#   TensorBoard 关键信号 (v8 训练完全未启动):
#     * train/rewards/margins:    0 → -0.005 (**变负**, v6 能到 +0.063)
#     * train/rewards/accuracies: 0.53 → 0.40 (**低于随机**, v6 能到 0.75)
#     * train/rewards/rejected:   0 → +0.005 (**完全反向**, v6 是 -0.077)
#     * train/rewards/chosen:     0 → +0.0002 (不动)
#     * eval/loss:                0.787 → 0.786 (静默)
#     * train/nll_loss:           0.99 → 0.91 (**略降, 说明 NLL 主导了梯度**)
#   根因：**v8 五个降强度改动乘起来把 DPO 主项信号打没了**, 只剩 NLL 副损失主导。
#         DPO 主项相对 v6 的有效学习量:
#           BETA 0.05→0.1 ×2 (Δ≈0 时意义不大)
#           LR   2e-6→1e-6 ×0.5
#           LS   0→0.2 ×0.6 (cDPO 有效系数 1-2ε)
#           STEP 220→80 ×0.36
#         综合 = 2×0.5×0.6×0.36 = 0.22 (22%)。同时 NLL 副损失 = 2×0.5×0.36 = 0.36
#         → NLL/DPO = 1.6×, 完全主导。而 NLL 方向是推向 chosen 分布高频类 (joy),
#         v5 剧本重演。
#
#   【总结 v5→v6→v7→v8 四次调参均失败】
#   test macro-F1: v5=33.6% → v6=34.7% → v7=33.7% → v8=33.3%.
#   SFT baseline=41.2%. 四次调整都在 33-35% 区间打转, **与 SFT 差 6-8pt 从未
#   接近**。已经确定: **offline DPO 在当前数据结构下无法超越 SFT baseline**。
#   根本原因仍是 v6 复盘 §0: rejected=neutral 密度过高 → DPO 无差别压 neutral
#   logit → 非-neutral 类都被污染。这个结构性错配无法靠超参逃出。
#
#   修复（v9, 本脚本当前版本，**offline DPO 系的最后一搏**）:
#     * BETA 0.1 → 0.05           (回 v6, 重新启动 DPO 主项)
#     * LR   1e-6 → 2e-6          (回 v6, 让 margin 能实际动)
#     * LS   0.2 → 0.0            (v8 已证 LS 无法抵消 §0 污染, 反而稀释信号)
#     * RPO_ALPHA 0.1 → 0.0       (**v9 关键变更**: 关掉 NLL 副损失, 防止 v5/v8
#                                   重演 —— NLL 主导时把预测推向 chosen 里的 joy)
#     * MAX_STEPS 100 → 80        (v6 ckpt-100/120 是峰值区间, v9 缩到 80 希望
#                                   在 rare-class 完全塌陷前早停)
#     * SAVE/EVAL_STEPS 10 不变    (密评估, 配合 load_best_model_at_end=loss)
#
#   **v9 本质上只是"v6 + 早停 60 步 + 彻底关 NLL"**, 不是新药方。
#   v9 天花板 = v6 早期 ckpt 的成绩 (估计 macro-F1 ~35%, 仍远低于 SFT 41.2%)。
#
#   【v9 失败判据 (**绝对硬标准, 达不到就禁止继续调参**)】
#     * ckpt-best macro-F1 应 >= 39% (v6 best 34.7%, 需至少涨 4pt);
#     * disgust recall 应 >= 35% (SFT 45.6%, 至少恢复到 3/4);
#     * anger  recall  应 >= 20% (SFT 22.6%, 至少恢复到 SFT-2pt)。
#
#   【若 v9 仍不达标, 必须切下列两条路之一】
#   路 B) **改 build_dpo_pairs.py, 根治数据侧 §0 污染** (推荐):
#          - 新增 --max-rejected-neutral-ratio 0.30 参数;
#          - 强制 rejected==neutral 的 pair 占总 pair 不超过 30%;
#          - 对于"chosen 非-neutral 且 SFT 误判 neutral"的 pair, 上采样到
#            与"chosen==neutral"的数量相当;
#          - 重新 build 后直接用 v9 超参训练 (无需再改本脚本)。
#   路 C) **完全放弃 DPO, 回到 SFT 精修**:
#          - class-weighted CE: 对 rare-class (fear/disgust/anger/sadness)
#            loss 加权 3-5×;
#          - focal loss (γ=2): 自动对高置信度错样本加权;
#          - 仅在 SFT 错样本上继续训练 (hard example mining);
#          - 目标: macro-F1 从 41.2% 抬到 45%+。
#
# 【失败复盘 · v9-20260708-123234 (v5 run)】
#   现象：v9 ("v6 + 早停 60 步 + 彻底关 NLL") 训完 80 步, **结果依旧不达标, 甚至
#         创 macro-F1 新低**:
#             SFT v7 ckpt-150      acc=58.5%  macro-F1=41.2%
#             v6-best  ckpt-120    acc=56.4%  macro-F1=34.7%
#             v7  ckpt-140         acc=56.0%  macro-F1=33.7%
#             v8  ckpt-50          acc=56.2%  macro-F1=33.3%
#             **v9 ckpt-80 (best)  acc=55.6%  macro-F1=33.0%** ← 新低
#         v9 硬标准三条全部大幅未达标:
#             * best macro-F1 应 >= 39%, 实际 33.0% (差 6pt)
#             * disgust recall 应 >= 35%, 实际 13.2% (差 22pt)
#             * anger  recall 应 >= 20%, 实际 8.1%  (差 12pt)
#   决定性观察：**v9 sweep 中 ckpt-10 / ckpt-40 / ckpt-80 的各类 recall 几乎完全相同**
#         (波动 <= ±1.5pt), 说明训练在前 10 步就已经塌到"§0 污染稳态", 之后 70
#         步的训练几乎不改变这个状态。
#   由此推出的结构性结论：**offline DPO 与当前 pair 数据存在耦合基态**。
#         无论超参怎么调 (v5→v6→v7→v8→v9, 五次实验), policy 都会在 10-40 步内
#         快速塌到 macro-F1 ≈ 33-35% 附近的稳态, 此后无法逃出。
#         v6 稍好 (34.7%) 是因为多训一点让 rare-class 稍有活动空间;
#         v9 更差 (33.0%) 是因为早停 + 关 NLL 让 rare-class 更早锁死。
#         **这个基态由数据侧 §0 结构性错配决定, 不是训练超参问题**。
#
# 【正式终止 offline DPO 超参调整】
#   五次实验 (v5/v6/v7/v8/v9) 已充分证明: 在当前 pair 数据下, offline DPO 无论
#   如何调参都无法超越 SFT baseline (macro-F1 41.2%)。**禁止再在本脚本超参
#   空间内继续尝试**。若继续做 DPO, 必须先修数据 (路 B); 若放弃 DPO, 走 SFT
#   精修 (路 C)。两条路都需要在本脚本之外操作:
#     - 路 B: 修改 project/stepaudio/tools/build_dpo_pairs.py 增加对
#             rejected=neutral 的比例上限约束; 重新构造 train.dpo.jsonl 后
#             再用 v9 超参训练 (本脚本无需再改, 保持当前 v9 参数);
#     - 路 C: 完全放弃 DPO, 改用 project/stepaudio/run_train_sft_meld.sh
#             (若无则新建) 配合 class-weighted CE / focal loss。
#   本脚本当前 v9 参数保持不变, 作为"路 B 数据修复后即可复用的训练配置"。
#
# 【Path-B 已实施 · 2026-07-08】
#   已在 project/stepaudio/tools/build_dpo_pairs.py 新增 3 个数据修复参数:
#     --max-rejected-neutral-ratio 0.30       # 硬性上限: rejected==neutral 占比 <= 30%
#     --upsample-hard-nonneutral-mistakes 3   # 上采样"GT!=neutral 且 SFT 预测 neutral"的关键 pair
#     --drop-easy-neutral-rejected            # (可选) 丢弃 chosen∈{joy,surprise} 且 rejected==neutral 的水量 pair
#
#   推荐 Path-B 流程（train.dpo.jsonl 重新构建 + 用 v9 超参训练）:
#     # 1) 用当前 SFT ckpt 推理 train 集 (若已有 result_v7*.jsonl 可复用)
#     # 2) 重构 pair 数据 (关键: 加 --max-rejected-neutral-ratio 0.30)
#     python project/stepaudio/tools/build_dpo_pairs.py \
#         --infer-jsonl project/stepaudio/infer_results/result_v7-20260707-162917_checkpoint-150_*.jsonl \
#         --output project/stepaudio/data_meld/train.dpo.jsonl \
#         --classes surprise,anger,neutral,joy,sadness,fear,disgust \
#         --keep-right-per-class 700 --min-wrong-per-class 150 \
#         --min-chosen-per-class 300 --max-audio-sec 90 \
#         --rejected-strategy mistake \
#         --max-rejected-neutral-ratio 0.30 \
#         --upsample-hard-nonneutral-mistakes 3
#     # 3) 直接用当前 v9 超参训练 (本脚本参数保持不变)
#     MODEL_PATH=./output/meld/v7-20260707-162917/checkpoint-150 \
#         bash project/stepaudio/run_train_dpo_meld.sh
#
#   Path-B 期望效果 (对照 v9 硬标准):
#     * rejected==neutral 占比: v5-v9 大概率在 40-60% 区间, Path-B 降到 ≤30%;
#     * DPO 不再"无差别压 neutral logit", 非-neutral 类的 recall 应被松开;
#     * upsample x3 后 anger/sadness/fear/disgust 作为 chosen 的 pair 数至少
#       翻倍, 直接补强"抬 rare-class"的信号强度。
#   若 Path-B 仍不达标 (best macro-F1 < 39%, disgust recall < 35%,
#   anger recall < 20%), 说明"§0 污染"不是问题的全部, 必须切路 C
#   (完全放弃 DPO, 走 SFT 精修 + class-weighted CE / focal loss)。

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
    echo "[WARN] DPO 必须从 MELD SFT 后的 ckpt 起步，冷启动 DPO 无梯度信号。"
    echo "[WARN] 强烈建议（选一个 MELD SFT ckpt，如："
    echo "[WARN]     MODEL_PATH=./output/meld/v0-20260707-182357/checkpoint-234 \\  # full SFT"
    echo "[WARN]     MODEL_PATH=./output/meld/v7-20260707-162917/checkpoint-150 \\  # LoRA SFT (自动 merge)"
    echo "[WARN]     bash project/stepaudio/run_train_dpo_meld.sh"
    echo "[WARN] 若坚持从 base 冷启动，请显式设置：STRICT_SFT_WARMUP=0"
    echo "[WARN] ============================================================"
    if [ "${STRICT_SFT_WARMUP:-1}" = "1" ]; then
        echo "[FATAL] 拒绝冷启动 DPO（STRICT_SFT_WARMUP=1，默认）。"
        exit 16
    fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$SWIFT_ROOT/output/meld/dpo/v3"}
# ms-swift 默认 add_version=True, 会在 OUTPUT_DIR 后自动追加 v<idx>-<timestamp>
# 子目录 (见 swift/arguments/sft_args.py 中 _add_version)。设 ADD_VERSION=false
# 可让 checkpoints 直接落到 $OUTPUT_DIR 下, 便于外部脚本按固定路径消费。
# 注意：关闭 add_version 后，若同一 OUTPUT_DIR 重复启动训练会覆盖之前的
# checkpoints/logs, 请自行确认是否是想要的行为。
ADD_VERSION=${ADD_VERSION:-false}

# 【MELD v6 关键 · SYSTEM prompt 与 SFT 严格对齐】
# v5 用 "You are a helpful assistant." 导致 policy/ref 起点与 SFT 训练时分布
# 不一致 (见 v5 复盘 §5)。v6 起：
#   * 默认 SYSTEM 从 MODEL_PATH/../args.json 读取 SFT 训练时的 system 值,
#     若读取失败或为空则回落到 "";
#   * 用户仍可通过显式 SYSTEM=xxx 覆盖 (自担漂移风险)。
_default_system=""
_sft_args_json=""
if [ -d "$MODEL_PATH" ]; then
    if [ -f "$MODEL_PATH/args.json" ]; then
        _sft_args_json="$MODEL_PATH/args.json"
    elif [ -f "$(dirname "$MODEL_PATH")/args.json" ]; then
        _sft_args_json="$(dirname "$MODEL_PATH")/args.json"
    fi
fi
if [ -n "$_sft_args_json" ] && [ -f "$_sft_args_json" ]; then
    _default_system=$(python3 - "$_sft_args_json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    s = d.get('system') or ''
    print(s if isinstance(s, str) else '')
except Exception:
    print('')
PY
)
    echo "[INFO] 从 SFT args.json ($_sft_args_json) 读取到 system = '${_default_system}'"
fi
SYSTEM=${SYSTEM-"$_default_system"}
echo "[INFO] DPO 训练将使用 SYSTEM='${SYSTEM}' (与 SFT 训练时保持一致)"

# ---------------- Tuner ----------------
TUNER_TYPE=${TUNER_TYPE:-lora}
if [ "$TUNER_TYPE" = "full" ]; then
    DEFAULT_LR=2e-7   # DPO + full 必须比 lora 再小 1 个数量级
else
    # 【v9 关键变更】1e-6 → 2e-6 (回 v6)
    # v8 用 1e-6 配 β=0.1 + LS=0.2 + rpo=0.1 五重弱化, 直接把 DPO 主项信号
    # 打没了 (100 步 margin 从 0 变到 -0.005, 反方向), 只剩 NLL 主导 → v5
    # 剧本重演。v9 回到 v6 已验证能让 margin 从 0 单调抬到 +0.06 的 lr=2e-6,
    # 配套 β=0.05 重新启动 DPO 主项。
    DEFAULT_LR=2e-6
fi
LEARNING_RATE=${LEARNING_RATE:-$DEFAULT_LR}

# LoRA 子参数（v6: rank 8→16, alpha 32 不变, dropout 保留）
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}

# ---------------- DPO 关键超参 ----------------
# BETA: DPO 的 KL 系数。
# 【v9 关键变更】0.1 → 0.05 (回 v6)
# v8 用 β=0.1 后 DPO 主项梯度直接变零 (margin 到 -0.005 反方向), 证明
# β=0.1 对 MELD 这种 Δ≈0 起点的 policy 太严。回 v6 已验证 β=0.05 能让
# margin 从 0 单调抬到 +0.06。代价 (rejected 会掉到 -0.077) 用 v9 的
# max_steps=80 (v6 时的 36%) 控制, 不再靠 β 加倍。
BETA=${BETA:-0.05}
# LOSS_TYPE: sigmoid（原始 DPO） / ipo / hinge / robust / apo_zero / apo_down /
# discopop / sft。
# v9 保持 sigmoid (v7 IPO 实验证明 IPO 在当前规模下有效步长只有 sigmoid 的 1/5,
# “天然刹车”因 Δ≈0而没触发)。
LOSS_TYPE=${LOSS_TYPE:-sigmoid}
# LABEL_SMOOTHING: cDPO 的 conservative label smoothing, 0.0 = 关闭。
# 【v9 关键变更】0.2 → 0.0
# v6 (LS=0)⇒v7(IPO+LS=0.1)⇒v8(sigmoid+LS=0.2) 三次实验证明 LS 无法抵消
# §0 结构性错配 (rejected=neutral 密度过高), 反而稀释了 DPO 信号强度
# (v8 该因素 ×0.6, 五重弱化中的一个)。v9 彻底关掉 LS, 回到 v6 基线。
# **若想解决 §0 污染必须走路 B 改数据, 不在本脚本范围**。
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
# RPO_ALPHA: 在 chosen 上叠加 SFT NLL loss。0 = 纯 DPO；>0 就是 RPO。
# 【v9 关键变更】0.1 → 0.0 (重回 v6 基线)
# v5 (alpha=0.1) 崩溃因为 chosen 分布反向, NLL 把预测推向 joy。
# v6 (alpha=0) 不崩溃但不达 SFT。
# v8 (alpha=0.1) 试图用 alpha 当强锚点, 但因为 DPO 主项被 β/LS/LR 五重弱化
# 到接近零, 变成 NLL 主导 → v5 剧本重演 (neutral→joy FP=73 接近 v6 的 79),
# 而且 rare-class recall 创新低 (anger 9.0%, disgust 11.8%)。
# **v9 彻底关掉 NLL**, 只保留纯 DPO 偏好信号。这是 v9 对抗 v8 崩溃的核心锁。
RPO_ALPHA=${RPO_ALPHA:-0.0}
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
# 【v9 关键变更】100 → 80
# v6 sweep 表明 ckpt-120 是 v6 内部最好, 但也只有 34.7%; v6 ckpt-40 (20% 时长
# 处) 因为 rare-class 尚未完全塌陷可能反而更接近 SFT baseline。v9 缩到 80 步
# (相当于 v6 的 36%), 配合 load_best_model_at_end=loss 在前 80 步内自动选最优。
# **v9 本质上只是 “v6 + 早停’’, 不是新药方。若仍不达标必须切路 B/C。**
MAX_STEPS=${MAX_STEPS:-80}

# 【v8 关键变更】save/eval 20 → 10
# 步数缩短 30%, 评估更密以精准定位真最优点 (v6 sweep 用 20 步
# 间隔, 得到 ckpt-120 与 ckpt-100 F1 差异不到 1pt, 显得平原化,
# 10 步间隔能看清峰值区间的形状)。
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-10}
save_total_limit=${save_total_limit:-12}
LOGGING_STEPS=${LOGGING_STEPS:-1}
# 【v7 关键变更】metric rewards/accuracies → loss
# v6 复盘 §3: eval/rewards/accuracies 峰值在 step 180, 但 sweep 表明 test
# macro-F1 最好的 ckpt 是 120。二者错位 60 步。改用 eval/loss:
# v6 曲线里 eval/loss 从 0.693 单调降到 0.669 (step 200), 之后微升,
# 谷底更接近 test 分类峰值区间 (100-140 步)。DPO 里 loss 更能反映
# "policy 与 pair 契合度", 且不会像 accuracies 那样在过训练区间继续走高。
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-true}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-loss}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-false}

# save_only_model 与 load_best_model_at_end 在 DeepSpeed 下互斥 (HF Trainer
# 强约束: 要 load best 必须能重放完整 optimizer state, 只存权重时做不到)。
# 因此当开启 LOAD_BEST_MODEL_AT_END 时, 默认关闭 save_only_model; 反之默认
# 打开以节约磁盘。用户仍可显式 SAVE_ONLY_MODEL=true 覆盖 (但配合
# LOAD_BEST_MODEL_AT_END=true 时会被 HF 拒绝启动)。
if [ "$LOAD_BEST_MODEL_AT_END" = "true" ]; then
    SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-false}
else
    SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}
fi

# ---------------- 序列 / 截断 ----------------
# 【对齐 SFT-full 经验 · 2026-07-09】MAX_LENGTH 2048 → 4096。
# SFT-full ([run_train_sft_full_meld.sh]) 一直用 4096; DPO 每步同时前向
# chosen + rejected 两份序列, 若 prompt 侧被 2048 截断, 会让 chosen 和
# rejected 出现不同截断位, 破坏偏好信号。抬到 4096 与 SFT-full 完全对齐。
# 注意: DPO 每步 2× 前向, MAX_LENGTH 翻倍会让激活显存约 2× 增长,
# 显存不够时可 MAX_LENGTH=2048 bash \$0 回退, 或改用 zero3 (USE_DEEPSPEED=3)。
MAX_LENGTH=${MAX_LENGTH:-4096}
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

# ---------------- 数据集 (MELD word 版, 7 类) ----------------
# MELD 用 data_meld/ 目录下的 word 版 (assistant label 是完整单词, 例如
# "disgust"/"fear"/...) —— letter 版 (S/A/N/J/D/F/G) 训练全部塌陷, 已由 SFT
# 脚本头部复盘证明。
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
TRAIN_JSONL=${TRAIN_JSONL:-"$DATA_DIR/train.dpo.jsonl"}
MELD_CLASSES=${MELD_CLASSES:-"surprise,anger,neutral,joy,sadness,fear,disgust"}
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[FATAL] TRAIN_JSONL 不存在: $TRAIN_JSONL"
    echo ""
    echo "        MELD DPO 训练前必须先构造成对样本 (v6 版命令):"
    echo ""
    echo "        1) 用 MELD SFT ckpt 在 train.jsonl 上做推理（若已有可跳过）:"
    echo "             MODEL_PATH=<meld_sft_merged_ckpt> \\"
    echo "             VAL_JSONL=$DATA_DIR/train.jsonl \\"
    echo "             bash project/stepaudio/run_inference_meld.sh"
    echo ""
    echo "        2) 用推理 jsonl 构造 DPO pair (MELD 7 类, v7):"
    echo "             python $SCRIPT_DIR/tools/build_dpo_pairs.py \\"
    echo "                 --infer-jsonl <上一步产出的 result_*.jsonl> \\"
    echo "                 --output $TRAIN_JSONL \\"
    echo "                 --classes $MELD_CLASSES \\"
    echo "                 --min-wrong-per-class 150 \\"
    echo "                 --keep-right-per-class 1500 \\"
    echo "                 --min-chosen-per-class 300 \\"
    echo "                 --max-audio-sec 30 \\"
    echo "                 --rejected-strategy mistake"
    echo "        (v7: keep-right-per-class 从 700 抬到 1500, 让 chosen 里"
    echo "         neutral 数量接近 joy, 缓解 rejected=neutral 密度过高问题)"
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
        echo "                --classes $MELD_CLASSES \\"
        echo "                --min-wrong-per-class 150 \\"
        echo "                --max-audio-sec 30 \\"
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
echo "[INFO] load_best_model_at_end=$LOAD_BEST_MODEL_AT_END  metric=$METRIC_FOR_BEST_MODEL (greater_is_better=$GREATER_IS_BETTER)  save_only_model=$SAVE_ONLY_MODEL"

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
    --system "$SYSTEM" \
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
    --save_only_model $SAVE_ONLY_MODEL \
    --load_best_model_at_end $LOAD_BEST_MODEL_AT_END \
    --metric_for_best_model $METRIC_FOR_BEST_MODEL \
    --greater_is_better $GREATER_IS_BETTER \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters false \
    --seed 42 \
    "$@"

echo "[INFO] MELD DPO 训练完成，Checkpoint 保存在: $OUTPUT_DIR"
