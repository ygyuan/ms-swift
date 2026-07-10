#!/usr/bin/env bash
# StepAudio2-mini 推理脚本（MELD 语音情感 7 分类）
# 任务: 输入一段音频，模型直接输出情感类别字符串，取值范围:
#       [surprise, anger, neutral, joy, sadness, fear, disgust]
# 因此本脚本默认:
#   - 关闭采样 (temperature=0, top_p=1)，使输出可复现，方便后续做 threshold 评估
#   - max_new_tokens 设小 (默认 2048)，分类只需要少量 token (最长 'surprise'/'sadness'/'disgust' 也在 3 token 内)
#   - 打开 --logprobs / --top_logprobs，让结果里带上 token 级别的概率分布，
#     供 eval_classification_meld.py 做 threshold / precision / recall / F1 分析
#
# [新增]  权重加载校验 & 输出健康检查:
#   - 推理前会把 MODEL_PATH 里的 config / safetensors 分片摘要（大小、head sha1）
#     落到 ${RESULT_PATH}.model_signature.json，方便日后核对"这次推理到底加载了
#     哪个模型目录"（防止 swift infer 静默 fallback 到基座权重导致假高分）。
#   - 推理后自动扫描前 HEALTH_SAMPLE_LIMIT 条结果（默认 500），检查 response 类别
#     分布 / first-token 平均置信度 / 候选类是否命中 top-20，触发规则时告警，
#     报告存到 ${RESULT_PATH}.health.json。可通过 HEALTH_CLASSES 覆盖候选类列表。
#
# 环境变量（均可覆盖默认值）:
#   MODEL_PATH    [必填] 要推理的模型/checkpoint 路径
#                 - 全参微调: 直接指向 checkpoint-xxx 目录
#                 - LoRA 微调: 指向含 adapter_config.json 的目录, 同时通过 BASE_MODEL 指定基座
#                 - 评估原始模型: 直接指向 HF 模型目录, 例如
#                   /apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini
#   BASE_MODEL    LoRA 基座模型 (仅 LoRA 模式需要, 默认 Step-Audio-2-mini)
#   VAL_JSONL     待推理的 JSONL (默认 project/stepaudio/data_meld/test.jsonl, MELD 官方 test 集)
#   RESULT_PATH   结果保存路径   (默认 project/stepaudio/infer_results/result_<ckpt>_<ts>.jsonl)
#   MAX_NEW_TOKENS / TEMPERATURE / TOP_P / TOP_LOGPROBS / LOGPROBS  推理超参
#   MAX_SAMPLES   只推理前 N 条 (调试用，0 表示全量)
#   NPROC_PER_NODE       数据并行进程数 (默认 1; >1 时会启用 torch.distributed.run 在多卡上
#                        DDP 切分 val_dataset, 每卡加载一份模型副本, 结果合并到 RESULT_PATH)
#   CUDA_VISIBLE_DEVICES 可用 GPU 列表 (默认根据 NPROC_PER_NODE 自动选为 0,1,...,N-1)
#   EVAL_BATCH_SIZE      每个进程的推理 batch 大小 (默认 1; 增大可提升吞吐, 但需注意:
#                        1) 显存占用线性增长; 2) batch>1 时各样本会做 padding,
#                        请先小批量与 batch=1 对比 logprobs 是否一致再放大)
#   USE_CACHE            是否启用 HF generate 的 KV cache. 默认 false (对齐 UltraEval-Audio
#                        的 use_cache=False, 是拿到 55.47 baseline 的关键之一).
#                        USE_CACHE=false 时脚本会把 helpers/pyshim 前置到 PYTHONPATH,
#                        通过 sitecustomize.py monkey-patch swift 强制关闭 KV cache;
#                        USE_CACHE=true 则保持 swift/HF 默认 (True), 用于回滚对比.
#
# 用法示例:
#   MODEL_PATH=/path/to/output/v0-xxx/checkpoint-1000 bash run_inference_meld.sh
#   MODEL_PATH=/path/to/lora_ckpt BASE_MODEL=/path/to/base bash run_inference_meld.sh
#   MODEL_PATH=/path/to/checkpoint MAX_SAMPLES=100 bash run_inference_meld.sh
#   # 也支持 KEY=VALUE 直接通过命令行覆盖, 例如:
#   bash run_inference_meld.sh MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=2
#   # 显式指定 RESULT_PATH 以便外层脚本拿到固定路径再喂给 run_eval_meld.sh:
#   bash run_inference_meld.sh MODEL_PATH=/path/to/ckpt RESULT_PATH=/tmp/result.jsonl
#   # 4 卡 DDP 并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=4 bash run_inference_meld.sh
#   # 指定卡 1,2,3 三张卡并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=1,2,3 bash run_inference_meld.sh

set -e

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 允许 `bash run_inference_meld.sh KEY=VALUE ...` 这种方式覆盖环境变量,
# 同时把这些 KEY=VALUE 从 $@ 中剔除, 避免被当成 swift infer 的额外参数透传。
PASS_THROUGH_ARGS=()
for kv in "$@"; do
    case "$kv" in
        *=*)
            export "$kv"
            ;;
        *)
            PASS_THROUGH_ARGS+=("$kv")
            ;;
    esac
done
set -- "${PASS_THROUGH_ARGS[@]}"

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data_meld/test.jsonl"}

# 分类任务默认参数（覆盖原来面向 chat/tts 的默认值）
# 注意: swift infer 没有 --do_sample 参数；temperature=0 即等价于贪心解码（do_sample=False）。
#
# 【v3 关键修复 · 对齐 UltraEval-Audio 55.47% baseline】
# ─────────────────────────────────────────────────────────
# 之前用同一份 wav + 同一份 prompt (从 UltraEval 严格重建的 test.jsonl) 跑 swift infer 还是
# 塌陷成 99.8% neutral (acc=48%), 而 UltraEval 官方能拿 55.47%. 排查 UltraEval-Audio 的
# audio_evals/models/step_audio_2_mini.py 发现 2 个决定性差异:
#
#   (A) 【必须】UltraEval 显式注入 system prompt "You are a helpful assistant.":
#         if not has_system:
#             messages.append({"role":"system","content":"You are a helpful assistant."})
#       完整 prompt 是: <|BOT|>system\nYou are a helpful assistant.<|EOT|><|BOT|>human\n...
#       Step-Audio-2-mini 训练时肯定用了 system prompt, 缺失时输出分布严重偏移.
#       swift 默认 --system 是 None, 我们必须显式传. 这是 55.47 vs 48.08 差距的核心原因.
#
#   (B) UltraEval 的 max_new_tokens=2048, 让模型自由生成完整词 (可能是 "neutral" 也可能
#       是 "the emotion is neutral", eval 时靠字面匹配前 4 字符). 我们之前 MAX_NEW_TOKENS=4/32
#       会把模型输出截成半个词, 虽然对 first-token argmax 无影响 (health 里 mean_p=0.54 说明
#       模型输出稳定), 但 response 字面无法做 threshold sweep. 【v4 对齐 UltraEval】直接放到
#       2048, 让模型自由生成, response 字面就能直接跟 UltraEval 的 55.47 baseline 对拍.
#       (贪心解码 temperature=0, 遇到 <|EOT|> 就停, 实测 200 样本平均 ~5 token, 长度上限只是保险.)
#
#   (C) 【必须】UltraEval 显式用 use_cache=False 调 model.generate. Step-Audio-2-mini 的
#       audio_encoder + adaptor 侧 KV cache 在 batch/序列拼接时会有 bug (输出会漂移),
#       UltraEval 官方就是靠关掉 KV cache 才拿到 55.47. swift CLI 没有 --use_cache 参数,
#       InferArguments 也无 generation_config 字段, 因此我们通过 sitecustomize.py 在
#       解释器启动时 monkey-patch swift.infer_engine.utils.prepare_generation_config,
#       强制 use_cache=False. 默认开启 (USE_CACHE=false); 需要还原 KV cache 时设
#       USE_CACHE=true.
SYSTEM=${SYSTEM:-"You are a helpful assistant."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
LOGPROBS=${LOGPROBS:-true}
TOP_LOGPROBS=${TOP_LOGPROBS:-20}
# 是否启用 HF generate 的 KV cache. 默认关 (对齐 UltraEval-Audio 的 use_cache=False).
# 只接受 true / false 两值. false 时通过 sitecustomize shim monkey-patch swift.
USE_CACHE=${USE_CACHE:-false}
USE_CACHE_LC=$(echo "$USE_CACHE" | tr '[:upper:]' '[:lower:]')
if [ "$USE_CACHE_LC" != "true" ] && [ "$USE_CACHE_LC" != "false" ]; then
    echo "[ERROR] USE_CACHE 只接受 true/false, 当前=$USE_CACHE" >&2
    exit 1
fi
# 处理超长样本：
#   - MAX_LENGTH: 输入 token 上限。**必须 ≤ 模型自身的 max_model_len（step_audio_2_mini=16384）**，
#       并给 max_new_tokens 留出余量，否则推理引擎会算出 max_tokens<=0 直接崩。
#       另外 step_audio_2_mini 只支持 attn_impl=eager，attention 显存 O(L^2)：
#           L=4096  → 单次 softmax ~0.5 GiB（28 heads），单卡 80G 稳过
#           L=8192  → ~7.5 GiB，勉强
#           L=15360 → ~26 GiB，极易 OOM（尤其显卡被别的进程占用时）
#       默认 4096 SFT/GRPO 训练时的 MAX_LENGTH 完全一致，保证输入分布匹配。
#       如需覆盖长音频推理，可显式抬高（同步降 EVAL_BATCH_SIZE、确保 GPU 独占）。
#   - TRUNCATION_STRATEGY: swift template 支持 raise/left/right/split；
#       'left'  -> 直接从左边截断（音频开头砍掉），最稳妥，不会抛错。
#       'right' -> 从右边截断（可能砍掉答案），慎用。
#       'raise' -> 抛异常终止推理（swift 里 delete=raise，不适合推理）。
MAX_LENGTH=${MAX_LENGTH:-4096}
TRUNCATION_STRATEGY=${TRUNCATION_STRATEGY:-left}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
if ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$EVAL_BATCH_SIZE" -lt 1 ]; then
    echo "[ERROR] EVAL_BATCH_SIZE 必须是 >=1 的整数, 当前=$EVAL_BATCH_SIZE" >&2
    exit 1
fi

# 多卡并行推理: NPROC_PER_NODE>1 时 swift CLI 会自动切换到 torch.distributed.run，
# 在每张卡上起一个进程，并在 swift/pipelines/infer/infer.py 里按 rank 对 val_dataset 调用
# dataset.shard(world_size, rank) 切片，最后合并写到 RESULT_PATH。
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [ "$NPROC_PER_NODE" -lt 1 ]; then
    echo "[ERROR] NPROC_PER_NODE 必须是 >=1 的整数, 当前=$NPROC_PER_NODE" >&2
    exit 1
fi

# 必须显式传入 MODEL_PATH (不再自动定位最新 checkpoint)
if [ -z "${MODEL_PATH:-}" ]; then
    echo "[ERROR] 请通过环境变量 MODEL_PATH 指定要推理的模型/checkpoint 目录。" >&2
    echo "        示例: MODEL_PATH=/path/to/output/v0-xxx/checkpoint-1000 bash $0" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH 不存在或不是目录: $MODEL_PATH" >&2
    exit 1
fi

# 区分 LoRA / full 权重：
#   - LoRA 产物目录特征是存在 adapter_config.json，此时需要同时给 swift infer
#     传 --model <base> 和 --adapters <adapter_dir>；否则仅 --model <full_ckpt> 即可。
BASE_MODEL=${BASE_MODEL:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
ADAPTERS=""
if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    ADAPTERS="$MODEL_PATH"
    MODEL_FOR_INFER="$BASE_MODEL"
else
    MODEL_FOR_INFER="$MODEL_PATH"
fi

# RESULT_PATH 默认带上 checkpoint 名与时间戳，避免不同实验互相覆盖
if [ -z "${RESULT_PATH+x}" ]; then
    CKPT_TAG=$(basename "$MODEL_PATH")
    PARENT_TAG=$(basename "$(dirname "$MODEL_PATH")")
    TS=$(date +%Y%m%d_%H%M%S)
    RESULT_PATH="$SCRIPT_DIR/infer_results/result_${PARENT_TAG}_${CKPT_TAG}_${TS}.jsonl"
fi

mkdir -p "$(dirname "$RESULT_PATH")"

# CUDA_VISIBLE_DEVICES: 未设置时默认根据 NPROC_PER_NODE 自动生成 0,1,...,N-1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
    else
        CUDA_VISIBLE_DEVICES=0
    fi
fi
export CUDA_VISIBLE_DEVICES

# 交叉校验: CUDA_VISIBLE_DEVICES 中的卡数要 >= NPROC_PER_NODE
NUM_VISIBLE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$NUM_VISIBLE" -lt "$NPROC_PER_NODE" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES (可见卡数=$NUM_VISIBLE: $CUDA_VISIBLE_DEVICES) 少于 NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 1
fi

# swift CLI 靠 NPROC_PER_NODE 环境变量判断是否走 torch.distributed.run
export NPROC_PER_NODE

# torch.distributed.run 默认使用 MASTER_PORT=29500 (由 swift/utils/torch_utils.py 兜底);
# 同一台机器如果并发跑多个 DDP 任务, 或历史进程未清理, 会撞到 EADDRINUSE.
# 策略:
#   - 用户/上层脚本 (如 run_sweep_eval_meld.sh) 显式给了 MASTER_PORT, 就尊重之;
#   - 否则 NPROC_PER_NODE>1 时用 python 挑一个操作系统当前空闲的端口;
#   - NPROC_PER_NODE=1 时不走 torch.distributed.run, 无需设置.
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    if [ -z "${MASTER_PORT:-}" ]; then
        # 用系统当前的 python 找一个空闲端口 (bind :0 让内核分配, 立刻 close 释放)
        # 注意此处只是"探测", 存在极小概率的 TOCTOU 竞态; 极端情况下可再次失败,
        # 用户可显式传 MASTER_PORT=xxxxx 兜底.
        _PY_FOR_PORT="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
        if [ -z "$_PY_FOR_PORT" ] || [ ! -x "$_PY_FOR_PORT" ]; then
            _PY_FOR_PORT=$(command -v python3 || command -v python)
        fi
        if [ -n "$_PY_FOR_PORT" ]; then
            MASTER_PORT=$("$_PY_FOR_PORT" -c 'import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || true)
        fi
        # 兜底: python 不可用或调用失败时, 在 [30000, 39999] 里随机取一个 (概率碰撞极低)
        if [ -z "$MASTER_PORT" ]; then
            MASTER_PORT=$(( ( RANDOM % 10000 ) + 30000 ))
        fi
    fi
    export MASTER_PORT
    echo "[INFO] MASTER_PORT          = $MASTER_PORT  [DDP 用; 未显式指定则自动挑空闲端口, 避免 29500 被占]"
fi

# 如果设置了 MAX_SAMPLES > 0，则派生一个抽样后的小 val 集
EFFECTIVE_VAL_JSONL="$VAL_JSONL"
if [ "$MAX_SAMPLES" -gt 0 ] && [ -f "$VAL_JSONL" ]; then
    SUB_JSONL="$SCRIPT_DIR/infer_results/_subset_${MAX_SAMPLES}.jsonl"
    head -n "$MAX_SAMPLES" "$VAL_JSONL" > "$SUB_JSONL"
    EFFECTIVE_VAL_JSONL="$SUB_JSONL"
    echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES，使用子集: $SUB_JSONL"
fi

echo "[INFO] MODEL_PATH    = $MODEL_PATH"
if [ -n "$ADAPTERS" ]; then
    echo "[INFO] BASE_MODEL    = $BASE_MODEL  [LoRA adapter 模式]"
    echo "[INFO] ADAPTERS      = $ADAPTERS"
else
    echo "[INFO] BASE_MODEL    = [全参检查点, 不需要 --adapters]"
fi
echo "[INFO] VAL_JSONL     = $EFFECTIVE_VAL_JSONL"
echo "[INFO] RESULT_PATH   = $RESULT_PATH"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] NPROC_PER_NODE       = $NPROC_PER_NODE  [>=2 则启用 DDP 数据并行]"
echo "[INFO] EVAL_BATCH_SIZE      = $EVAL_BATCH_SIZE  [每个进程的推理 batch 大小]"
echo "[INFO] temperature=$TEMPERATURE  [0 表示贪心], top_p=$TOP_P, max_new_tokens=$MAX_NEW_TOKENS"
echo "[INFO] logprobs=$LOGPROBS, top_logprobs=$TOP_LOGPROBS"
echo "[INFO] max_length=$MAX_LENGTH, truncation_strategy=$TRUNCATION_STRATEGY"
echo "[INFO] use_cache=$USE_CACHE_LC  [false 时通过 sitecustomize shim 强制关 KV cache, 对齐 UltraEval-Audio]"

# ---- 通过 sitecustomize.py 注入 use_cache=False (仅当 USE_CACHE=false) ----
# 详见 helpers/pyshim/sitecustomize.py 顶部注释. 这里做两件事:
#   1) 把 helpers/pyshim 前置到 PYTHONPATH, 让 python 启动时自动 import sitecustomize
#      (CPython 官方机制, 对 torch.distributed.run 起的每个子进程同样生效, DDP 下也 OK).
#   2) 用 STEPAUDIO_DISABLE_KV_CACHE=1 作为触发开关, 避免这个 sitecustomize 在其他
#      无关 python 进程里意外接管 swift 的 generation_config.
PYSHIM_DIR="$SCRIPT_DIR/helpers/pyshim"
if [ "$USE_CACHE_LC" = "false" ]; then
    if [ -f "$PYSHIM_DIR/sitecustomize.py" ]; then
        export PYTHONPATH="$PYSHIM_DIR${PYTHONPATH:+:$PYTHONPATH}"
        export STEPAUDIO_DISABLE_KV_CACHE=1
        echo "[INFO] 已启用 sitecustomize shim 强制 use_cache=False (PYTHONPATH 前置: $PYSHIM_DIR)"
    else
        echo "[WARN] USE_CACHE=false 但未找到 $PYSHIM_DIR/sitecustomize.py, 无法关闭 KV cache" >&2
    fi
fi

# 选择 swift 入口
# 注意: 子 shell 中 python/swift 可能因为 rc 钩子被切到别的 conda 环境
# (例如机器上 /root/custom.bashrc 会把 python 改回 env-3.6.8)，
# 因此这里优先用「当前已激活环境」对应的解释器，避免 PATH 漂移。
#   1) 显式 CONDA_SWIFT_BIN
#   2) $CONDA_PREFIX/bin/swift (即激活的 conda env 中的 swift)
#   3) PATH 中的 swift
#   4) $CONDA_PREFIX/bin/python -m swift.cli.main
#   5) PATH 中的 python -m swift.cli.main (最后兜底)
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-}"
if [ -z "$CONDA_SWIFT_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/swift" ]; then
    CONDA_SWIFT_BIN="$CONDA_PREFIX/bin/swift"
fi

if [ -n "$CONDA_SWIFT_BIN" ] && [ -x "$CONDA_SWIFT_BIN" ]; then
    echo "[INFO] 使用 swift CLI: $CONDA_SWIFT_BIN"
    export PATH="$(dirname "$CONDA_SWIFT_BIN"):$PATH"
    SWIFT_CMD=("$CONDA_SWIFT_BIN")
elif command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    echo "[INFO] 未找到 swift CLI，回退到: $CONDA_PREFIX/bin/python -m swift.cli.main"
    SWIFT_CMD=("$CONDA_PREFIX/bin/python" -m swift.cli.main)
else
    echo "[WARN] 未找到 swift CLI 且未检测到 CONDA_PREFIX，回退到 PATH 中的 python -m swift.cli.main"
    echo "       若报 ModuleNotFoundError，请先 conda activate 正确的环境，或显式设置 CONDA_SWIFT_BIN。"
    SWIFT_CMD=(python -m swift.cli.main)
fi

INFER_ARGS=(
    --model "$MODEL_FOR_INFER"
    --model_type step_audio2_mini
    --val_dataset "$EFFECTIVE_VAL_JSONL"
    --result_path "$RESULT_PATH"
    --attn_impl eager
    --torch_dtype bfloat16
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --logprobs "$LOGPROBS"
    --top_logprobs "$TOP_LOGPROBS"
    --max_batch_size "$EVAL_BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --truncation_strategy "$TRUNCATION_STRATEGY"
    --stream false
)
# 【v3】对齐 UltraEval-Audio 的 system prompt 注入 (Step-Audio-2 训练模板需要 system).
# 若空字符串则跳过 --system, 保留传空的能力.
if [ -n "$SYSTEM" ]; then
    INFER_ARGS+=(--system "$SYSTEM")
fi
if [ -n "$ADAPTERS" ]; then
    INFER_ARGS+=(--adapters "$ADAPTERS")
fi

# ---- 权重签名 (推理前) ----
# 目的：把要加载的 checkpoint 的关键指纹（safetensors 分片名、大小、
# 声明的 total_size、config 里的 architectures / model_type / hidden_size 等）
# 落到 ${RESULT_PATH}.model_signature.json，以后即使推理输出可疑，也能通过
# 该文件确认"当时到底指向了哪个模型目录 / 那个目录长什么样"。
# 之前排查 v17 的一次误加载事故时就是靠 predict.jsonl 反推才发现 swift infer
# 拿到的其实是别的模型；有了签名文件后可以直接一键比对。
SIG_MODEL_PATH="$MODEL_PATH"
if [ -n "$ADAPTERS" ]; then
    # LoRA 模式下也顺带记录基座路径
    SIG_ADAPTERS="$ADAPTERS"
else
    SIG_ADAPTERS=""
fi
SIG_JSON="${RESULT_PATH}.model_signature.json"
mkdir -p "$(dirname "$SIG_JSON")"
# 说明: 这段签名逻辑之前是 `python3 - <<'PYEOF' ... PYEOF` 内联 heredoc.
# 在 NPROC_PER_NODE>1 (DDP) 场景下, `swift infer` 会启动 torch.distributed.run
# 及若干子进程, 它们与父 bash 共享 stdin. 观察到 heredoc 的 stdin 会被吞掉,
# 导致 heredoc 后半段 python 代码"泄漏"回 bash 主进程被当作命令解析,
# 报出诡异的 "run_inference.sh: line 386: 日志中比对: command not found",
# 并让 bash 以 rc=127 退出、上层 sweep 误判为"推理失败".
# 改为调用外部 .py 文件, 从根上避免这种 stdin 竞态.
# 另外用 `set +e` 局部包裹, 保证签名工具本身如果失败也不会中断主流程.
SIG_HELPER="$SCRIPT_DIR/helpers/write_model_signature.py"
if [ -f "$SIG_HELPER" ]; then
    set +e
    SIG_MODEL_PATH="$SIG_MODEL_PATH" \
    SIG_BASE_MODEL="$MODEL_FOR_INFER" \
    SIG_ADAPTERS="$SIG_ADAPTERS" \
    SIG_JSON="$SIG_JSON" \
    python3 "$SIG_HELPER"
    _sig_rc=$?
    set -e
    if [ "$_sig_rc" -ne 0 ]; then
        echo "[WARN] 计算权重签名失败 (rc=$_sig_rc), 继续推理"
    fi
    unset _sig_rc
else
    echo "[WARN] 未找到签名脚本: $SIG_HELPER, 跳过权重签名"
fi

"${SWIFT_CMD[@]}" infer "${INFER_ARGS[@]}" "$@"

echo "[INFO] 推理完成，结果保存在: $RESULT_PATH"

if [ -f "$RESULT_PATH" ]; then
    num_results=$(wc -l < "$RESULT_PATH")
    echo "[INFO] 共生成 $num_results 条结果"

    # ---- 推理后健康检查 ----
    # 目的：尽早发现"模型未按预期加载 / 已塌陷"这类静默故障。
    # 之前 v17 有一次跑出 97.7% 的假高分，实际是 swift infer 静默 fallback 到别的
    # 权重上；也见过训练崩坏的 checkpoint 只输出单一类别。这两种病理情况都会
    # 在 response 分布 / first-token 分布上留下明显痕迹，本步骤自动检测并告警。
    HEALTH_JSON="${RESULT_PATH}.health.json"
    # 与签名脚本同理: 之前是 heredoc, 在 DDP 场景下会被 stdin 竞态摧毁
    # (报出 "line 386: 日志中比对: command not found", rc=127).
    # 抽到独立 .py 文件 + set +e 局部包裹, 保证健康检查失败绝不会污染主流程.
    HEALTH_HELPER="$SCRIPT_DIR/helpers/inference_health_check.py"
    if [ -f "$HEALTH_HELPER" ]; then
        set +e
        HEALTH_INPUT="$RESULT_PATH" \
        HEALTH_OUTPUT="$HEALTH_JSON" \
        HEALTH_CLASSES="${HEALTH_CLASSES:-s,a,n,j,d,f,g,surprise,anger,neutral,joy,sadness,fear,disgust}" \
        HEALTH_SAMPLE_LIMIT="${HEALTH_SAMPLE_LIMIT:-500}" \
        python3 "$HEALTH_HELPER"
        _health_rc=$?
        set -e
        if [ "$_health_rc" -ne 0 ]; then
            echo "[WARN] 健康检查脚本失败 (rc=$_health_rc), 不影响推理结果本身"
        fi
        unset _health_rc
    else
        echo "[WARN] 未找到健康检查脚本: $HEALTH_HELPER, 跳过健康检查"
    fi

    echo "[INFO] 下一步可直接运行评估:"
    echo "       bash $SCRIPT_DIR/run_eval_meld.sh \\"
    echo "            RESULT_PATH=$RESULT_PATH \\"
    echo "            VAL_JSONL=$VAL_JSONL"
fi
