#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B 推理脚本（音频场景 5 分类）
# 任务: 输入一段音频，模型直接输出类别字符串，取值范围:
#       [speech, music, noise, porn, song]
# 因此本脚本默认:
#   - 关闭采样 (temperature=0, top_p=1)，使输出可复现，方便后续做 threshold 评估
#   - max_new_tokens 设小 (默认 8)，分类只需要少量 token
#   - 打开 --logprobs / --top_logprobs，让结果里带上 token 级别的概率分布，
#     供 eval_classification.py 做 threshold / precision / recall / F1 分析
#
# Qwen3-Omni 专属注意:
#   1. 30B-A3B 是 MoE thinker + talker 的双模型结构。做"音频->文本分类"推理时
#      只需要 thinker，必须 ENABLE_AUDIO_OUTPUT=0 关闭 talker，否则会额外加载
#      ~7GB talker 权重并占大量显存。
#   2. attn_impl 必须用 flash_attn（不是 eager），否则 30B 模型几乎不可推。
#   3. LoRA 微调产物（adapter_config.json 存在）需通过 --adapters 单独传入，
#      --model 指向基座 Qwen3-Omni-30B-A3B-Instruct。
#   4. 多模态 token 数量由环境变量控制（默认值与训练一致）：
#      IMAGE_MAX_TOKEN_NUM / VIDEO_MAX_TOKEN_NUM / FPS_MAX_FRAMES。
#
# 环境变量（均可覆盖默认值）:
#   MODEL_PATH    [必填] 要推理的模型/checkpoint 路径
#                 - 全参微调: 直接指向 checkpoint-xxx 目录
#                 - LoRA 微调: 指向含 adapter_config.json 的目录, 同时通过 BASE_MODEL 指定基座
#                 - 评估原始模型: 直接指向 HF 模型目录, 例如
#                   /apdcephfs_qy3/share_301069248/huggingface/Qwen/Qwen3-Omni-30B-A3B-Instruct
#   BASE_MODEL    LoRA 基座模型 (仅 LoRA 模式需要, 默认 Qwen3-Omni-30B-A3B-Instruct)
#   VAL_JSONL     待推理的 JSONL (默认 project/qwen3-omni/data/val.jsonl)
#   RESULT_PATH   结果保存路径   (默认 project/qwen3-omni/infer_results/result_<ckpt>_<ts>.jsonl)
#   MAX_NEW_TOKENS / TEMPERATURE / TOP_P / TOP_LOGPROBS / LOGPROBS  推理超参
#   MAX_SAMPLES   只推理前 N 条 (调试用，0 表示全量)
#   NPROC_PER_NODE       数据并行进程数 (默认 1; >1 时会启用 torch.distributed.run 在多卡上
#                        DDP 切分 val_dataset, 每卡加载一份模型副本, 结果合并到 RESULT_PATH)
#   CUDA_VISIBLE_DEVICES 可用 GPU 列表 (默认根据 NPROC_PER_NODE 自动选为 0,1,...,N-1)
#   EVAL_BATCH_SIZE      每个进程的推理 batch 大小 (默认 1; 增大可提升吞吐, 但需注意:
#                        1) 显存占用线性增长; 2) batch>1 时各样本会做 padding,
#                        请先小批量与 batch=1 对比 logprobs 是否一致再放大)
#   ATTN_IMPL            注意力实现 (默认 flash_attn; 30B-MoE 不要用 eager)
#   ENABLE_AUDIO_OUTPUT  是否加载 talker (默认 0=关闭, 分类任务不需要语音输出)
#   IMAGE_MAX_TOKEN_NUM / VIDEO_MAX_TOKEN_NUM / FPS_MAX_FRAMES  多模态 token 上限
#   USE_SHM_CACHE        =1 时把基座模型预拷到 /dev/shm 加速多卡加载
#                        (CephFS/NFS 上 4 卡 DDP 加载从 ~30 分钟降到 <2 分钟,
#                         tmpfs 占 RAM, 重启自动清空; 默认 0=关闭)
#   SHM_CACHE_DIR        缓存根目录 (默认 /dev/shm/swift_model_cache)
#
# 用法示例:
#   MODEL_PATH=/path/to/output/qwen3_omni/v1_lora/checkpoint-1000 bash run_inference.sh
#   MODEL_PATH=/path/to/lora_ckpt BASE_MODEL=/path/to/base bash run_inference.sh
#   MODEL_PATH=/path/to/checkpoint MAX_SAMPLES=100 bash run_inference.sh
#   # 4 卡 DDP 并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=4 bash run_inference.sh
#   # 指定卡 1,2,3 三张卡并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=1,2,3 bash run_inference.sh
#   # 4 卡并行 + 模型缓存到内存盘 (推荐, 同节点连续多次评估时收益巨大):
#   MODEL_PATH=/path/to/lora_ckpt NPROC_PER_NODE=4 USE_SHM_CACHE=1 bash run_inference.sh

set -e

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

VAL_JSONL=${VAL_JSONL:-"$SCRIPT_DIR/data/val.jsonl"}

# 分类任务默认参数（覆盖原来面向 chat/tts 的默认值）
# 注意: swift infer 没有 --do_sample 参数；temperature=0 即等价于贪心解码（do_sample=False）。
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
LOGPROBS=${LOGPROBS:-true}
TOP_LOGPROBS=${TOP_LOGPROBS:-20}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
ATTN_IMPL=${ATTN_IMPL:-flash_attn}
if ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$EVAL_BATCH_SIZE" -lt 1 ]; then
    echo "[ERROR] EVAL_BATCH_SIZE 必须是 >=1 的整数, 当前=$EVAL_BATCH_SIZE" >&2
    exit 1
fi

# Qwen3-Omni 专属环境变量 -------------------------------------------------------
# 关闭 talker（语音输出），分类任务不需要，且能省下大量显存与加载时间。
export ENABLE_AUDIO_OUTPUT=${ENABLE_AUDIO_OUTPUT:-0}
# 多模态 token 上限（与训练默认一致）
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}
# 缓解 MoE/长序列下的显存碎片
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

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
    echo "        示例: MODEL_PATH=$SWIFT_ROOT/output/qwen3_omni/v1_lora/checkpoint-1000 bash $0" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH 不存在或不是目录: $MODEL_PATH" >&2
    exit 1
fi

# 区分 LoRA / full 权重：
#   - LoRA 产物目录特征是存在 adapter_config.json，此时需要同时给 swift infer
#     传 --model <base> 和 --adapters <adapter_dir>；否则仅 --model <full_ckpt> 即可。
BASE_MODEL=${BASE_MODEL:-/apdcephfs_qy3/share_301069248/huggingface/Qwen/Qwen3-Omni-30B-A3B-Instruct}
ADAPTERS=""
if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    ADAPTERS="$MODEL_PATH"
    MODEL_FOR_INFER="$BASE_MODEL"
else
    MODEL_FOR_INFER="$MODEL_PATH"
fi

# /dev/shm 缓存加速 ----------------------------------------------------------
# 模型权重一般放在远程共享盘 (CephFS/NFS) 上，4 卡 DDP 时每张卡都要从远程盘读
# 完整一份模型 (~132GB)，单流带宽被瓜分后可能慢到 ~30 分钟。
# 开启 USE_SHM_CACHE=1 后，会把基座大模型预拷到本地 tmpfs (/dev/shm)，
# 后续所有 rank 都从内存盘读，实测 30 分钟 -> <2 分钟。
#
# 行为说明：
#   - 仅缓存 MODEL_FOR_INFER (LoRA 模式下=基座; 全参模式下=ckpt 本身)
#   - LoRA adapter 本身只有几十 MB，不缓存
#   - 用源目录的 safetensors.index.json mtime+size 做指纹，命中跳过拷贝
#   - /dev/shm 空间不足时自动降级到源路径 (给 WARN，不中断流程)
#   - 缓存目录默认 /dev/shm/swift_model_cache/<basename>，可由 SHM_CACHE_DIR 覆盖
#   - tmpfs 占用的是 RAM，重启自动清空；想手动清: rm -rf /dev/shm/swift_model_cache
USE_SHM_CACHE=${USE_SHM_CACHE:-1}
SHM_CACHE_DIR=${SHM_CACHE_DIR:-/dev/shm/swift_model_cache}
if [ "$USE_SHM_CACHE" = "1" ]; then
    if [ ! -d "/dev/shm" ]; then
        echo "[WARN] USE_SHM_CACHE=1 但 /dev/shm 不存在，降级使用源路径"
    else
        SRC_DIR="$MODEL_FOR_INFER"
        DST_DIR="$SHM_CACHE_DIR/$(basename "$SRC_DIR")"
        # 计算指纹：优先 model.safetensors.index.json，缺失时用 config.json
        FP_SRC=""
        for f in model.safetensors.index.json config.json; do
            if [ -f "$SRC_DIR/$f" ]; then
                FP_SRC="$SRC_DIR/$f"
                break
            fi
        done
        if [ -z "$FP_SRC" ]; then
            echo "[WARN] $SRC_DIR 下找不到 model.safetensors.index.json/config.json，无法校验缓存一致性，降级使用源路径"
        else
            FP_NEW=$(stat -c '%s_%Y' "$FP_SRC" 2>/dev/null)
            FP_FILE="$DST_DIR/.swift_cache_fingerprint"
            FP_OLD=""
            [ -f "$FP_FILE" ] && FP_OLD=$(cat "$FP_FILE" 2>/dev/null)

            if [ -d "$DST_DIR" ] && [ "$FP_OLD" = "$FP_NEW" ] && [ -n "$FP_NEW" ]; then
                echo "[INFO] /dev/shm 缓存命中: $DST_DIR (指纹=$FP_NEW)"
                MODEL_FOR_INFER="$DST_DIR"
            else
                # 估算源大小 + tmpfs 可用空间，避免拷到一半 OOM
                SRC_SIZE_KB=$(du -sk "$SRC_DIR" 2>/dev/null | awk '{print $1}')
                SHM_AVAIL_KB=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
                # 留 10% buffer
                NEED_KB=$((SRC_SIZE_KB + SRC_SIZE_KB / 10))
                if [ -z "$SRC_SIZE_KB" ] || [ "$SHM_AVAIL_KB" -lt "$NEED_KB" ]; then
                    echo "[WARN] /dev/shm 空间不足 (需要约 $((SRC_SIZE_KB/1024/1024))GB, 可用 $((SHM_AVAIL_KB/1024/1024))GB)，降级使用源路径"
                else
                    echo "[INFO] 开始把模型缓存到 /dev/shm: $SRC_DIR -> $DST_DIR"
                    echo "       源大小 ~$((SRC_SIZE_KB/1024/1024))GB，tmpfs 可用 ~$((SHM_AVAIL_KB/1024/1024))GB"
                    rm -rf "$DST_DIR"
                    mkdir -p "$DST_DIR"
                    T0=$(date +%s)
                    # cp -a 保留链接/时间戳；--reflink=auto 在支持的 fs 上做 CoW (tmpfs 不支持但不会报错)
                    if cp -a "$SRC_DIR/." "$DST_DIR/"; then
                        echo "$FP_NEW" > "$FP_FILE"
                        T1=$(date +%s)
                        echo "[INFO] 缓存完成，耗时 $((T1-T0)) 秒"
                        MODEL_FOR_INFER="$DST_DIR"
                    else
                        echo "[WARN] 缓存到 /dev/shm 失败，降级使用源路径"
                        rm -rf "$DST_DIR"
                    fi
                fi
            fi
        fi
    fi
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
if [ "$USE_SHM_CACHE" = "1" ]; then
    echo "[INFO] USE_SHM_CACHE        = 1  [基座模型已就绪于内存盘]"
    echo "[INFO] MODEL_FOR_INFER      = $MODEL_FOR_INFER"
else
    echo "[INFO] USE_SHM_CACHE        = 0  [设为 1 可把模型缓存到 /dev/shm 加速多卡加载]"
fi
echo "[INFO] EVAL_BATCH_SIZE      = $EVAL_BATCH_SIZE  [每个进程的推理 batch 大小]"
echo "[INFO] ATTN_IMPL            = $ATTN_IMPL"
echo "[INFO] ENABLE_AUDIO_OUTPUT  = $ENABLE_AUDIO_OUTPUT  [0=关闭 talker, 分类任务不需要]"
echo "[INFO] IMAGE_MAX_TOKEN_NUM=$IMAGE_MAX_TOKEN_NUM VIDEO_MAX_TOKEN_NUM=$VIDEO_MAX_TOKEN_NUM FPS_MAX_FRAMES=$FPS_MAX_FRAMES"
echo "[INFO] temperature=$TEMPERATURE  [0 表示贪心], top_p=$TOP_P, max_new_tokens=$MAX_NEW_TOKENS"
echo "[INFO] logprobs=$LOGPROBS, top_logprobs=$TOP_LOGPROBS"

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
    PY_BIN="$(dirname "$CONDA_SWIFT_BIN")/python"
elif command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
    PY_BIN=$(command -v python3 || command -v python)
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    echo "[INFO] 未找到 swift CLI，回退到: $CONDA_PREFIX/bin/python -m swift.cli.main"
    SWIFT_CMD=("$CONDA_PREFIX/bin/python" -m swift.cli.main)
    PY_BIN="$CONDA_PREFIX/bin/python"
else
    echo "[WARN] 未找到 swift CLI 且未检测到 CONDA_PREFIX，回退到 PATH 中的 python -m swift.cli.main"
    echo "       若报 ModuleNotFoundError，请先 conda activate 正确的环境，或显式设置 CONDA_SWIFT_BIN。"
    SWIFT_CMD=(python -m swift.cli.main)
    PY_BIN=$(command -v python3 || command -v python)
fi

# 依赖前置检查：Qwen3-Omni 必装包，避免起完多 rank 才发现缺包
#   - qwen_omni_utils >= 0.0.9：模型加载强制依赖（swift/model/models/qwen.py 中
#     调用了 transformers.utils.versions.require_version('qwen_omni_utils>=0.0.9')，
#     该函数走 importlib.metadata.version()，仅看 dist-info 元数据，不看 import）
#   - soundfile / decord：音视频样本预处理依赖
# 设 SKIP_DEP_CHECK=1 可跳过该检查
if [ "${SKIP_DEP_CHECK:-0}" != "1" ]; then
    "$PY_BIN" - <<'PY' || {
import importlib, importlib.metadata, sys
missing = []
# 必须 import 成功 + dist-info 元数据存在 + 版本 >= 0.0.9
try:
    importlib.import_module("qwen_omni_utils")
    v = importlib.metadata.version("qwen_omni_utils")
    parts = tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    if parts < (0, 0, 9):
        missing.append(("qwen_omni_utils", f"version {v} < 0.0.9"))
except Exception as e:
    missing.append(("qwen_omni_utils", repr(e)))
for pkg in ("soundfile", "decord"):
    try:
        importlib.import_module(pkg)
    except Exception as e:
        missing.append((pkg, repr(e)))
if missing:
    sys.stderr.write("[ERROR] Qwen3-Omni 缺失以下依赖，请先安装后再启动推理：\n")
    for pkg, err in missing:
        sys.stderr.write(f"  - {pkg}: {err}\n")
    sys.stderr.write('\n建议：\n  pip install -U "qwen_omni_utils>=0.0.9" soundfile decord\n')
    sys.exit(2)
PY
        echo "[FATAL] 依赖检查未通过，已中断启动。"
        exit 2
    }
    echo "[INFO] Qwen3-Omni 依赖检查通过 (qwen_omni_utils / soundfile / decord)"
fi

INFER_ARGS=(
    --model "$MODEL_FOR_INFER"
    --model_type qwen3_omni_moe
    --template qwen3_omni
    --val_dataset "$EFFECTIVE_VAL_JSONL"
    --result_path "$RESULT_PATH"
    --attn_impl "$ATTN_IMPL"
    --torch_dtype bfloat16
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --logprobs "$LOGPROBS"
    --top_logprobs "$TOP_LOGPROBS"
    --max_batch_size "$EVAL_BATCH_SIZE"
    --stream false
)
if [ -n "$ADAPTERS" ]; then
    INFER_ARGS+=(--adapters "$ADAPTERS")
fi

"${SWIFT_CMD[@]}" infer "${INFER_ARGS[@]}" "$@"

echo "[INFO] 推理完成，结果保存在: $RESULT_PATH"

if [ -f "$RESULT_PATH" ]; then
    num_results=$(wc -l < "$RESULT_PATH")
    echo "[INFO] 共生成 $num_results 条结果"
    echo "[INFO] 下一步可直接运行评估:"
    echo "       bash $SCRIPT_DIR/run_eval.sh \\"
    echo "            RESULT_PATH=$RESULT_PATH \\"
    echo "            VAL_JSONL=$VAL_JSONL"
fi
