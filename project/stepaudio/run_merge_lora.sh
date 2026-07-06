#!/usr/bin/env bash
# StepAudio2-mini  Merge-LoRA 脚本
# 将 LoRA adapter 与基座模型合并, 导出为标准 HuggingFace safetensors 格式,
# 每个 shard 大小上限 5GB (由 --max_shard_size 控制).
#
# 底层调用 swift export --merge_lora true, 内部逻辑见:
#   swift/pipelines/export/merge_lora.py -> save_checkpoint(..., max_shard_size='5GB')
#
# 环境变量 (均可覆盖默认值, 也支持 `bash run_merge_lora.sh KEY=VALUE ...` 形式):
#   ADAPTER_PATH   [必填] LoRA adapter 目录, 里面必须有 adapter_config.json
#                  例如: <SWIFT_ROOT>/output/v0-xxx/checkpoint-1600
#   BASE_MODEL     基座模型路径, 默认 stepfun-ai/Step-Audio-2-mini
#   OUTPUT_DIR     合并后输出目录 (默认 <ADAPTER_PATH>-merged)
#   MAX_SHARD_SIZE 单个 safetensors 分片上限 (默认 5GB, 满足 <=5GB 要求)
#   SAFE_SERIALIZATION  是否用 safetensors 保存 (默认 true, HF 推荐格式)
#   TORCH_DTYPE    合并时权重加载 dtype (默认 bfloat16, 与训练/推理一致)
#   DEVICE_MAP     模型加载 device_map (默认 cpu 最稳但慢; 显存充足可改 auto/cuda:0)
#   ATTN_IMPL      注意力实现 (默认 eager; step_audio2_mini 目前仅支持 eager)
#   EXIST_OK       输出目录已存在时是否覆盖 (默认 1=允许; 0 时若已存在会报错)
#   CUDA_VISIBLE_DEVICES  可见卡 (默认 0; DEVICE_MAP=cpu 时不会真正用到 GPU)
#
# 用法示例:
#   # 最简: 用默认基座 + bf16 + cpu 合并, 结果放 <ADAPTER_PATH>-merged
#   ADAPTER_PATH=/path/to/checkpoint-1600 bash run_merge_lora.sh
#
#   # 指定输出目录 + 用 GPU 加速合并 (基座+lora 需能塞进单卡)
#   ADAPTER_PATH=/path/to/checkpoint-1600 \
#   OUTPUT_DIR=/path/to/merged_ckpt-1600 \
#   DEVICE_MAP=auto CUDA_VISIBLE_DEVICES=0 \
#   bash run_merge_lora.sh
#
#   # 分片改为 2GB (对存储/传输更友好, 但小文件更多)
#   ADAPTER_PATH=/path/to/checkpoint-1600 MAX_SHARD_SIZE=2GB bash run_merge_lora.sh
#
#   # 一行 KEY=VALUE 形式:
#   bash run_merge_lora.sh ADAPTER_PATH=/path/to/checkpoint-1600 MAX_SHARD_SIZE=5GB

set -e

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 允许 `bash run_merge_lora.sh KEY=VALUE ...` 覆盖环境变量
for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
    esac
done

# ---- 必填参数 ----
if [ -z "${ADAPTER_PATH:-}" ]; then
    echo "[ERROR] 请通过环境变量 ADAPTER_PATH 指定 LoRA adapter 目录." >&2
    echo "        示例: ADAPTER_PATH=$SWIFT_ROOT/output/v0-xxx/checkpoint-1600 bash $0" >&2
    exit 1
fi
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "[ERROR] ADAPTER_PATH 不存在或不是目录: $ADAPTER_PATH" >&2
    exit 1
fi
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "[ERROR] ADAPTER_PATH 中未找到 adapter_config.json, 不是合法的 LoRA 产物: $ADAPTER_PATH" >&2
    echo "        (若这是全参 checkpoint, 无需 merge, 可直接推理/使用)" >&2
    exit 1
fi

# ---- 默认参数 ----
BASE_MODEL=${BASE_MODEL:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] BASE_MODEL 不存在或不是目录: $BASE_MODEL" >&2
    exit 1
fi

# 默认输出目录: <ADAPTER_PATH>-merged, 与 swift export 内置默认 (<adapter>-merged) 一致,
# 且明确写出来便于后续脚本引用
if [ -z "${OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="${ADAPTER_PATH%/}-merged"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"

MAX_SHARD_SIZE=${MAX_SHARD_SIZE:-5GB}
SAFE_SERIALIZATION=${SAFE_SERIALIZATION:-true}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
DEVICE_MAP=${DEVICE_MAP:-cpu}
# StepAudio2-mini 目前仅支持 eager attention (与 run_train_swift.sh / run_inference.sh 保持一致);
# sdpa/flash_attention_2 会因为 audio encoder 侧的 mask 组装差异报错.
ATTN_IMPL=${ATTN_IMPL:-eager}
EXIST_OK=${EXIST_OK:-1}

# CUDA_VISIBLE_DEVICES: DEVICE_MAP=cpu 时不会真正用到 GPU, 但 swift 内部仍会 import cuda.
# 默认给 0 号卡, 避免 import 阶段找不到设备.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 若目标已存在且不允许覆盖, 提前报错; 允许覆盖 (EXIST_OK=1) 时删除旧目录, 避免 swift 里的
# "The weight directory for the merged LoRA already exists ... skipping the saving process." 静默跳过
if [ -d "$OUTPUT_DIR" ]; then
    if [ "$EXIST_OK" = "1" ]; then
        echo "[WARN] 输出目录已存在, EXIST_OK=1 -> 清理后覆盖: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    else
        echo "[ERROR] 输出目录已存在: $OUTPUT_DIR (设置 EXIST_OK=1 可自动覆盖)" >&2
        exit 1
    fi
fi

echo "[INFO] ADAPTER_PATH        = $ADAPTER_PATH"
echo "[INFO] BASE_MODEL          = $BASE_MODEL"
echo "[INFO] OUTPUT_DIR          = $OUTPUT_DIR"
echo "[INFO] MAX_SHARD_SIZE      = $MAX_SHARD_SIZE   [每个 shard 上限]"
echo "[INFO] SAFE_SERIALIZATION  = $SAFE_SERIALIZATION [true=safetensors, HF 推荐]"
echo "[INFO] TORCH_DTYPE         = $TORCH_DTYPE"
echo "[INFO] DEVICE_MAP          = $DEVICE_MAP       [cpu 最稳但慢, auto/cuda:0 更快但需大显存]"
echo "[INFO] ATTN_IMPL           = $ATTN_IMPL         [step_audio2_mini 仅支持 eager]"
echo "[INFO] CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"

# ---- 选择 swift 入口 (与 run_inference.sh / run_train_swift.sh 保持一致的探测逻辑) ----
# 目的: 避免 rc 钩子把 python 切到别的 conda 环境 (例如 /root/custom.bashrc 会切到 3.6.8)
CONDA_SWIFT_BIN="${CONDA_SWIFT_BIN:-}"
if [ -z "$CONDA_SWIFT_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/swift" ]; then
    CONDA_SWIFT_BIN="$CONDA_PREFIX/bin/swift"
fi
# 若当前 shell 未 activate conda, 兜底到仓库推荐环境 env-3.12.11 (与训练脚本一致)
if [ -z "$CONDA_SWIFT_BIN" ] && [ -x "/data/miniconda3/envs/env-3.12.11/bin/swift" ]; then
    CONDA_SWIFT_BIN="/data/miniconda3/envs/env-3.12.11/bin/swift"
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
    echo "[INFO] 未找到 swift CLI, 回退到 $CONDA_PREFIX/bin/python -m swift.cli.main"
    SWIFT_CMD=("$CONDA_PREFIX/bin/python" -m swift.cli.main)
    PY_BIN="$CONDA_PREFIX/bin/python"
else
    echo "[WARN] 未找到 swift CLI 且未检测到 CONDA_PREFIX, 回退到 PATH 中的 python -m swift.cli.main"
    SWIFT_CMD=(python -m swift.cli.main)
    PY_BIN=$(command -v python3 || command -v python)
fi

# ---- 依赖前置检查: StepAudio2-mini 必需的运行时依赖 ----
# 只做非常轻量的检查: transformers 能加载, 并能通过 model_type=step_audio2_mini 找到注册项.
# StepAudio2-mini 走的是 trust_remote_code (仓库内 modeling_step_audio2.py) 路径,
# 不像 Qwen3-Omni 需要特定 transformers 版本, 所以这里不硬卡 transformers 版本.
if [ "${SKIP_DEP_CHECK:-0}" != "1" ]; then
    "$PY_BIN" - <<'PY' || {
import sys
missing = []
try:
    import transformers  # noqa: F401
except Exception as e:
    missing.append(("transformers", repr(e)))
try:
    import torch  # noqa: F401
except Exception as e:
    missing.append(("torch", repr(e)))
try:
    # 注意: MODEL_MAPPING 定义在 swift.model.model_meta, 通过 swift.model 顶层暴露;
    # 之前误写作 swift.llm 会 ModuleNotFoundError.
    from swift.model import MODEL_MAPPING  # noqa
    from swift.model.constant import MLLMModelType
    if MLLMModelType.step_audio2_mini not in MODEL_MAPPING:
        missing.append(("swift", "MODEL_MAPPING 中未注册 step_audio2_mini, 请更新 ms-swift"))
except Exception as e:
    missing.append(("swift", repr(e)))
if missing:
    sys.stderr.write("[ERROR] StepAudio2-mini merge-lora 缺失以下依赖:\n")
    for pkg, err in missing:
        sys.stderr.write(f"  - {pkg}: {err}\n")
    sys.stderr.write('\n建议:\n')
    sys.stderr.write('  1) 先 conda activate 已装好 ms-swift 的环境 (仓库推荐 env-3.12.11)\n')
    sys.stderr.write('  2) 或显式设置 CONDA_SWIFT_BIN 指向可用 swift\n')
    sys.exit(2)
PY
        echo "[FATAL] 依赖检查未通过, 已中断."
        exit 2
    }
    echo "[INFO] StepAudio2-mini merge-lora 依赖检查通过 (transformers / torch / swift.step_audio2_mini)"
fi

# ---- 组装 swift export 参数 ----
# 关键参数说明:
#   --merge_lora true     启用 merge & unload
#   --adapters <path>     LoRA 目录 (可多选, 这里只用一个)
#   --model <base>        基座模型 (LoRA 场景必填, 否则会尝试从 adapter_config 里推)
#   --model_type step_audio2_mini
#   --template  step_audio2_mini
#   --attn_impl eager     StepAudio2-mini 目前仅支持 eager, 与训练/推理保持一致
#   --output_dir <out>    输出目录
#   --safe_serialization  用 safetensors 格式 (HF 官方推荐)
#   --max_shard_size 5GB  每个分片文件上限, 5GB 是 HF hub 单文件推荐上限
#   --torch_dtype bfloat16  加载 dtype, 与训练/推理一致以避免精度回退
#   --exist_ok true       允许目标目录已存在 (前面已清理, 这里保险)
EXPORT_ARGS=(
    --model "$BASE_MODEL"
    --adapters "$ADAPTER_PATH"
    --model_type step_audio2_mini
    --template step_audio2_mini
    --merge_lora true
    --output_dir "$OUTPUT_DIR"
    --safe_serialization "$SAFE_SERIALIZATION"
    --max_shard_size "$MAX_SHARD_SIZE"
    --torch_dtype "$TORCH_DTYPE"
    --device_map "$DEVICE_MAP"
    --attn_impl "$ATTN_IMPL"
    --exist_ok true
)

echo "[INFO] 开始 merge-lora ..."
echo "[INFO] 命令: ${SWIFT_CMD[*]} export ${EXPORT_ARGS[*]}"

T0=$(date +%s)
"${SWIFT_CMD[@]}" export "${EXPORT_ARGS[@]}"
T1=$(date +%s)

echo "[INFO] merge-lora 完成, 耗时 $((T1-T0)) 秒"
echo "[INFO] 合并后模型目录: $OUTPUT_DIR"

# ---- 结果核对: 打印关键文件, 校验分片大小 ----
if [ -d "$OUTPUT_DIR" ]; then
    echo "[INFO] 输出目录内容:"
    ls -lh "$OUTPUT_DIR" | awk 'NR==1 || /\.(safetensors|bin|json|txt|py|model)$/'

    # 校验没有超过 MAX_SHARD_SIZE 的分片 (只是提示性检查, 不作硬失败)
    # 把 "5GB" 之类简单转成字节数; 只支持 GB/MB, 与 huggingface 内部保持一致
    LIMIT_STR="$MAX_SHARD_SIZE"
    LIMIT_BYTES=""
    case "$LIMIT_STR" in
        *GB|*gb) LIMIT_BYTES=$(( ${LIMIT_STR%[Gg][Bb]} * 1024 * 1024 * 1024 )) ;;
        *MB|*mb) LIMIT_BYTES=$(( ${LIMIT_STR%[Mm][Bb]} * 1024 * 1024 )) ;;
    esac
    if [ -n "$LIMIT_BYTES" ]; then
        BAD=$(find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) \
                -size +"${LIMIT_BYTES}c" 2>/dev/null || true)
        if [ -n "$BAD" ]; then
            echo "[WARN] 以下分片超过 $MAX_SHARD_SIZE, 请检查:" >&2
            echo "$BAD" >&2
        else
            echo "[INFO] 所有分片均 <= $MAX_SHARD_SIZE ✅"
        fi
    fi

    echo
    echo "[INFO] 下一步可直接把该目录当基座 (无需 --adapters) 用于推理:"
    echo "       MODEL_PATH=$OUTPUT_DIR bash $SCRIPT_DIR/run_inference.sh"
else
    echo "[ERROR] 未生成输出目录: $OUTPUT_DIR" >&2
    exit 1
fi
