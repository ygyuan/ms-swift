#!/bin/bash
# Batch inference on the event-classification test set with the GRPO-trained
# Qwen3-1.7B model. Output: <CKPT>/infer_event.jsonl
set -e

export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.12.11/lib:$LD_LIBRARY_PATH
export WANDB_DISABLED=true
export PYTHONWARNINGS="ignore"

# ===== Config =====
CKPT="${CKPT:-/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3-1.7B/event/v2/checkpoint-0}"
# Test set: alpaca-style jsonl with fields {instruction,input,output,label,...}.
# Defaults to the training jsonl so that this script is runnable out-of-the-box;
# in practice you should override TEST_JSONL with a held-out set.
TEST_JSONL="${TEST_JSONL:-/apdcephfs_qy3/share_301069248/data/video/event_rag/merge/train_ayden_v1.jsonl}"
OUT_FILE="${OUT_FILE:-${CKPT}/infer_event.jsonl}"
# Comma-separated GPU ids for data-parallel inference, e.g. "0,1,2,3".
GPU="${GPU:-0,1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TP="${TP:-1}"
# Inference backend: vllm | pt
# Default to `pt` because this environment ships transformers>=5.x which
# is not yet compatible with vllm (see also infer_gsm8k_qwen3.5.sh).
# If you set BACKEND=vllm but vllm is not importable, we automatically
# fall back to pt below.
BACKEND="${BACKEND:-pt}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-10240}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.85}"
# Required when the checkpoint dir matches multiple model_types
# (e.g. qwen3 / qwen3_emb / qwen3_reranker for Qwen3-1.7B). Aligns with the
# base model used in train_event_qwen3.sh.
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
# Same disambiguation is also needed for `template_type`. swift may otherwise
# match multiple candidates such as
#   ['qwen3', 'deepseek_r1', 'qwen3_guard', 'yufeng_xguard',
#    'qwen3_thinking', 'qwen3_nothinking']
# `qwen3` is a mixed thinking/no-thinking template; combined with
# `--enable_thinking false` below it reproduces the training-time behavior.
TEMPLATE="${TEMPLATE:-qwen3}"
# Match training (--enable_thinking false) so the model does not emit
# <think>...</think> blocks that would hurt format_rate.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# Optional: cap the number of evaluation samples (0 = use all).
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# ===== Build a swift-compatible jsonl from the alpaca-style test jsonl =====
PROMPT_JSONL="${CKPT}/event_test_prompts.jsonl"
mkdir -p "$(dirname "$OUT_FILE")"

/data/miniconda3/envs/env-3.12.11/bin/python - <<PYEOF
import json, os, sys
src = "${TEST_JSONL}"
dst = "${PROMPT_JSONL}"
max_n = int("${MAX_SAMPLES}" or "0")
n = 0
with open(src, "r", encoding="utf-8") as fr, open(dst, "w", encoding="utf-8") as fw:
    for line in fr:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        instr = (obj.get("instruction") or "").strip()
        inp = (obj.get("input") or "").strip()
        label = obj.get("label") or obj.get("solution") or ""
        if not instr or not label:
            continue
        # Keep the prompt format identical to the training data layout:
        # system  := instruction (contains the "# 类别列表" block)
        # user    := input (the raw text to be classified)
        item = {
            "messages": [
                {"role": "system", "content": instr},
                {"role": "user",   "content": inp},
            ],
            "label": str(label),
            "solution": str(label),  # also keep `solution`, matching --columns mapping
        }
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        n += 1
        if max_n and n >= max_n:
            break
print(f"[prepare] wrote {n} prompts -> {dst}")
PYEOF

# ===== Batch inference =====
# Auto-fallback when user requested vllm but it is not importable in this
# Python env (typical when transformers>=5.x is installed and vllm is not).
if [ "${BACKEND}" = "vllm" ]; then
    if ! /data/miniconda3/envs/env-3.12.11/bin/python -c "import vllm" >/dev/null 2>&1; then
        echo "[warn] BACKEND=vllm but 'vllm' is not importable in this env; falling back to BACKEND=pt"
        BACKEND="pt"
    fi
fi
IFS=',' read -ra GPU_ARR <<< "${GPU}"
NUM_GPUS=${#GPU_ARR[@]}
echo "[infer] backend=${BACKEND}, gpus=${GPU} (n=${NUM_GPUS})"

run_shard() {
    local gpu_id="$1"
    local shard_jsonl="$2"
    local shard_out="$3"
    local log_file="$4"

    if [ "${BACKEND}" = "vllm" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        /data/miniconda3/envs/env-3.12.11/bin/swift infer \
            --model "${CKPT}" \
            --model_type "${MODEL_TYPE}" \
            --template "${TEMPLATE}" \
            --val_dataset "${shard_jsonl}" \
            --infer_backend vllm \
            --vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
            --vllm_gpu_memory_utilization ${VLLM_MEM_UTIL} \
            --vllm_tensor_parallel_size ${TP} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p 1.0 \
            --enable_thinking ${ENABLE_THINKING} \
            --stream false \
            --result_path "${shard_out}" > "${log_file}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        /data/miniconda3/envs/env-3.12.11/bin/swift infer \
            --model "${CKPT}" \
            --model_type "${MODEL_TYPE}" \
            --template "${TEMPLATE}" \
            --val_dataset "${shard_jsonl}" \
            --infer_backend pt \
            --max_batch_size ${MAX_BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p 1.0 \
            --enable_thinking ${ENABLE_THINKING} \
            --stream false \
            --result_path "${shard_out}" > "${log_file}" 2>&1
    fi
}

if [ "${NUM_GPUS}" -le 1 ]; then
    run_shard "${GPU_ARR[0]:-0}" "${PROMPT_JSONL}" "${OUT_FILE}" "/dev/stdout"
else
    SHARD_DIR="${CKPT}/_event_shards_$$"
    mkdir -p "${SHARD_DIR}"
    echo "[infer] sharding prompts into ${SHARD_DIR}/"

    /data/miniconda3/envs/env-3.12.11/bin/python - <<PYEOF
import os
n = ${NUM_GPUS}
src = "${PROMPT_JSONL}"
out_dir = "${SHARD_DIR}"
shards = [open(os.path.join(out_dir, f"shard_{i}.jsonl"), "w", encoding="utf-8") for i in range(n)]
total = 0
with open(src, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        shards[i % n].write(line)
        total += 1
for fp in shards:
    fp.close()
print(f"[shard] split {total} prompts into {n} shards")
PYEOF

    pids=()
    for ((i=0; i<NUM_GPUS; i++)); do
        gpu_id="${GPU_ARR[$i]}"
        shard_in="${SHARD_DIR}/shard_${i}.jsonl"
        shard_out="${SHARD_DIR}/result_${i}.jsonl"
        shard_log="${SHARD_DIR}/log_${i}.txt"
        echo "[infer] launching shard ${i} on GPU ${gpu_id} (log: ${shard_log})"
        run_shard "${gpu_id}" "${shard_in}" "${shard_out}" "${shard_log}" &
        pids+=($!)
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            fail=1
        fi
    done
    if [ "${fail}" -ne 0 ]; then
        echo "[infer] ERROR: at least one shard failed. See ${SHARD_DIR}/log_*.txt"
        exit 1
    fi

    echo "[infer] merging shard outputs -> ${OUT_FILE}"
    > "${OUT_FILE}"
    for ((i=0; i<NUM_GPUS; i++)); do
        shard_out="${SHARD_DIR}/result_${i}.jsonl"
        if [ -f "${shard_out}" ]; then
            cat "${shard_out}" >> "${OUT_FILE}"
        else
            echo "[warn] missing shard output: ${shard_out}"
        fi
    done

    if [ "${KEEP_SHARDS:-0}" != "1" ]; then
        rm -rf "${SHARD_DIR}"
    else
        echo "[infer] shard intermediates kept at ${SHARD_DIR}"
    fi
fi

echo "[infer] done -> ${OUT_FILE}"
