#!/bin/bash
# Batch inference on GSM8K test set with the GRPO-trained Qwen3.5-2B model.
# Output: <CKPT>/infer_gsm8k.jsonl
set -e

export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.12.11/lib:$LD_LIBRARY_PATH
export WANDB_DISABLED=true
export PYTHONWARNINGS="ignore"

# ===== Config =====
CKPT="${CKPT:-/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3.5-2B/v12-20260423-120546/checkpoint-3736}"
TEST_PARQUET="${TEST_PARQUET:-/apdcephfs_qy3/share_301069248/huggingface/gsm8k/main/test-00000-of-00001.parquet}"
OUT_FILE="${OUT_FILE:-${CKPT}/infer_gsm8k.jsonl}"
# Comma-separated GPU ids for data-parallel inference, e.g. "0,1,2,3"
GPU="${GPU:-0,1,2,3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TP="${TP:-1}"
# Inference backend: vllm | pt
# Use 'pt' when running models that require transformers>=5.x (vllm not yet compatible).
BACKEND="${BACKEND:-pt}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"

SYSTEM_PROMPT="You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."

# ===== Build a swift-compatible jsonl from parquet =====
PROMPT_JSONL="${CKPT}/gsm8k_test_prompts.jsonl"
mkdir -p "$(dirname "$OUT_FILE")"

/data/miniconda3/envs/env-3.12.11/bin/python - <<PYEOF
import json, pandas as pd, os
df = pd.read_parquet("${TEST_PARQUET}")
sys_prompt = """${SYSTEM_PROMPT}"""
out = "${PROMPT_JSONL}"
with open(out, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": row["question"]},
            ],
            "solution": row["answer"],
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"[prepare] wrote {len(df)} prompts -> {out}")
PYEOF

# ===== Batch inference =====
# Parse GPU list -> array
IFS=',' read -ra GPU_ARR <<< "${GPU}"
NUM_GPUS=${#GPU_ARR[@]}
echo "[infer] backend=${BACKEND}, gpus=${GPU} (n=${NUM_GPUS})"

# ----- Helper: run a single shard on one GPU -----
run_shard() {
    local gpu_id="$1"
    local shard_jsonl="$2"
    local shard_out="$3"
    local log_file="$4"

    if [ "${BACKEND}" = "vllm" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        /data/miniconda3/envs/env-3.12.11/bin/swift infer \
            --model "${CKPT}" \
            --val_dataset "${shard_jsonl}" \
            --infer_backend vllm \
            --vllm_max_model_len 8192 \
            --vllm_gpu_memory_utilization 0.85 \
            --vllm_tensor_parallel_size ${TP} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p 1.0 \
            --stream false \
            --result_path "${shard_out}" > "${log_file}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        /data/miniconda3/envs/env-3.12.11/bin/swift infer \
            --model "${CKPT}" \
            --val_dataset "${shard_jsonl}" \
            --infer_backend pt \
            --max_batch_size ${MAX_BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p 1.0 \
            --stream false \
            --result_path "${shard_out}" > "${log_file}" 2>&1
    fi
}

if [ "${NUM_GPUS}" -le 1 ]; then
    # ----- Single GPU path (no sharding) -----
    run_shard "${GPU_ARR[0]:-0}" "${PROMPT_JSONL}" "${OUT_FILE}" "/dev/stdout"
else
    # ----- Multi-GPU data-parallel path -----
    SHARD_DIR="${CKPT}/_shards_$$"
    mkdir -p "${SHARD_DIR}"
    echo "[infer] sharding prompts into ${SHARD_DIR}/"

    # Split prompts into NUM_GPUS shards (round-robin to balance length).
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

    # Launch one process per GPU in parallel.
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

    # Wait for all shards; abort if any fail.
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

    # Merge shard outputs.
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

    # Optional cleanup (keep logs/shards by default for debugging).
    if [ "${KEEP_SHARDS:-0}" != "1" ]; then
        rm -rf "${SHARD_DIR}"
    else
        echo "[infer] shard intermediates kept at ${SHARD_DIR}"
    fi
fi

echo "[infer] done -> ${OUT_FILE}"
