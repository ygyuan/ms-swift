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
# Suppress the empty `<think>\n\n</think>\n\n` block that Qwen3's chat template
# normally injects as the assistant response prefix even when
# enable_thinking=false. Setting `--response_prefix ''` tells swift to use an
# empty string instead of the meta default `non_thinking_prefix`, which:
#   1. removes the <think></think> tokens from the prompt's assistant prefix,
#      so the prompt fed to the model truly has no think scaffolding;
#   2. removes the same string from the recorded `response` field (swift
#      otherwise prepends `response_prefix` back to the response text in
#      base.py: `response = response_prefix + response`).
# As a result the jsonl response starts directly with the predicted category.
# Set to "keep" to restore the default behavior.
STRIP_THINK_PREFIX="${STRIP_THINK_PREFIX:-true}"
# Optional: cap the number of evaluation samples (0 = use all).
MAX_SAMPLES="${MAX_SAMPLES:-0}"
# Emit per-token logprobs in the result jsonl. The downstream evaluator
# (eval_event.py) reads `choices[0].logprobs.content[0].logprob` (i.e. the
# first generated token) as a per-sample confidence in [0,1] (= exp(logprob))
# to compute precision/recall/F1 vs threshold curves.
LOGPROBS="${LOGPROBS:-true}"
TOP_LOGPROBS="${TOP_LOGPROBS:-3}"

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
            "solution": str(label),  # also keep 'solution', matching --columns mapping
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

# Build the --response_prefix arg lazily so we can pass an empty string
# when STRIP_THINK_PREFIX=true. swift's CLI parses `--response_prefix ''`
# as a literal empty value (its priority is higher than `non_thinking_prefix`,
# see swift/template/base.py:_get_response_prefix), suppressing the
# default `<think>\n\n</think>\n\n` block from both the prompt and the
# recorded response.
# NOTE: We must use a bash array (not a plain string) here, otherwise word
# splitting would turn `--response_prefix ""` into the two literal tokens
# `--response_prefix` and `""` (the quote characters themselves), and
# argparse would receive '""' as the prefix value -- not an empty string.
RESPONSE_PREFIX_ARGS=()
if [ "${STRIP_THINK_PREFIX}" = "true" ] || [ "${STRIP_THINK_PREFIX}" = "1" ]; then
    RESPONSE_PREFIX_ARGS=(--response_prefix "")
    echo "[infer] STRIP_THINK_PREFIX=true -> --response_prefix '' (no <think></think> in output)"
else
    echo "[infer] STRIP_THINK_PREFIX=false -> keep template default non_thinking_prefix"
fi

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
            "${RESPONSE_PREFIX_ARGS[@]}" \
            --logprobs ${LOGPROBS} \
            --top_logprobs ${TOP_LOGPROBS} \
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
            "${RESPONSE_PREFIX_ARGS[@]}" \
            --logprobs ${LOGPROBS} \
            --top_logprobs ${TOP_LOGPROBS} \
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

# ===== Post-process: strip leading empty-think block from response & logprobs =====
# Some checkpoints (e.g. those whose tokenizer's chat_template hard-codes
#   '<think>\n\n</think>\n\n' for enable_thinking=false at the assistant
#   generation prompt position) are trained to *generate* this empty think
# block as the very first tokens of the answer, even when swift sets
# `--response_prefix ''` (which only suppresses the *prompt-side* prefix, not
# the model's own learned behavior).
#
# Concretely, the per-token logprobs sequence in such jsonl looks like:
#   tokens[0..3] = '<think>', '\n\n', '</think>', '\n\n'   (logprob ~0)
#   tokens[4..]  = the real answer (e.g. '安全', '恶意', ...)
# We strip the leading think block (if any) from BOTH the textual `response`
# field AND the `logprobs.content` array, so that downstream consumers
# (eval_event.py, threshold sweep, etc.) see logprobs[0] as the first
# *answer* token directly, without needing any compatibility shim.
#
# Set STRIP_THINK_POSTPROC=false to skip this rewrite (e.g. for debugging).
STRIP_THINK_POSTPROC="${STRIP_THINK_POSTPROC:-true}"
if [ "${STRIP_THINK_POSTPROC}" = "true" ] || [ "${STRIP_THINK_POSTPROC}" = "1" ]; then
    echo "[postproc] stripping leading <think>...</think> from response & logprobs in ${OUT_FILE}"
    /data/miniconda3/envs/env-3.12.11/bin/python - <<PYEOF
import json, os, re, sys
src = "${OUT_FILE}"
tmp = src + ".tmp"

# Match a leading <think>...</think> block (with optional surrounding
# whitespace/newlines) at the very start of the response.
THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
# Tokens that the model emits as part of the empty-think scaffolding.
# We only drop a contiguous prefix of these, stopping at the first
# non-think token (so we never accidentally drop real answer tokens that
# happen to be whitespace).
THINK_TOKEN_SET = {"<think>", "</think>"}
WS_TOKEN_SET = {"\n", "\n\n", " ", "  ", "\t"}

n_in = 0
n_stripped_text = 0
n_stripped_logprobs = 0
with open(src, "r", encoding="utf-8") as fr, open(tmp, "w", encoding="utf-8") as fw:
    for line in fr:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        n_in += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            fw.write(line + "\n")
            continue
        # 1) strip from textual response
        resp = obj.get("response")
        if isinstance(resp, str):
            new_resp = THINK_RE.sub("", resp, count=1)
            if new_resp != resp:
                obj["response"] = new_resp
                n_stripped_text += 1
        # 2) strip aligned prefix from logprobs.content
        lp = obj.get("logprobs")
        if isinstance(lp, dict):
            content = lp.get("content")
            if isinstance(content, list) and content:
                # Find end of the leading '<think>...</think>(\n\n)?' run.
                # Walk forward until we have seen </think> AND consumed any
                # immediately following whitespace tokens, then cut.
                seen_close = False
                cut = 0
                for i, tok in enumerate(content):
                    if not isinstance(tok, dict):
                        break
                    t = tok.get("token", "")
                    if not seen_close:
                        if t in THINK_TOKEN_SET or t in WS_TOKEN_SET or t == "":
                            if t == "</think>":
                                seen_close = True
                            cut = i + 1
                            continue
                        else:
                            # First non-think, non-ws token before seeing </think>:
                            # the response does not start with a think block in
                            # the token stream -> do not strip anything.
                            cut = 0
                            break
                    else:
                        # already past </think>: only consume immediately
                        # following whitespace tokens, then stop.
                        if t in WS_TOKEN_SET:
                            cut = i + 1
                            continue
                        break
                if cut > 0 and seen_close:
                    lp["content"] = content[cut:]
                    n_stripped_logprobs += 1
        fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

os.replace(tmp, src)
print(f"[postproc] processed {n_in} lines; stripped think from response in {n_stripped_text}, from logprobs in {n_stripped_logprobs}")
PYEOF
fi

echo "[infer] done (post-processed) -> ${OUT_FILE}"
