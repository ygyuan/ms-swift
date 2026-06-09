#!/bin/bash
# One-click: run inference on the event-classification test set, then
# evaluate accuracy / format-rate using EventAccuracy / EventFormat ORMs.
#
# Usage:
#   bash project/scripts/eval_event_qwen3.sh
#   CKPT=/path/to/checkpoint-xxx bash project/scripts/eval_event_qwen3.sh
#   TEST_JSONL=/path/to/test.jsonl bash project/scripts/eval_event_qwen3.sh
#   SKIP_INFER=1 bash project/scripts/eval_event_qwen3.sh   # reuse cached infer
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CKPT="${CKPT:-/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3-1.7B/event/v1/v2-20260603-172851/checkpoint-3000}"
# Held-out alpaca-style jsonl with fields {instruction,input,output,label,...}.
# Defaults to the training jsonl for convenience; override with your own test set.
export TEST_JSONL="${TEST_JSONL:-/apdcephfs_qy3/share_301069248/data/video/event_rag/merge/test_ayden_v1.jsonl}"
export OUT_FILE="${OUT_FILE:-${CKPT}/infer_event.jsonl}"
export GPU="${GPU:-0,1,2,3,4,5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export BACKEND="${BACKEND:-pt}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
# Avoid swift's "Multiple possible types found: ['qwen3', 'qwen3_emb',
# 'qwen3_reranker']" error by pinning model_type explicitly.
export MODEL_TYPE="${MODEL_TYPE:-qwen3}"
# template_type also needs explicit disambiguation
# (candidates may include qwen3 / qwen3_thinking / qwen3_nothinking /
#  deepseek_r1 / qwen3_guard / yufeng_xguard).
# `qwen3` is a mixed thinking template; with ENABLE_THINKING=false below
# this matches the training-time configuration.
export TEMPLATE="${TEMPLATE:-qwen3}"
# Match training (`--enable_thinking false`) so the model does not emit
# <think>...</think> blocks during evaluation.
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
REPORT="${REPORT:-${CKPT}/eval_event_report.json}"

# 1) inference (skip if cached and SKIP_INFER=1)
if [ "${SKIP_INFER:-0}" != "1" ]; then
    bash "${SCRIPT_DIR}/infer_event_qwen3.sh"
else
    echo "[skip-infer] using existing ${OUT_FILE}"
fi

# 2) evaluate
/data/miniconda3/envs/env-3.12.11/bin/python "${SCRIPT_DIR}/eval_event.py" \
    --infer_file "${OUT_FILE}" \
    --gt_jsonl "${TEST_JSONL}" \
    --report "${REPORT}" \
    --show_errors 5
