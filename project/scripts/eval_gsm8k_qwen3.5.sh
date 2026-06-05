#!/bin/bash
# One-click: run inference on GSM8K test set, then evaluate accuracy.
#
# Usage:
#   bash project/scripts/eval_gsm8k_qwen3.5.sh
#   CKPT=/path/to/checkpoint-3736 bash project/scripts/eval_gsm8k_qwen3.5.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# export CKPT="${CKPT:-/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3.5-2B/v12-20260423-120546/checkpoint-3736}"
export CKPT="${CKPT:-/apdcephfs_qy3/share_301069248/users/yougenyuan/software/github/ms-swift/output/Qwen3.5-2B/v13/v1-20260529-140829/checkpoint-1200}"
export TEST_PARQUET="${TEST_PARQUET:-/apdcephfs_qy3/share_301069248/huggingface/gsm8k/main/test-00000-of-00001.parquet}"
export OUT_FILE="${OUT_FILE:-${CKPT}/infer_gsm8k.jsonl}"
export GPU="${GPU:-0,1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
REPORT="${REPORT:-${CKPT}/eval_gsm8k_report.json}"

# 1) inference (skip if cached and SKIP_INFER=1)
if [ "${SKIP_INFER:-0}" != "1" ]; then
    bash "${SCRIPT_DIR}/infer_gsm8k_qwen3.5.sh"
else
    echo "[skip-infer] using existing ${OUT_FILE}"
fi

# 2) evaluate
/data/miniconda3/envs/env-3.12.11/bin/python "${SCRIPT_DIR}/eval_gsm8k.py" \
    --infer_file "${OUT_FILE}" \
    --gt_parquet "${TEST_PARQUET}" \
    --report "${REPORT}" \
    --show_errors 5
