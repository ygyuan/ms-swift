#!/usr/bin/env bash
# gen_real_self_asr.sh
# ------------------------------------------------------------
# 对 MELD self_asr 三份数据集 (train/dev/test) 分别用 base Step-Audio-2-mini
# 跑一遍推理, 得到"真实 self-ASR" (transcript 段由 base 模型自己生成, 不再
# 使用外部/gt ASR). 这一步是 v12 训练的前置条件.
#
# 输入:
#   ${DATA_DIR}/train.r1omni_self_asr.jsonl
#   ${DATA_DIR}/dev.r1omni_self_asr.jsonl
#   ${DATA_DIR}/test.r1omni_self_asr.jsonl
#   (每条含 messages[user+assistant] / audios / label / asr_text / key)
#
# 处理:
#   1) 派生"仅 user 段"的输入子集 (base 模型自由生成完整 transcript+answer)
#   2) 对每份数据 8 卡 DDP 跑 swift infer, 保存原始 result.jsonl
#      (含 response / logprobs / audios), 供后续 build 脚本 join
#
# 环境变量:
#   DATA_DIR              [默认 ../data_meld]   三份 self_asr jsonl 所在目录
#   OUT_DIR               [默认 ../data_meld/real_self_asr]  推理产物落地目录
#   BASE_MODEL_PATH       [默认 Step-Audio-2-mini 官方 HF 目录] base 模型路径
#   NPROC_PER_NODE        [默认 8]              数据并行卡数
#   CUDA_VISIBLE_DEVICES  [默认 0,1,2,3,4,5,6,7]
#   MAX_NEW_TOKENS        [默认 256]            base 生成上限
#   REPETITION_PENALTY    [默认 1.1]            抑制 no. no. no. 死循环
#   SPLITS                [默认 "train dev test"]  要处理的子集
#   SKIP_EXISTING         [默认 1]              产物存在则跳过 (方便断点续跑)
#
# 用法示例:
#   bash gen_real_self_asr.sh
#   bash gen_real_self_asr.sh SPLITS="dev test"
#   bash gen_real_self_asr.sh CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4
# ------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 允许 KEY=VALUE 位置参数
for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
    esac
done

DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data_meld"}
OUT_DIR=${OUT_DIR:-"$DATA_DIR/real_self_asr"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/apdcephfs_qy3/share_301069248/huggingface/stepfun-ai/Step-Audio-2-mini}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
SPLITS=${SPLITS:-"train dev test"}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export CUDA_VISIBLE_DEVICES
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "[gen_real_self_asr] DATA_DIR         = $DATA_DIR"
echo "[gen_real_self_asr] OUT_DIR          = $OUT_DIR"
echo "[gen_real_self_asr] BASE_MODEL_PATH  = $BASE_MODEL_PATH"
echo "[gen_real_self_asr] NPROC_PER_NODE   = $NPROC_PER_NODE"
echo "[gen_real_self_asr] CUDA_VISIBLE     = $CUDA_VISIBLE_DEVICES"
echo "[gen_real_self_asr] MAX_NEW_TOKENS   = $MAX_NEW_TOKENS"
echo "[gen_real_self_asr] REPETITION_PENALTY=$REPETITION_PENALTY"
echo "[gen_real_self_asr] SPLITS           = $SPLITS"
echo "=========================================================="

for split in $SPLITS; do
    SRC="$DATA_DIR/${split}.r1omni_self_asr.jsonl"
    IN_USER_ONLY="$OUT_DIR/${split}.user_only.jsonl"
    OUT_RESULT="$OUT_DIR/${split}.base_pred.jsonl"

    if [ ! -f "$SRC" ]; then
        echo "[SKIP] $SRC 不存在, 跳过 $split"
        continue
    fi

    # ---------- Step A: 派生 user-only 输入 ----------
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$IN_USER_ONLY" ]; then
        echo "[$split] user-only 输入已存在, 跳过派生: $IN_USER_ONLY"
    else
        echo "[$split] 派生 user-only 输入..."
        python3 - <<PYEOF
import json
src = "$SRC"
dst = "$IN_USER_ONLY"
n = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        d = json.loads(line)
        user_msg = next((m for m in d["messages"] if m["role"] == "user"), None)
        if user_msg is None:
            continue
        # 保留 audios+key 供事后 join, 去掉 assistant / label / asr_text 等
        obj = {
            "messages": [user_msg],
            "audios":   d["audios"],
            "key":      d.get("key", ""),
        }
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n += 1
print(f"[$split] wrote {n} user-only samples -> {dst}")
PYEOF
    fi

    # ---------- Step B: 8 卡 DDP swift infer ----------
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$OUT_RESULT" ] && [ -s "$OUT_RESULT" ]; then
        n_lines=$(wc -l < "$OUT_RESULT")
        n_input=$(wc -l < "$IN_USER_ONLY")
        if [ "$n_lines" -ge "$n_input" ]; then
            echo "[$split] base 推理产物已存在 (n_lines=$n_lines >= n_input=$n_input), 跳过: $OUT_RESULT"
            continue
        else
            echo "[$split] 产物行数不足 ($n_lines < $n_input), 重新推理"
            rm -f "$OUT_RESULT"
        fi
    fi

    echo "[$split] 开始 base ASR 推理 -> $OUT_RESULT"
    MODEL_PATH="$BASE_MODEL_PATH" \
    VAL_JSONL="$IN_USER_ONLY" \
    RESULT_PATH="$OUT_RESULT" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    EVAL_BATCH_SIZE=1 \
      bash "$SCRIPT_DIR/run_inference_meld.sh" \
        --repetition_penalty "$REPETITION_PENALTY"

    echo "[$split] 完成 -> $OUT_RESULT"
done

echo ""
echo "[gen_real_self_asr] 全部完成. 产物列表:"
ls -la "$OUT_DIR"/*.base_pred.jsonl 2>/dev/null || true
