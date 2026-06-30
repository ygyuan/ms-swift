#!/usr/bin/env bash
# StepAudio2-mini 数据准备一键脚本：
#   CSV (原始数据) → 切片 wav → train.jsonl / val.jsonl → 音频可读性校验
#
# 环境变量（均可覆盖默认值）：
#   INPUT_CSV       原始 CSV 路径（必填）
#   OUTPUT_DIR      JSONL/切片音频输出目录 (默认 project/stepaudio/data)
#   VAL_RATIO       验证集比例 (默认 0.1，最少 1 条)
#   MAX_SAMPLES     最大样本数限制 (默认不限)
#   SEED            随机种子 (默认 42)
#   NO_SLICE        true/false，若为 true 则不切片，直接复用原始 user_audio (默认 false)
#
# 示例：
#   INPUT_CSV=/data/full_duplex/v1009.csv bash run_prepare_data.sh
#   INPUT_CSV=/path/x.csv VAL_RATIO=0.05 MAX_SAMPLES=2000 bash run_prepare_data.sh

set -e
export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_CSV=${INPUT_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/data"}
VAL_RATIO=${VAL_RATIO:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SEED=${SEED:-42}
NO_SLICE=${NO_SLICE:-false}

if [ -z "$INPUT_CSV" ]; then
    echo "[ERROR] 必须提供 INPUT_CSV 环境变量，指向原始 CSV 文件"
    echo "        例: INPUT_CSV=/path/to/data.csv bash $0"
    exit 1
fi
if [ ! -f "$INPUT_CSV" ]; then
    echo "[ERROR] CSV 文件不存在: $INPUT_CSV"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
ALL_JSONL="$OUTPUT_DIR/all.jsonl"
TRAIN_JSONL="$OUTPUT_DIR/train.jsonl"
VAL_JSONL="$OUTPUT_DIR/val.jsonl"
SLICE_DIR="$OUTPUT_DIR/sliced_audios"

# 选择 Python 解释器
CONDA_PY="${CONDA_PY:-/data/miniconda3/envs/env-3.12.11/bin/python}"
if command -v python >/dev/null 2>&1 && python -c "import torchaudio,pandas" 2>/dev/null; then
    PY=python
elif [ -x "$CONDA_PY" ]; then
    PY="$CONDA_PY"
    echo "[INFO] 自动使用: $PY"
else
    echo "[ERROR] 找不到含 torchaudio+pandas 的 Python 解释器"
    exit 1
fi

echo "[INFO] INPUT_CSV    = $INPUT_CSV"
echo "[INFO] OUTPUT_DIR   = $OUTPUT_DIR"
echo "[INFO] VAL_RATIO    = $VAL_RATIO  SEED=$SEED  MAX_SAMPLES=${MAX_SAMPLES:-(unlimited)}"
echo "[INFO] NO_SLICE     = $NO_SLICE"
echo "[INFO] PY           = $PY"

# 1) CSV → all.jsonl（一次性生成，再做 train/val 划分）
CONVERT_ARGS=(
    --input "$INPUT_CSV"
    --output "$ALL_JSONL"
    --output_audio_dir "$SLICE_DIR"
)
if [ -n "$MAX_SAMPLES" ]; then
    CONVERT_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [ "$NO_SLICE" = "true" ] || [ "$NO_SLICE" = "1" ]; then
    CONVERT_ARGS+=(--no_slice)
fi

echo "[STEP 1/3] 执行 CSV → JSONL 转换 + 音频切片..."
"$PY" "$SCRIPT_DIR/convert_to_swift_format.py" "${CONVERT_ARGS[@]}"

if [ ! -f "$ALL_JSONL" ]; then
    echo "[ERROR] 转换失败，未生成 $ALL_JSONL"
    exit 1
fi
TOTAL=$(wc -l < "$ALL_JSONL")
echo "[INFO] 共生成 $TOTAL 条样本"

# 2) 切分 train/val + 校验所有 audios 路径可读
echo "[STEP 2/3] 切分 train/val 并校验音频可读性..."
"$PY" - <<PY_EOF
import json, os, random, sys
random.seed($SEED)

all_path  = "$ALL_JSONL"
train_path= "$TRAIN_JSONL"
val_path  = "$VAL_JSONL"
val_ratio = float("$VAL_RATIO")

# 读取并过滤掉音频缺失的样本
ok, bad = [], []
with open(all_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        audios = item.get("audios") or []
        missing = [p for p in audios if not (isinstance(p, str) and os.path.exists(p))]
        if missing:
            bad.append((item, missing))
        else:
            ok.append(item)

print(f"[INFO] 有效样本: {len(ok)} / 异常样本(音频不存在): {len(bad)}")
for item, miss in bad[:5]:
    print(f"       [SKIP] missing={miss}")

if not ok:
    print("[ERROR] 没有任何有效样本，终止")
    sys.exit(1)

# 尝试用 soundfile 抽样校验（避免 soundfile 解码失败）
try:
    import soundfile as sf
    sample = random.sample(ok, min(5, len(ok)))
    for s in sample:
        for p in s["audios"]:
            with sf.SoundFile(p) as f:
                pass
    print(f"[INFO] 随机抽样 {len(sample)} 条用 soundfile 校验通过")
except Exception as e:
    print(f"[WARN] soundfile 校验异常: {e}")

random.shuffle(ok)
n_val = max(1, int(len(ok) * val_ratio)) if len(ok) > 1 else 0
val_set = ok[:n_val]
train_set = ok[n_val:]
print(f"[INFO] train={len(train_set)}  val={len(val_set)}")

with open(train_path, "w", encoding="utf-8") as f:
    for it in train_set:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
with open(val_path, "w", encoding="utf-8") as f:
    for it in val_set:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"[OK] 写入 {train_path}")
print(f"[OK] 写入 {val_path}")
PY_EOF

# 3) 简短统计
echo "[STEP 3/3] 完成。摘要："
echo "  train: $TRAIN_JSONL  ($(wc -l < "$TRAIN_JSONL") 条)"
echo "  val  : $VAL_JSONL    ($(wc -l < "$VAL_JSONL") 条)"
if [ "$NO_SLICE" != "true" ] && [ "$NO_SLICE" != "1" ]; then
    n_wav=$(find "$SLICE_DIR" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
    echo "  audio: $SLICE_DIR  ($n_wav 个 wav)"
fi
echo "[INFO] 现在可以直接运行: bash $SCRIPT_DIR/run_train_swift.sh"
