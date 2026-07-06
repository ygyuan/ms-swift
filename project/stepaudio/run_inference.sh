#!/usr/bin/env bash
# StepAudio2-mini 推理脚本（音频场景 5 分类）
# 任务: 输入一段音频，模型直接输出类别字符串，取值范围:
#       [speech, music, noise, porn, song]
# 因此本脚本默认:
#   - 关闭采样 (temperature=0, top_p=1)，使输出可复现，方便后续做 threshold 评估
#   - max_new_tokens 设小 (默认 8)，分类只需要少量 token
#   - 打开 --logprobs / --top_logprobs，让结果里带上 token 级别的概率分布，
#     供 eval_classification.py 做 threshold / precision / recall / F1 分析
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
#   VAL_JSONL     待推理的 JSONL (默认 project/stepaudio/data/val.jsonl)
#   RESULT_PATH   结果保存路径   (默认 project/stepaudio/infer_results/result_<ckpt>_<ts>.jsonl)
#   MAX_NEW_TOKENS / TEMPERATURE / TOP_P / TOP_LOGPROBS / LOGPROBS  推理超参
#   MAX_SAMPLES   只推理前 N 条 (调试用，0 表示全量)
#   NPROC_PER_NODE       数据并行进程数 (默认 1; >1 时会启用 torch.distributed.run 在多卡上
#                        DDP 切分 val_dataset, 每卡加载一份模型副本, 结果合并到 RESULT_PATH)
#   CUDA_VISIBLE_DEVICES 可用 GPU 列表 (默认根据 NPROC_PER_NODE 自动选为 0,1,...,N-1)
#   EVAL_BATCH_SIZE      每个进程的推理 batch 大小 (默认 1; 增大可提升吞吐, 但需注意:
#                        1) 显存占用线性增长; 2) batch>1 时各样本会做 padding,
#                        请先小批量与 batch=1 对比 logprobs 是否一致再放大)
#
# 用法示例:
#   MODEL_PATH=/path/to/output/v0-xxx/checkpoint-1000 bash run_inference.sh
#   MODEL_PATH=/path/to/lora_ckpt BASE_MODEL=/path/to/base bash run_inference.sh
#   MODEL_PATH=/path/to/checkpoint MAX_SAMPLES=100 bash run_inference.sh
#   # 也支持 KEY=VALUE 直接通过命令行覆盖, 例如:
#   bash run_inference.sh MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=2
#   # 显式指定 RESULT_PATH 以便外层脚本拿到固定路径再喂给 run_eval.sh:
#   bash run_inference.sh MODEL_PATH=/path/to/ckpt RESULT_PATH=/tmp/result.jsonl
#   # 4 卡 DDP 并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=4 bash run_inference.sh
#   # 指定卡 1,2,3 三张卡并行推理:
#   MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=1,2,3 bash run_inference.sh

set -e

export LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

# 允许 `bash run_inference.sh KEY=VALUE ...` 这种方式覆盖环境变量,
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
#   - 用户/上层脚本 (如 run_sweep_eval.sh) 显式给了 MASTER_PORT, 就尊重之;
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
    --stream false
)
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
SIG_MODEL_PATH="$SIG_MODEL_PATH" \
SIG_BASE_MODEL="$MODEL_FOR_INFER" \
SIG_ADAPTERS="$SIG_ADAPTERS" \
SIG_JSON="$SIG_JSON" \
python3 - <<'PYEOF' || echo "[WARN] 计算权重签名失败，继续推理"
import json, os, sys, hashlib

MODEL_PATH = os.environ.get('SIG_MODEL_PATH', '')
BASE_MODEL = os.environ.get('SIG_BASE_MODEL', '')
ADAPTERS = os.environ.get('SIG_ADAPTERS', '')
OUT = os.environ['SIG_JSON']

def head_sha1(path, nbytes=1024*1024):
    """取文件前 1MB 计算 sha1 作为快速指纹（safetensors 头部包含元数据 JSON）。"""
    try:
        with open(path, 'rb') as f:
            data = f.read(nbytes)
        return hashlib.sha1(data).hexdigest()
    except Exception as e:
        return f'<err:{e}>'

def scan_dir(root):
    if not root or not os.path.isdir(root):
        return {'exists': False, 'path': root}
    info = {'exists': True, 'path': root, 'files': []}
    # 只关心权重和关键 config
    interested = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue
        if name.endswith(('.safetensors', '.bin')) \
                or name in ('config.json', 'generation_config.json',
                            'model.safetensors.index.json', 'adapter_config.json',
                            'adapter_model.safetensors', 'adapter_model.bin'):
            interested.append((name, full))
    for name, full in interested:
        try:
            st = os.stat(full)
            entry = {'name': name, 'size': st.st_size, 'mtime': int(st.st_mtime)}
            if name.endswith(('.safetensors', '.bin')):
                entry['head_sha1'] = head_sha1(full)
            info['files'].append(entry)
        except Exception as e:
            info['files'].append({'name': name, 'error': repr(e)})

    # 抽取 config.json 关键字段
    cfg_path = os.path.join(root, 'config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            info['config_summary'] = {
                'architectures': cfg.get('architectures'),
                'model_type': cfg.get('model_type'),
                'hidden_size': cfg.get('hidden_size'),
                'transformers_version': cfg.get('transformers_version'),
                'dtype': cfg.get('dtype') or cfg.get('torch_dtype'),
            }
        except Exception as e:
            info['config_summary'] = {'error': repr(e)}

    # safetensors 分片索引
    idx_path = os.path.join(root, 'model.safetensors.index.json')
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, 'r') as f:
                idx = json.load(f)
            info['safetensors_index'] = {
                'total_size': idx.get('metadata', {}).get('total_size'),
                'num_shards': len(set(idx.get('weight_map', {}).values())),
                'num_tensors': len(idx.get('weight_map', {})),
            }
        except Exception as e:
            info['safetensors_index'] = {'error': repr(e)}

    # LoRA adapter_config
    ada_path = os.path.join(root, 'adapter_config.json')
    if os.path.isfile(ada_path):
        try:
            with open(ada_path, 'r') as f:
                info['adapter_config'] = json.load(f)
        except Exception as e:
            info['adapter_config'] = {'error': repr(e)}
    return info

sig = {
    'model_path': MODEL_PATH,
    'base_model': BASE_MODEL,
    'adapters': ADAPTERS or None,
    'model_dir': scan_dir(MODEL_PATH),
}
if ADAPTERS and BASE_MODEL and BASE_MODEL != MODEL_PATH:
    sig['base_model_dir'] = scan_dir(BASE_MODEL)

with open(OUT, 'w') as f:
    json.dump(sig, f, ensure_ascii=False, indent=2)

# 简要摘要打印，方便直接在日志中比对
print('[SIGNATURE]')
print(f'  model_path = {MODEL_PATH}')
if MODEL_PATH != BASE_MODEL:
    print(f'  base_model = {BASE_MODEL}')
if ADAPTERS:
    print(f'  adapters   = {ADAPTERS}')
md = sig['model_dir']
if md.get('exists'):
    cs = md.get('config_summary', {})
    print(f'  arch={cs.get("architectures")}  model_type={cs.get("model_type")}  hidden_size={cs.get("hidden_size")}  tf_ver={cs.get("transformers_version")}')
    idx = md.get('safetensors_index') or {}
    if idx:
        print(f'  safetensors: total_size={idx.get("total_size")}  num_shards={idx.get("num_shards")}  num_tensors={idx.get("num_tensors")}')
    for e in md.get('files', []):
        if e.get('name', '').endswith(('.safetensors', '.bin')):
            print(f'  {e.get("name")}: size={e.get("size")}  head_sha1={e.get("head_sha1")}')
print(f'[SIGNATURE] saved to: {OUT}')
PYEOF

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
    HEALTH_INPUT="$RESULT_PATH" \
    HEALTH_OUTPUT="$HEALTH_JSON" \
    HEALTH_CLASSES="${HEALTH_CLASSES:-speech,music,noise,porn,song}" \
    HEALTH_SAMPLE_LIMIT="${HEALTH_SAMPLE_LIMIT:-500}" \
    python3 - <<'PYEOF' || echo "[WARN] 健康检查脚本失败（不影响推理结果本身）"
import json, os, sys, math
from collections import Counter

INP = os.environ['HEALTH_INPUT']
OUT = os.environ['HEALTH_OUTPUT']
CLS = [c.strip().lower() for c in os.environ['HEALTH_CLASSES'].split(',') if c.strip()]
LIMIT = int(os.environ['HEALTH_SAMPLE_LIMIT'])

resp_counter = Counter()
first_top1 = Counter()      # first token top1 是哪些 token
first_class_hit = Counter() # first token top1 命中候选类的次数
first_top1_logprob_sum = 0.0
first_top1_logprob_n = 0
class_in_top20 = Counter()  # 候选类词汇在 top20 中出现的次数
n_records = 0
n_with_logprob = 0

with open(INP, 'r') as f:
    for i, line in enumerate(f):
        if i >= LIMIT:
            break
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        n_records += 1
        r = str(j.get('response', '')).strip().lower()
        resp_counter[r] += 1

        lp = j.get('logprobs') or {}
        content = lp.get('content') if isinstance(lp, dict) else None
        if not content:
            continue
        first = content[0]
        if not isinstance(first, dict):
            continue
        n_with_logprob += 1
        tok = str(first.get('token', '')).strip().lower()
        first_top1[tok] += 1
        if tok in CLS:
            first_class_hit[tok] += 1
        try:
            lp1 = float(first.get('logprob'))
            first_top1_logprob_sum += lp1
            first_top1_logprob_n += 1
        except Exception:
            pass
        top = first.get('top_logprobs') or []
        seen = set()
        for e in top:
            t = str(e.get('token', '')).strip().lower()
            if t in CLS and t not in seen:
                seen.add(t)
                class_in_top20[t] += 1

# 汇总
n_response_classes = len(set(resp_counter))
resp_class_covered = [c for c in CLS if resp_counter.get(c, 0) > 0]
mean_top1_logprob = (first_top1_logprob_sum / first_top1_logprob_n) if first_top1_logprob_n else None
mean_top1_prob = math.exp(mean_top1_logprob) if mean_top1_logprob is not None else None
class_top20_coverage = {c: class_in_top20.get(c, 0) for c in CLS}
class_top20_hits_at_least_once = sum(1 for c in CLS if class_in_top20.get(c, 0) > 0)

warnings = []
# 规则 1: 类别塌陷（response 只有 1 类）
if n_records >= 100 and len(resp_class_covered) <= 1:
    warnings.append(
        f'[COLLAPSE] response 分布只覆盖 {len(resp_class_covered)} 类（{resp_class_covered}），'
        f'样本数 {n_records}。疑似模型已塌陷（训崩）或权重未正确加载。'
    )
# 规则 2: 5 个候选类里有 <=2 类被 response 覆盖（在候选类>=4 时视为塌陷）
elif n_records >= 100 and len(resp_class_covered) <= 2 and len(CLS) >= 4:
    warnings.append(
        f'[LOW-DIVERSITY] response 分布只覆盖 {len(resp_class_covered)}/{len(CLS)} 个候选类'
        f'（{resp_class_covered}），样本数 {n_records}。疑似类别塌陷。'
    )
# 规则 3: first-token top1 平均概率过低 (<0.3)，说明模型对分类根本不确定
if mean_top1_prob is not None and n_with_logprob >= 100 and mean_top1_prob < 0.3:
    warnings.append(
        f'[LOW-CONFIDENCE] first-token top1 平均概率 {mean_top1_prob:.3f} < 0.3。'
        f'疑似模型未针对该分类任务微调，或加载了错误的权重。'
    )
# 规则 4: 5 个候选类中 <=2 类曾出现在 top-20，说明模型对候选词表根本不熟
if n_with_logprob >= 100 and class_top20_hits_at_least_once <= 2 and len(CLS) >= 4:
    warnings.append(
        f'[UNKNOWN-VOCAB] 5 个候选类中只有 {class_top20_hits_at_least_once} 类曾出现在 top-20。'
        f'疑似模型对本任务的目标词表不熟悉。'
    )

report = {
    'result_path': INP,
    'n_records_scanned': n_records,
    'n_with_logprob': n_with_logprob,
    'response_distribution': dict(resp_counter.most_common()),
    'response_class_covered': resp_class_covered,
    'first_token_top1_distribution': dict(first_top1.most_common(20)),
    'first_token_top1_class_hit': dict(first_class_hit),
    'first_token_top1_mean_logprob': mean_top1_logprob,
    'first_token_top1_mean_prob': mean_top1_prob,
    'candidate_class_top20_hits': class_top20_coverage,
    'candidate_class_top20_hits_at_least_once': class_top20_hits_at_least_once,
    'warnings': warnings,
}

with open(OUT, 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('[HEALTH] ------------------------------------------------------------------')
print(f'[HEALTH] scanned={n_records}, with_logprob={n_with_logprob}')
print(f'[HEALTH] response distribution: {dict(resp_counter.most_common())}')
print(f'[HEALTH] response classes covered: {resp_class_covered} ({len(resp_class_covered)}/{len(CLS)})')
print(f'[HEALTH] first-token top1 mean logprob = {mean_top1_logprob}, mean prob = {mean_top1_prob}')
print(f'[HEALTH] candidate classes hitting top-20: {class_top20_coverage} '
      f'(at_least_once = {class_top20_hits_at_least_once}/{len(CLS)})')
if warnings:
    print('[HEALTH] !!!! WARNINGS !!!!')
    for w in warnings:
        print('[HEALTH]   ' + w)
    print('[HEALTH] 建议先排查：')
    print('[HEALTH]   1) 是否是全参 checkpoint 但被静默 fallback 到基座（对比 model_signature.json）')
    print('[HEALTH]   2) 是否是 LoRA 但漏传 --adapters 或 BASE_MODEL 有误')
    print('[HEALTH]   3) 训练本身是否已塌陷（对比训练输出目录里的 predict.jsonl）')
else:
    print('[HEALTH] OK - 未发现明显异常')
print(f'[HEALTH] saved to: {OUT}')
print('[HEALTH] ------------------------------------------------------------------')
PYEOF

    echo "[INFO] 下一步可直接运行评估:"
    echo "       bash $SCRIPT_DIR/run_eval.sh \\"
    echo "            RESULT_PATH=$RESULT_PATH \\"
    echo "            VAL_JSONL=$VAL_JSONL"
fi
