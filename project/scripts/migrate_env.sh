#!/bin/bash
# ============================================================================
#  Conda 环境高效迁移脚本：ceph 网络盘 -> 本地盘
# ============================================================================
#  原理：
#    1) rsync 整包拷贝 env 目录（比 conda env create 快 5~10 倍）
#    2) sed 修复所有硬编码的 env 绝对路径（shebang / conda-meta / pkg 配置）
#    3) 验证 python 和关键库能否正常 import
#
#  用法：
#    bash migrate_env.sh                        # 使用默认路径
#    bash migrate_env.sh <SRC_ENV> <DST_ENV>    # 自定义源/目标
#
#  示例：
#    bash migrate_env.sh \
#      /apdcephfs_hzlf/share_303924399/users/yominyan/miniconda/envs/llama_v2 \
#      /root/envs/llama_v2
#
#  迁移完使用方式：
#    /root/envs/llama_v2/bin/python xxx.py
#    # 或
#    source /root/envs/llama_v2/bin/activate
# ============================================================================

set -e

# ---------- 默认路径（可通过命令行覆盖） ----------
# SRC="${1:-/apdcephfs_hzlf/share_303924399/users/yominyan/miniconda/envs/llama_v2}"
SRC="${1:-/apdcephfs_qy3/share_301069248/users/yougenyuan/backup/miniconda3/envs/llama_v2}"
DST="${2:-/data/miniconda3/envs/llama_v2}"

# 并发度（rsync 按一级子目录并发）
PARALLEL="${PARALLEL:-8}"

# ---------- 打印配置 ----------
echo "============================================================"
echo "  Conda 环境迁移"
echo "============================================================"
echo "  SRC      : $SRC"
echo "  DST      : $DST"
echo "  PARALLEL : $PARALLEL"
echo "------------------------------------------------------------"

# ---------- 前置检查 ----------
if [ ! -d "$SRC" ]; then
    echo "[ERROR] 源目录不存在: $SRC"
    exit 1
fi
if [ ! -x "$SRC/bin/python" ]; then
    echo "[ERROR] 源目录下没有 bin/python，看起来不是一个 conda/venv 环境"
    exit 1
fi
if [ -e "$DST" ]; then
    echo "[WARN] 目标目录已存在: $DST"
    read -p "        是否删除并重建？[y/N] " ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        rm -rf "$DST"
    else
        echo "        保留旧目录，rsync 将进行增量同步"
    fi
fi

mkdir -p "$(dirname "$DST")"

# ---------- 检查本地盘空间 ----------
echo "[0/4] 检查目标盘可用空间 ..."
df -h "$(dirname "$DST")" | tail -1
SRC_SIZE_KB=$(du -sk --apparent-size "$SRC" 2>/dev/null | awk '{print $1}')
SRC_SIZE_GB=$(awk -v k="$SRC_SIZE_KB" 'BEGIN{printf "%.1f", k/1024/1024}')
echo "      源 env 约 ${SRC_SIZE_GB} GB"
echo ""

# ---------- Step 1: rsync 拷贝 ----------
echo "[1/4] rsync 拷贝（并发 $PARALLEL 路）..."
t0=$(date +%s)
mkdir -p "$DST"

# 一级子目录并发 rsync
# bin / lib / share / conda-meta / include / ssl ...
ls -A "$SRC" | xargs -n1 -P"$PARALLEL" -I{} \
    rsync -a --delete "$SRC/{}" "$DST/"

t1=$(date +%s)
echo "      rsync 耗时: $((t1 - t0)) s"
echo ""

# ---------- Step 2: 修复硬编码路径 ----------
echo "[2/4] 修复 shebang / conda-meta / 配置中的硬编码路径 ..."
t0=$(date +%s)

# 仅对文本文件做 sed，避免把二进制 .so / .pyc 改坏
# grep -I 自动跳过二进制；-l 只列文件名；-r 递归
FILES=$(grep -rlI --binary-files=without-match "$SRC" "$DST" 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "      未发现硬编码路径（可能已是相对路径，或 env 为空）"
else
    N=$(echo "$FILES" | wc -l)
    echo "      发现 $N 个文件包含旧路径，开始替换 ..."
    echo "$FILES" | xargs -r -n 50 -P "$PARALLEL" sed -i "s|$SRC|$DST|g"
fi

# conda-meta 里的 JSON 通常也要改（即使上面已覆盖，这里兜底）
if [ -d "$DST/conda-meta" ]; then
    find "$DST/conda-meta" -maxdepth 1 -name "*.json" -print0 \
        | xargs -0 -r -n 50 -P "$PARALLEL" sed -i "s|$SRC|$DST|g" 2>/dev/null || true
fi

t1=$(date +%s)
echo "      路径修复耗时: $((t1 - t0)) s"
echo ""

# ---------- Step 3: 验证 python ----------
echo "[3/4] 验证 Python 可用性 ..."
if [ ! -x "$DST/bin/python" ]; then
    echo "[ERROR] $DST/bin/python 不可执行"
    exit 1
fi

"$DST/bin/python" - <<PYEOF
import sys, os
print(f"  python executable : {sys.executable}")
print(f"  python version    : {sys.version.split()[0]}")
print(f"  sys.prefix        : {sys.prefix}")
expected = os.environ.get("EXPECTED_PREFIX", "")
if expected and not sys.prefix.startswith(expected):
    print(f"  [WARN] sys.prefix 没指向目标目录 {expected}")
PYEOF

echo ""

# ---------- Step 4: 验证关键库 import ----------
echo "[4/4] 验证关键库 import（缺失的会警告，不会中断）..."
"$DST/bin/python" - <<'PYEOF'
import importlib, time
pkgs = ["torch", "transformers", "vllm", "flask", "numpy"]
for p in pkgs:
    t0 = time.perf_counter()
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "?")
        dt = time.perf_counter() - t0
        print(f"  ✅ {p:14s} {v:15s}  (import {dt:.2f}s)")
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  ❌ {p:14s} FAILED ({dt:.2f}s): {type(e).__name__}: {e}")
PYEOF

echo ""
echo "============================================================"
echo "  ✅ 迁移完成"
echo "============================================================"
echo "  使用方式："
echo "    $DST/bin/python your_script.py"
echo "    # 或"
echo "    source $DST/bin/activate"
echo ""
echo "  之后把你的 benchmark 命令里的 python 换成："
echo "    $DST/bin/python"
echo "  对比一下 vLLM 导入耗时是否从 3 分钟降到 ~20 秒。"
echo "============================================================"
