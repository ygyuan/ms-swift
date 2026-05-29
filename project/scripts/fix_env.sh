#!/usr/bin/env bash
# =============================================================================
# fix_env.sh
# -----------------------------------------------------------------------------
# One-shot script to fix the broken Python env where:
#   - NVIDIA driver supports up to CUDA 12.2 (driver version 12020)
#   - but `torch 2.11.0+cu130` was installed by mistake, so
#     `torch.cuda.is_available()` returns False and training crashes with
#     "Your setup doesn't support bf16/gpu".
#
# Strategy:
#   1) Uninstall the mismatched torch / torchvision / torchaudio.
#   2) Install torch 2.4.1 + cu121 wheels that the driver 12.2 can run.
#   3) Force-reinstall GPU-dependent packages (bitsandbytes, vllm, deepspeed)
#      so their native parts are rebuilt/linked against the new torch.
#   4) Print a verification report at the end.
#
# Usage:
#   bash project/scripts/fix_env.sh
#
# NOTE:
#   - Run this INSIDE the conda env you want to fix (env-3.12.11).
#   - Requires network access to https://download.pytorch.org and the
#     configured pip mirror (Tencent Cloud in your case).
#   - flash-attn is NOT reinstalled here; its wheel must match
#     torch2.4+cu121+cp312 exactly, download it manually from:
#     https://github.com/Dao-AILab/flash-attention/releases
# =============================================================================

set -Eeuo pipefail

log()  { echo -e "\033[1;32m[fix_env]\033[0m $*"; }
warn() { echo -e "\033[1;33m[fix_env][warn]\033[0m $*" >&2; }
err()  { echo -e "\033[1;31m[fix_env][err ]\033[0m $*" >&2; }

# ---- 0. Sanity check: we are inside a conda/venv env ------------------------
PY_BIN="$(command -v python || true)"
if [[ -z "${PY_BIN}" ]]; then
  err "python not found in PATH. Activate your conda env first."
  exit 1
fi
log "Using python: ${PY_BIN}"
log "Python version: $(python -V 2>&1)"

# Optional: abort if running in system python by accident
if [[ "${PY_BIN}" == "/usr/bin/python"* ]]; then
  err "You are using the system python. Activate your conda env first."
  exit 1
fi

# ---- 1. Show current torch state (best-effort, do not fail) -----------------
log "Current torch state (before fix):"
python - <<'PY' || true
try:
    import torch
    print("  torch         :", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    print("  device count  :", torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception as e:
    print("  [no working torch]:", e)
PY

# ---- 2. Uninstall mismatched torch stack ------------------------------------
log "Uninstalling existing torch / torchvision / torchaudio ..."
pip uninstall -y torch torchvision torchaudio || true

# ---- 3. Install torch 2.4.1 + cu121 -----------------------------------------
# cu121 wheels run fine on driver >= 525 (your driver supports CUDA 12.2).
TORCH_VER="2.4.1"
TV_VER="0.19.1"
TA_VER="2.4.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

log "Installing torch==${TORCH_VER} (+cu121) from ${TORCH_INDEX} ..."
pip install \
  "torch==${TORCH_VER}" \
  "torchvision==${TV_VER}" \
  "torchaudio==${TA_VER}"

# ---- 4. Force-reinstall packages that embed / link against torch ------------
# bitsandbytes: was reported "compiled without GPU support".
log "Force-reinstalling bitsandbytes ..."
pip install --force-reinstall --no-deps bitsandbytes || warn "bitsandbytes reinstall failed"

# vllm: native kernels must match torch version.
log "Force-reinstalling vllm (>=0.5.1) ..."
pip install --force-reinstall --no-deps "vllm>=0.5.1" || warn "vllm reinstall failed"

# deepspeed: JIT / prebuilt ops depend on torch.
log "Force-reinstalling deepspeed (<0.19) ..."
pip install --force-reinstall --no-deps "deepspeed<0.19" || warn "deepspeed reinstall failed"

# trl / transformers:
# Newer trl (>=0.23) hard-imports `FSDPModule` from `torch.distributed.fsdp`,
# which is only publicly exported in torch>=2.6. Since we pinned torch to 2.4.1
# (to match driver CUDA 12.2), we must downgrade trl (and the transformers
# version matched to it) accordingly.
#   - trl==0.18.2          : last stable version supporting torch 2.4 without
#                            the FSDP2 `FSDPModule` hard dependency; GRPO is
#                            fully supported.
#   - transformers==4.49.0 : compatible with trl 0.18.x and torch 2.4.
log "Downgrading trl and transformers to torch-2.4-compatible versions ..."
pip install --force-reinstall --no-deps "trl==0.18.2"          || warn "trl reinstall failed"
pip install --force-reinstall --no-deps "transformers==4.49.0" || warn "transformers reinstall failed"

# ---- 5. Verification --------------------------------------------------------
log "Verification (after fix):"
python - <<'PY'
import sys
ok = True

def line(k, v):
    print(f"  {k:<22}: {v}")

try:
    import torch
    line("torch",                torch.__version__)
    line("cuda available",       torch.cuda.is_available())
    line("cuda device count",    torch.cuda.device_count())
    if torch.cuda.is_available():
        line("cuda runtime",     torch.version.cuda)
        line("bf16 supported",   torch.cuda.is_bf16_supported())
        for i in range(torch.cuda.device_count()):
            line(f"gpu[{i}]",    torch.cuda.get_device_name(i))
    else:
        ok = False
except Exception as e:
    print("  [torch import failed]:", e)
    ok = False

try:
    import bitsandbytes as bnb  # noqa: F401
    line("bitsandbytes",         bnb.__version__)
except Exception as e:
    print("  [bitsandbytes import failed]:", e)

try:
    import deepspeed  # noqa: F401
    line("deepspeed",            deepspeed.__version__)
except Exception as e:
    print("  [deepspeed import failed]:", e)

try:
    import vllm  # noqa: F401
    line("vllm",                 vllm.__version__)
except Exception as e:
    print("  [vllm import failed]:", e)

# flash-attn is optional here (must be installed manually with a wheel
# matching torch2.4+cu121+cp312); we only print, never fail.
try:
    import flash_attn  # noqa: F401
    line("flash_attn",           flash_attn.__version__)
except Exception as e:
    print("  [flash_attn not installed]:", e)
    print("  -> download a matching wheel from "
          "https://github.com/Dao-AILab/flash-attention/releases")

sys.exit(0 if ok else 2)
PY

RC=$?
if [[ ${RC} -eq 0 ]]; then
  log "Environment fix finished successfully. You can now retry:"
  log "  bash project/scripts/train_gsm8k_qwen3.5.sh"
else
  err "Verification reported GPU still unavailable (rc=${RC})."
  err "Check NVIDIA driver version with: nvidia-smi"
  err "If driver is older than required for cu121 torch, upgrade driver first."
  exit ${RC}
fi
