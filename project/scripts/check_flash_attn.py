"""Quick environment self-check for flash-attn related GPU stack.

Checks (best-effort, never hard-fails so it can be used in CI logs):
  * torch / CUDA / bf16 capability
  * flash_attn import + version + SM check
  * SDPA flash backend availability
  * fla / causal_conv1d (optional, only if installed)
"""

from __future__ import annotations


def _line(k: str, v: object) -> None:
    print(f"  {k:<24}: {v}")


def check_torch() -> None:
    try:
        import torch
    except Exception as e:
        print("[torch] import failed:", e)
        return

    _line("torch", torch.__version__)
    _line("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    _line("cuda runtime", torch.version.cuda)
    _line("device count", torch.cuda.device_count())
    _line("bf16 supported", torch.cuda.is_bf16_supported())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        _line(f"gpu[{i}]", f"{torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")

    # SDPA flash backend availability (PyTorch built-in fast attention).
    try:
        flash_ok = torch.backends.cuda.flash_sdp_enabled()
        _line("sdpa flash enabled", flash_ok)
    except Exception as e:  # pragma: no cover
        _line("sdpa flash enabled", f"unknown ({e})")


def check_flash_attn() -> None:
    try:
        import flash_attn
    except Exception as e:
        print("[flash_attn] import failed:", e)
        return
    _line("flash_attn", getattr(flash_attn, "__version__", "unknown"))
    # flash-attn 2.x requires sm_80+ (Ampere) for bf16 path.
    try:
        import torch

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:
                print("[flash_attn] WARN: GPU sm<80, bf16 flash path unsupported.")
    except Exception:
        pass


def check_optional() -> None:
    for name in ("fla", "causal_conv1d"):
        try:
            mod = __import__(name)
            _line(name, getattr(mod, "__version__", "unknown"))
        except Exception as e:
            print(f"[{name}] not available:", e)


if __name__ == "__main__":
    print("== torch / cuda ==")
    check_torch()
    print("== flash_attn ==")
    check_flash_attn()
    print("== optional ==")
    check_optional()