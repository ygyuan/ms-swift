#!/usr/bin/env python3
"""Write a "model signature" JSON file describing the checkpoint that
run_inference.sh is about to load.

This script was originally embedded as a bash heredoc inside
run_inference.sh (`python3 - <<'PYEOF' ... PYEOF`).  Under DDP
(NPROC_PER_NODE>1) the heredoc got mixed up with the child process's
stdin handling and its remaining lines were re-interpreted as bash
commands, which caused a mysterious `... : command not found` failure
and made the whole checkpoint sweep abort.  Splitting the logic out to
a real .py file avoids the stdin race entirely.

Inputs are read from environment variables (kept identical to the
previous heredoc contract):

    SIG_MODEL_PATH  the checkpoint dir being loaded (LoRA adapter dir
                    or a full checkpoint dir)
    SIG_BASE_MODEL  base model dir actually passed to `swift infer`
                    (equal to SIG_MODEL_PATH for full-param ckpts)
    SIG_ADAPTERS    optional adapter dir (LoRA mode); "" if not used
    SIG_JSON        output JSON path

Failures are non-fatal: any error is printed and we exit 0 so the
outer bash script keeps running.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback


def head_sha1(path: str, nbytes: int = 1024 * 1024) -> str:
    """Return sha1 of the first ``nbytes`` bytes of ``path``.

    Only the head is hashed so this stays fast even for multi-GB
    safetensors shards; the safetensors header is at the start of the
    file so it is enough to detect "wrong file loaded" style bugs.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(nbytes)
        return hashlib.sha1(data).hexdigest()
    except Exception as e:  # pragma: no cover - defensive
        return f"<err:{e}>"


def scan_dir(root: str) -> dict:
    if not root or not os.path.isdir(root):
        return {"exists": False, "path": root}

    info: dict = {"exists": True, "path": root, "files": []}

    interested = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue
        if name.endswith((".safetensors", ".bin")) or name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ):
            interested.append((name, full))

    for name, full in interested:
        try:
            st = os.stat(full)
            entry = {"name": name, "size": st.st_size, "mtime": int(st.st_mtime)}
            if name.endswith((".safetensors", ".bin")):
                entry["head_sha1"] = head_sha1(full)
            info["files"].append(entry)
        except Exception as e:
            info["files"].append({"name": name, "error": repr(e)})

    cfg_path = os.path.join(root, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            info["config_summary"] = {
                "architectures": cfg.get("architectures"),
                "model_type": cfg.get("model_type"),
                "hidden_size": cfg.get("hidden_size"),
                "transformers_version": cfg.get("transformers_version"),
                "dtype": cfg.get("dtype") or cfg.get("torch_dtype"),
            }
        except Exception as e:
            info["config_summary"] = {"error": repr(e)}

    idx_path = os.path.join(root, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, "r") as f:
                idx = json.load(f)
            info["safetensors_index"] = {
                "total_size": idx.get("metadata", {}).get("total_size"),
                "num_shards": len(set(idx.get("weight_map", {}).values())),
                "num_tensors": len(idx.get("weight_map", {})),
            }
        except Exception as e:
            info["safetensors_index"] = {"error": repr(e)}

    ada_path = os.path.join(root, "adapter_config.json")
    if os.path.isfile(ada_path):
        try:
            with open(ada_path, "r") as f:
                info["adapter_config"] = json.load(f)
        except Exception as e:
            info["adapter_config"] = {"error": repr(e)}

    return info


def main() -> int:
    model_path = os.environ.get("SIG_MODEL_PATH", "")
    base_model = os.environ.get("SIG_BASE_MODEL", "")
    adapters = os.environ.get("SIG_ADAPTERS", "")
    out = os.environ.get("SIG_JSON", "")

    if not out:
        print("[SIGNATURE][WARN] SIG_JSON not set, skip", file=sys.stderr)
        return 0

    sig = {
        "model_path": model_path,
        "base_model": base_model,
        "adapters": adapters or None,
        "model_dir": scan_dir(model_path),
    }
    if adapters and base_model and base_model != model_path:
        sig["base_model_dir"] = scan_dir(base_model)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(sig, f, ensure_ascii=False, indent=2)

    # Short summary so the log lets us eyeball "did we load the right
    # thing" without opening the JSON.
    print("[SIGNATURE]")
    print(f"  model_path = {model_path}")
    if model_path != base_model:
        print(f"  base_model = {base_model}")
    if adapters:
        print(f"  adapters   = {adapters}")
    md = sig["model_dir"]
    if md.get("exists"):
        cs = md.get("config_summary", {}) or {}
        print(
            f"  arch={cs.get('architectures')}  model_type={cs.get('model_type')}  "
            f"hidden_size={cs.get('hidden_size')}  tf_ver={cs.get('transformers_version')}"
        )
        idx = md.get("safetensors_index") or {}
        if idx:
            print(
                f"  safetensors: total_size={idx.get('total_size')}  "
                f"num_shards={idx.get('num_shards')}  num_tensors={idx.get('num_tensors')}"
            )
        for e in md.get("files", []):
            if e.get("name", "").endswith((".safetensors", ".bin")):
                print(
                    f"  {e.get('name')}: size={e.get('size')}  "
                    f"head_sha1={e.get('head_sha1')}"
                )
    print(f"[SIGNATURE] saved to: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Never fail the outer bash pipeline just because of a
        # signature bookkeeping error.
        traceback.print_exc()
        sys.exit(0)
