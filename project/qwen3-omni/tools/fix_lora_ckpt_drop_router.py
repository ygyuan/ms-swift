#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix a Qwen3-Omni-MoE LoRA checkpoint that was accidentally trained with
LoRA weights attached to `mlp.gate` (the MoE top-k router).

Newer `peft` versions refuse to inject LoRA into non-Linear "parameter
container" modules such as `Qwen3OmniMoeThinkerTextTopKRouter`, causing
`swift export --merge_lora` to fail with:

    ValueError: Target module Qwen3OmniMoeThinkerTextTopKRouter() is not
    supported. Currently, only the following modules are supported: ...

For classification / retrieval / non-router-sensitive downstream tasks,
router LoRA weights typically contribute <0.3% of total LoRA params and
can be safely dropped without meaningfully hurting task quality.

This script:
  1. Backs up `adapter_model.safetensors` and `adapter_config.json`
     (unless --no-backup is passed).
  2. Removes every tensor whose key contains `.mlp.gate.lora_` (but keeps
     `.mlp.gate_proj.lora_`, which is the SwiGLU gate Linear -- unrelated
     to the MoE router).
  3. Rewrites `target_modules` in adapter_config.json to drop the bare
     `gate` alternative from the regex (keeping `gate_proj`).
  4. Writes both files back atomically.

Run:
    python fix_lora_ckpt_drop_router.py \
        --ckpt-dir output/qwen3_omni/v1_lora/checkpoint-1000

Dry-run (recommended first):
    python fix_lora_ckpt_drop_router.py \
        --ckpt-dir output/qwen3_omni/v1_lora/checkpoint-1000 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


ROUTER_KEY_PATTERN = re.compile(r'\.mlp\.gate\.lora_[AB]\.weight$')


def is_router_key(key: str) -> bool:
    """Return True iff `key` refers to a `.mlp.gate.lora_{A,B}.weight` tensor.

    We deliberately require the exact `.mlp.gate.lora_A/B.weight` suffix so
    that `.mlp.gate_proj.lora_*` (SwiGLU gate projection Linear) is NOT
    matched -- only the bare `gate` (MoE router) is removed.
    """
    return bool(ROUTER_KEY_PATTERN.search(key))


def strip_gate_from_regex(regex: str) -> str:
    """Remove the bare `gate` alternative from a target-modules regex.

    Handles the three positions in an alternation group:
        (gate|...)   ->   (...)
        (...|gate|...) -> (...|...)
        (...|gate)   ->   (...)

    Only the standalone `gate` token is stripped; `gate_proj` is preserved
    because the pattern is anchored with `|` boundaries or `(`/`)`.
    """
    out = regex
    # Middle: |gate|
    out = re.sub(r'\|gate\|', '|', out)
    # Leading: (gate|
    out = re.sub(r'\(gate\|', '(', out)
    # Trailing: |gate)
    out = re.sub(r'\|gate\)', ')', out)
    # Only-token: (gate)  -- unlikely but handle it
    out = re.sub(r'\(gate\)', '()', out)
    return out


def load_state_dict(safetensors_path: Path):
    from safetensors.torch import load_file
    return load_file(str(safetensors_path))


def save_state_dict(state_dict, safetensors_path: Path):
    from safetensors.torch import save_file
    # atomic write via temp file
    tmp_path = safetensors_path.with_suffix(safetensors_path.suffix + '.tmp')
    save_file(state_dict, str(tmp_path))
    tmp_path.replace(safetensors_path)


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + '.bak-drop-router')
    if bak.exists():
        # keep the earliest backup untouched
        return bak
    shutil.copy2(path, bak)
    return bak


def process_checkpoint(ckpt_dir: Path, dry_run: bool, no_backup: bool) -> int:
    ckpt_dir = ckpt_dir.resolve()
    if not ckpt_dir.is_dir():
        print(f'[FATAL] ckpt-dir not found: {ckpt_dir}', file=sys.stderr)
        return 2

    st_path = ckpt_dir / 'adapter_model.safetensors'
    cfg_path = ckpt_dir / 'adapter_config.json'
    if not st_path.exists():
        print(f'[FATAL] adapter_model.safetensors not found in {ckpt_dir}',
              file=sys.stderr)
        return 2
    if not cfg_path.exists():
        print(f'[FATAL] adapter_config.json not found in {ckpt_dir}',
              file=sys.stderr)
        return 2

    print(f'[INFO] ckpt_dir : {ckpt_dir}')
    print(f'[INFO] dry_run  : {dry_run}')
    print(f'[INFO] backup   : {"disabled" if no_backup else "enabled"}')
    print()

    # ---- 1. inspect adapter_config.json --------------------------------
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    old_target_modules = cfg.get('target_modules')
    if not isinstance(old_target_modules, str):
        print(f'[WARN] adapter_config.target_modules is not a str-regex: '
              f'{type(old_target_modules).__name__}. Skipping regex rewrite.')
        new_target_modules = old_target_modules
    else:
        new_target_modules = strip_gate_from_regex(old_target_modules)
        if new_target_modules == old_target_modules:
            print('[INFO] adapter_config.target_modules already has no bare '
                  '`gate` token. Nothing to change in config.')
        else:
            print('[INFO] adapter_config.target_modules will be rewritten:')
            print(f'  before: {old_target_modules}')
            print(f'  after : {new_target_modules}')
        print()

    # ---- 2. inspect safetensors --------------------------------------
    print('[INFO] loading adapter_model.safetensors ...')
    state_dict = load_state_dict(st_path)
    total_before = len(state_dict)
    router_keys = [k for k in state_dict if is_router_key(k)]
    print(f'[INFO] total tensors             : {total_before}')
    print(f'[INFO] router (`mlp.gate`) tensors: {len(router_keys)} '
          f'(will be dropped)')
    if router_keys:
        print('[INFO] first 6 router tensors to drop:')
        for k in router_keys[:6]:
            print(f'         {k}   shape={tuple(state_dict[k].shape)}   '
                  f'dtype={state_dict[k].dtype}')
        if len(router_keys) > 6:
            print(f'         ... and {len(router_keys) - 6} more')
    print()

    # cross-check: gate_proj must NOT be in the drop list
    gp_hits = [k for k in router_keys if 'gate_proj' in k]
    if gp_hits:
        print('[FATAL] regex accidentally matched `gate_proj` keys, aborting:',
              file=sys.stderr)
        for k in gp_hits[:3]:
            print(f'         {k}', file=sys.stderr)
        return 3

    # ---- 3. dry-run stops here ---------------------------------------
    if dry_run:
        print('[DRY-RUN] no file was modified.')
        print('[DRY-RUN] rerun without --dry-run to apply the changes.')
        return 0

    # ---- 4. backup + write --------------------------------------------
    if not no_backup:
        st_bak = backup(st_path)
        cfg_bak = backup(cfg_path)
        if st_bak:
            print(f'[INFO] backup created: {st_bak.name}')
        if cfg_bak:
            print(f'[INFO] backup created: {cfg_bak.name}')

    if router_keys:
        for k in router_keys:
            del state_dict[k]
        print(f'[INFO] writing pruned adapter_model.safetensors '
              f'({len(state_dict)} tensors, was {total_before}) ...')
        save_state_dict(state_dict, st_path)
    else:
        print('[INFO] no router tensor found in safetensors; skipping rewrite.')

    if isinstance(old_target_modules, str) and new_target_modules != old_target_modules:
        cfg['target_modules'] = new_target_modules
        tmp_cfg = cfg_path.with_suffix(cfg_path.suffix + '.tmp')
        with open(tmp_cfg, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write('\n')
        tmp_cfg.replace(cfg_path)
        print('[INFO] adapter_config.json updated.')

    print()
    print('[DONE] checkpoint patched. Now retry:')
    print(f'       swift export --merge_lora true --adapters {ckpt_dir}')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Drop MoE router LoRA weights from a Qwen3-Omni LoRA '
                    'checkpoint so that `peft` can merge it.')
    parser.add_argument('--ckpt-dir', required=True, type=Path,
                        help='Path to the LoRA checkpoint directory '
                             '(contains adapter_model.safetensors + '
                             'adapter_config.json).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report what would change; do not touch '
                             'any file.')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating .bak-drop-router copies '
                             '(NOT recommended).')
    args = parser.parse_args()

    rc = process_checkpoint(
        ckpt_dir=args.ckpt_dir,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )
    sys.exit(rc)


if __name__ == '__main__':
    main()
