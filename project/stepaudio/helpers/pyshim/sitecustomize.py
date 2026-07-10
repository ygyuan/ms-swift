"""sitecustomize for run_inference_meld.sh — force ``use_cache=False`` in
``model.generate`` to align with UltraEval-Audio's evaluation harness.

Why this file exists
--------------------
UltraEval-Audio's ``step_audio_2_mini.py`` invokes the model with
``use_cache=False`` at generate time.  Empirically we saw that switching this
flag alone can move MELD 7-way accuracy by several points on the same
checkpoint / same audio / same prompt, so to make our numbers directly
comparable to their published 55.47% baseline we need to match that setting.

swift's CLI (``swift infer``) does **not** expose a ``--use_cache`` argument,
and ``InferArguments`` has no ``generation_config`` override either.  Rather
than patching swift itself (which would affect every downstream user), we
prepend the directory containing this file to ``PYTHONPATH`` from
``run_inference_meld.sh``.  CPython auto-imports any top-level module named
``sitecustomize`` on interpreter start, so this file gets executed for both
the parent bash → python process **and** every torch.distributed.run child
worker under DDP.

Because arbitrarily forcing ``use_cache=False`` in every swift job would be
surprising, we gate the patch behind an explicit env var:

    STEPAUDIO_DISABLE_KV_CACHE=1  ->  monkey-patch is installed
    (anything else)               ->  this file is a no-op

The env var is set by ``run_inference_meld.sh`` right before it invokes
``swift infer``, so it only takes effect for that one run.

Implementation
--------------
We patch ``swift.infer_engine.utils.prepare_generation_config``; that function
is the single funnel where the transformers backend converts a
``RequestConfig`` into an HF ``GenerationConfig`` right before ``.generate``.
After swift's own logic returns, we simply overwrite
``generation_config.use_cache = False``.  This is far safer than reaching
into ``Template.generate`` or ``PreTrainedModel.generate`` themselves.
"""

from __future__ import annotations

import os
import sys


def _install_patch() -> None:
    try:
        from swift.infer_engine import utils as _u
    except Exception as e:
        # swift may not be importable in every process (e.g. some helper
        # scripts).  Silently give up — this file must never break unrelated
        # Python processes just because it happens to be on PYTHONPATH.
        print(f"[stepaudio-shim] swift not importable, skip use_cache patch: {e}",
              file=sys.stderr)
        return

    orig = getattr(_u, "prepare_generation_config", None)
    if orig is None or getattr(orig, "_stepaudio_use_cache_patched", False):
        return

    def _patched(model_generation_config, request_config, tokenizer):
        gc = orig(model_generation_config, request_config, tokenizer)
        if gc is not None:
            try:
                gc.use_cache = False
            except Exception as e:
                print(f"[stepaudio-shim] failed to set use_cache=False: {e}",
                      file=sys.stderr)
        return gc

    _patched._stepaudio_use_cache_patched = True  # type: ignore[attr-defined]
    _u.prepare_generation_config = _patched
    print("[stepaudio-shim] monkey-patched "
          "swift.infer_engine.utils.prepare_generation_config to force "
          "use_cache=False", file=sys.stderr)


if os.environ.get("STEPAUDIO_DISABLE_KV_CACHE", "") == "1":
    _install_patch()
