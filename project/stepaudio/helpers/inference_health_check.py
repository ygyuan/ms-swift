#!/usr/bin/env python3
"""Post-inference "health check" for run_inference.sh.

Originally this logic was inlined into run_inference.sh as a
`python3 - <<'PYEOF' ... PYEOF` heredoc.  Under DDP that heredoc could
race with `swift infer`'s stdin handling and cause the tail of the
python source to leak back into bash and be executed as commands
(rc=127, `... : command not found`).  Extracting to a real .py file
avoids that class of failure entirely.

Inputs come from environment variables (kept identical to the previous
heredoc contract):

    HEALTH_INPUT         path to the inference result jsonl
    HEALTH_OUTPUT        path to write the health report json to
    HEALTH_CLASSES       comma-separated candidate classes (lowercased)
    HEALTH_SAMPLE_LIMIT  scan at most this many records (int)

The script always exits 0 - a health-check failure must never mark the
outer inference run as failed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from collections import Counter


def main() -> int:
    inp = os.environ.get("HEALTH_INPUT", "")
    out = os.environ.get("HEALTH_OUTPUT", "")
    cls_env = os.environ.get("HEALTH_CLASSES", "")
    limit_env = os.environ.get("HEALTH_SAMPLE_LIMIT", "500")

    if not inp or not out:
        print("[HEALTH][WARN] HEALTH_INPUT / HEALTH_OUTPUT not set, skip",
              file=sys.stderr)
        return 0

    cls = [c.strip().lower() for c in cls_env.split(",") if c.strip()]
    try:
        limit = int(limit_env)
    except ValueError:
        limit = 500

    resp_counter: Counter = Counter()
    first_top1: Counter = Counter()          # what token first-top1 lands on
    first_class_hit: Counter = Counter()     # first-top1 hitting a candidate class
    first_top1_logprob_sum = 0.0
    first_top1_logprob_n = 0
    class_in_top20: Counter = Counter()      # candidate class appearing in top20
    n_records = 0
    n_with_logprob = 0

    try:
        f = open(inp, "r")
    except Exception as e:
        print(f"[HEALTH][WARN] cannot open {inp}: {e}", file=sys.stderr)
        return 0

    with f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            n_records += 1
            r = str(j.get("response", "")).strip().lower()
            resp_counter[r] += 1

            lp = j.get("logprobs") or {}
            content = lp.get("content") if isinstance(lp, dict) else None
            if not content:
                continue
            first = content[0]
            if not isinstance(first, dict):
                continue
            n_with_logprob += 1
            tok = str(first.get("token", "")).strip().lower()
            first_top1[tok] += 1
            if tok in cls:
                first_class_hit[tok] += 1
            try:
                lp1 = float(first.get("logprob"))
                first_top1_logprob_sum += lp1
                first_top1_logprob_n += 1
            except Exception:
                pass
            top = first.get("top_logprobs") or []
            seen = set()
            for e in top:
                t = str(e.get("token", "")).strip().lower()
                if t in cls and t not in seen:
                    seen.add(t)
                    class_in_top20[t] += 1

    resp_class_covered = [c for c in cls if resp_counter.get(c, 0) > 0]
    mean_top1_logprob = (
        first_top1_logprob_sum / first_top1_logprob_n
        if first_top1_logprob_n
        else None
    )
    mean_top1_prob = (
        math.exp(mean_top1_logprob) if mean_top1_logprob is not None else None
    )
    class_top20_coverage = {c: class_in_top20.get(c, 0) for c in cls}
    class_top20_hits_at_least_once = sum(
        1 for c in cls if class_in_top20.get(c, 0) > 0
    )

    warnings: list = []
    # Rule 1: response distribution collapsed to a single class.
    if n_records >= 100 and len(resp_class_covered) <= 1:
        warnings.append(
            f"[COLLAPSE] response covers only {len(resp_class_covered)} class"
            f" ({resp_class_covered}), n={n_records}. Possible model collapse"
            f" or wrong weights loaded."
        )
    # Rule 2: with 4+ candidates, coverage <=2 is suspicious.
    elif n_records >= 100 and len(resp_class_covered) <= 2 and len(cls) >= 4:
        warnings.append(
            f"[LOW-DIVERSITY] response covers only "
            f"{len(resp_class_covered)}/{len(cls)} candidate classes "
            f"({resp_class_covered}), n={n_records}."
        )
    # Rule 3: mean first-token top1 probability too low.
    if (
        mean_top1_prob is not None
        and n_with_logprob >= 100
        and mean_top1_prob < 0.3
    ):
        warnings.append(
            f"[LOW-CONFIDENCE] first-token top1 mean prob "
            f"{mean_top1_prob:.3f} < 0.3. Model probably not fine-tuned for"
            f" this task or wrong weights loaded."
        )
    # Rule 4: candidate vocab rarely appears in top-20.
    if (
        n_with_logprob >= 100
        and class_top20_hits_at_least_once <= 2
        and len(cls) >= 4
    ):
        warnings.append(
            f"[UNKNOWN-VOCAB] only {class_top20_hits_at_least_once} of "
            f"{len(cls)} candidate classes ever appeared in top-20."
        )

    report = {
        "result_path": inp,
        "n_records_scanned": n_records,
        "n_with_logprob": n_with_logprob,
        "response_distribution": dict(resp_counter.most_common()),
        "response_class_covered": resp_class_covered,
        "first_token_top1_distribution": dict(first_top1.most_common(20)),
        "first_token_top1_class_hit": dict(first_class_hit),
        "first_token_top1_mean_logprob": mean_top1_logprob,
        "first_token_top1_mean_prob": mean_top1_prob,
        "candidate_class_top20_hits": class_top20_coverage,
        "candidate_class_top20_hits_at_least_once": class_top20_hits_at_least_once,
        "warnings": warnings,
    }

    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as fout:
            json.dump(report, fout, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[HEALTH][WARN] failed to write {out}: {e}", file=sys.stderr)

    print("[HEALTH] " + "-" * 66)
    print(f"[HEALTH] scanned={n_records}, with_logprob={n_with_logprob}")
    print(f"[HEALTH] response distribution: {dict(resp_counter.most_common())}")
    print(
        f"[HEALTH] response classes covered: {resp_class_covered}"
        f" ({len(resp_class_covered)}/{len(cls)})"
    )
    print(
        f"[HEALTH] first-token top1 mean logprob = {mean_top1_logprob},"
        f" mean prob = {mean_top1_prob}"
    )
    print(
        f"[HEALTH] candidate classes hitting top-20: {class_top20_coverage}"
        f" (at_least_once = {class_top20_hits_at_least_once}/{len(cls)})"
    )
    if warnings:
        print("[HEALTH] !!!! WARNINGS !!!!")
        for w in warnings:
            print("[HEALTH]   " + w)
        print("[HEALTH] Suggested checks:")
        print(
            "[HEALTH]   1) full-param ckpt silently fell back to base"
            " (compare model_signature.json)"
        )
        print(
            "[HEALTH]   2) LoRA run missing --adapters or wrong BASE_MODEL"
        )
        print(
            "[HEALTH]   3) training itself has collapsed"
            " (compare training predict.jsonl)"
        )
    else:
        print("[HEALTH] OK - no obvious anomaly")
    print(f"[HEALTH] saved to: {out}")
    print("[HEALTH] " + "-" * 66)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(0)
