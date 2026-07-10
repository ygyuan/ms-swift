#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build offline DPO pairs for stepaudio classification.

Pipeline (offline DPO, no rollout at train time):
  1) Take SFT-checkpoint inference results on train.balanced.jsonl (or any
     labelled jsonl) that contain: messages / audios / label / response.
  2) For every sample:
        - `chosen`   = ground-truth label (e.g. "porn")
        - `rejected` = a WRONG label (single word)
     We prefer the SFT model's actual mistake (`response != label`) as the
     rejected — that IS the "hard negative" we want DPO to push down.
     For samples that SFT already got right, we fall back to a class-conditional
     hard-negative sampled from the empirical confusion prior (see below), so
     that DPO can still keep polishing the boundary of easy classes.
  3) Class re-balancing:
        - keep ALL wrong-samples of every class (they are the DPO signal),
        - down-sample right-samples of majority classes (speech/noise) to a
          bounded per-class cap so the pair distribution isn't dominated by
          "boring" pairs that add no gradient once policy = SFT.

Output format (ms-swift compatible, dpo):
    {"messages": [...user...], "audios": [...], "rejected_response": "<label>"}
The trainer will treat messages[-1] (the SFT-target label) as `chosen` and
`rejected_response` as `rejected`, then compute the paired-DPO loss.

Typical usage:

    python project/stepaudio/tools/build_dpo_pairs.py \
        --infer-jsonl project/stepaudio/infer_results/result_<sft_ckpt>.jsonl \
        --output project/stepaudio/data/train.dpo.jsonl \
        --classes speech,music,noise,porn,song \
        --keep-right-per-class 800 \
        --min-wrong-per-class 200

Design notes
------------
* We deliberately do NOT re-run inference here. The user has already produced
  inference jsonl on the SFT ckpt (see infer_results/). That file already has
  every field we need (messages / audios / label / response).
* `messages` in the source jsonl is a 2-turn (user + assistant) list where
  assistant.content == label; ms-swift DPO datacollator wants the *chosen*
  answer to be messages[-1].content (already true) and rejected_response to be
  a short string. We just replace assistant.content with the ground-truth label
  (in case the source jsonl has been overwritten) and add rejected_response.
* We enforce chosen != rejected via a strict guard.
* Audio duration filter (see --max-audio-sec, default 45s): DPO does 4
  forwards per step (policy chosen/rejected + ref chosen/rejected), and the
  audio encoder's attention is O(L^2). Empirically the current data has
  p90 ~= 52s / p95 ~= 168s / max ~= 600s, so a small long-tail of clips
  triggers 3x-4x GPU-memory spikes on random steps and drives training to
  OOM. Dropping clips longer than 45s removes that tail while keeping ~90%
  of samples.

Path-B (data-side fix for §0 rejected=neutral pollution, 2026-07-08)
--------------------------------------------------------------------
Five rounds of offline DPO hyperparameter tuning (v5/v6/v7/v8/v9) all failed
to beat the SFT baseline (macro-F1 41.2%); the best any variant reached was
34.7%. Post-mortem in run_train_dpo_meld.sh identified the root cause as
`rejected=neutral` being far too dense in the pair pool — DPO then
indiscriminately pushes neutral logits down, but since SFT already routes
rare classes (anger/sadness/disgust) through neutral confusion, pushing
neutral down also collapses their logits. No amount of β/lr/rpo_alpha/steps
tuning can escape this coupling on the training-data side.

Path-B fixes the data. Three new flags:

  --max-rejected-neutral-ratio  (default 0.30, disabled if 0)
      Hard upper bound on the fraction of pairs whose rejected==neutral.
      Enforced by capping rejected_cap[neutral] at floor(pool_size * ratio),
      which then reroutes overflow to non-saturated classes via the existing
      _resample_rejected() path (same mechanism as --balance-token-flow).

  --upsample-hard-nonneutral-mistakes  (integer factor, default 1 = off)
      Duplicate `hard non-neutral mistakes` (SFT predicted `neutral` while
      GT was a non-neutral rare class) by this factor. These are exactly the
      pairs where DPO should raise `chosen=rare_class` and push down
      `rejected=neutral`; upsampling them concentrates gradient on the pain
      point and pulls rare-class recall out of the collapsed state.

  --drop-easy-neutral-rejected  (default 0 = off)
      Drop pairs whose rejected is neutral AND whose chosen class is
      well-served by SFT (currently: joy / surprise). These pairs carry
      the highest §0 pollution risk (they push neutral down on classes
      that don't need it).

Recommended Path-B invocation:

    python project/stepaudio/tools/build_dpo_pairs.py \
        --infer-jsonl project/stepaudio/infer_results/result_<sft>.jsonl \
        --output project/stepaudio/data_meld/train.dpo.jsonl \
        --classes surprise,anger,neutral,joy,sadness,fear,disgust \
        --keep-right-per-class 700 \
        --min-wrong-per-class 150 \
        --min-chosen-per-class 300 \
        --max-audio-sec 90 \
        --rejected-strategy mistake \
        --max-rejected-neutral-ratio 0.30 \
        --upsample-hard-nonneutral-mistakes 3

Then reuse the current v9 training config in run_train_dpo_meld.sh without
touching any hyperparameter (β=0.05, lr=2e-6, LS=0, rpo_alpha=0, steps=80).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def probe_wav_duration(path: str) -> Optional[float]:
    """Return duration in seconds of a wav file, reading only its header.
    Returns None if the file is missing or not a readable RIFF/WAV.
    Uses stdlib `wave` only (no soundfile dependency, no data load)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with contextlib.closing(wave.open(path, "rb")) as w:
            nframes = w.getnframes()
            rate = w.getframerate()
        if rate <= 0:
            return None
        return nframes / float(rate)
    except Exception:
        return None


def max_audio_duration(audios: List[str]) -> Optional[float]:
    """Max duration across a sample's audio list. Returns None only if
    every audio failed to probe (treated as 'unknown' by caller)."""
    dur_seen = False
    m = 0.0
    for a in audios or []:
        d = probe_wav_duration(a)
        if d is None:
            continue
        dur_seen = True
        if d > m:
            m = d
    return m if dur_seen else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infer-jsonl", required=True, type=Path,
                   help="SFT ckpt inference jsonl (must contain messages / "
                        "audios / label / response fields).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output DPO jsonl path.")
    p.add_argument("--classes",
                   default="speech,music,noise,porn,song",
                   help="Comma-separated class labels.")
    p.add_argument("--keep-right-per-class", type=int, default=800,
                   help="For samples SFT already got right, keep at most this "
                        "many per class (down-sample majority classes).")
    p.add_argument("--min-wrong-per-class", type=int, default=0,
                   help="If a class has fewer than this many wrong-samples, "
                        "up-sample (with replacement) to hit this floor.")
    p.add_argument("--min-chosen-per-class", type=int, default=0,
                   help="Per-class floor for the number of pairs whose "
                        "`chosen` == this class in the final DPO pool. If a "
                        "class has fewer than N pairs after the wrong+right "
                        "assembly, top up by up-sampling (with replacement) "
                        "from that class's remaining right-samples first, "
                        "then from its wrong-samples if still short. 0 "
                        "(default) = disabled, preserves legacy behaviour. "
                        "Motivation: literal `equal-per-class preference "
                        "pair sampling` from the long-tail RL literature. "
                        "Note that --balance-token-flow already implicitly "
                        "balances the pool by capping rejected count at "
                        "chosen count; this flag is the symmetric floor on "
                        "the chosen side and is opt-in.")
    p.add_argument("--rejected-strategy",
                   choices=["mistake", "confusion", "uniform"],
                   default="mistake",
                   help="How to pick the rejected label: "
                        "mistake=use SFT actual wrong prediction (fallback to "
                        "confusion when SFT is right); "
                        "confusion=sample from a per-class confusion prior "
                        "(learned from mistakes seen in this file); "
                        "uniform=sample uniformly from other classes.")
    p.add_argument("--balance-token-flow", type=int, default=1,
                   help="1=enforce token-level balance between chosen and "
                        "rejected (default). For every class c, cap the "
                        "number of pairs whose rejected==c at "
                        "min(count(chosen==c), max-rejected-per-class-cap). "
                        "This is the fix for the observed DPO collapse where "
                        "minority classes (porn/noise) got systematically "
                        "pushed down because they appeared as `rejected` far "
                        "more often than as `chosen`. Set to 0 to disable and "
                        "reproduce the legacy (broken) behaviour.")
    p.add_argument("--max-rejected-per-class", type=int, default=0,
                   help="Extra absolute cap on how many times each class can "
                        "appear as `rejected`. 0 (default) means only the "
                        "chosen-count cap from --balance-token-flow applies. "
                        "Useful if you want to be even stricter than pure "
                        "token-flow balance.")
    p.add_argument("--drop-right-samples", action="store_true",
                   help="Drop samples SFT already got right. These pairs "
                        "carry ~0 DPO gradient (policy=ref on them) but the "
                        "chosen nll_loss (rpo_alpha) still fires and biases "
                        "the model toward majority classes. Enabling this "
                        "makes the DPO run focus on hard negatives only and "
                        "is highly recommended when SFT accuracy is already "
                        "very high (>=95%%).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Only print stats, do not write output.")
    # ---- audio duration filtering (avoid OOM on long clips) ----
    p.add_argument("--max-audio-sec", type=float, default=45.0,
                   help="Drop samples whose max audio duration exceeds this "
                        "many seconds. 0 or negative = no upper limit. "
                        "Default 45s is chosen from the empirical distribution "
                        "(p90 ~= 52s, p95 ~= 168s, max ~= 600s in current data): "
                        "clipping above 45s removes the ~10%% long tail that "
                        "causes DPO OOM (audio encoder attention is O(L^2), "
                        "and DPO does 4x forwards per step).")
    p.add_argument("--min-audio-sec", type=float, default=0.3,
                   help="Drop samples shorter than this (probably silence / "
                        "corrupted).")
    p.add_argument("--keep-unreadable-audio", action="store_true",
                   help="By default samples whose audio can't be probed are "
                        "kept (assume ok). Setting this flag makes that "
                        "explicit; use --drop-unreadable-audio to instead "
                        "throw them away.")
    p.add_argument("--drop-unreadable-audio", action="store_true",
                   help="Drop samples whose audio duration can't be read.")
    # ---- Path-B (§0 rejected=neutral pollution fix, 2026-07-08) ----
    # See file docstring "Path-B" section for the full motivation.
    p.add_argument("--max-rejected-neutral-ratio", type=float, default=0.0,
                   help="Hard upper bound on the fraction of final pairs "
                        "whose rejected==neutral. 0 (default) = disabled "
                        "(preserves legacy behaviour). Recommended: 0.30. "
                        "This is Path-B's primary knob: when >0, we override "
                        "rejected_cap[neutral] = min(chosen_count[neutral], "
                        "floor(pool_size * ratio)) so the token-flow balancer "
                        "reroutes the overflow to other classes via "
                        "_resample_rejected(). Root cause fix for the "
                        "observed DPO collapse: rejected=neutral was so "
                        "dense that DPO indiscriminately pushed the neutral "
                        "logit down, which via SFT's confusion structure "
                        "also collapsed rare classes (anger/sadness/disgust) "
                        "that get routed through neutral. Only meaningful "
                        "when 'neutral' is in --classes.")
    p.add_argument("--upsample-hard-nonneutral-mistakes", type=int, default=1,
                   help="Integer replication factor for `hard non-neutral "
                        "mistakes`: samples where GT is a NON-neutral class "
                        "and SFT actually predicted `neutral`. These pairs "
                        "are the exact signal DPO needs (raise chosen=rare, "
                        "push down rejected=neutral) and are usually rare. "
                        "1 (default) = off / no duplication. Recommended: 3. "
                        "Duplicating with replacement concentrates gradient "
                        "on the pain point but risks overfitting the small "
                        "set of clips; keep the factor modest (<=5).")
    p.add_argument("--drop-easy-neutral-rejected", action="store_true",
                   help="Drop pairs whose rejected is `neutral` AND chosen "
                        "class is in {joy, surprise} (SFT already serves "
                        "these well). These pairs carry the highest §0 "
                        "pollution risk (push neutral down on classes that "
                        "don't need it). Off by default; the ratio cap "
                        "above already handles the aggregate; this is an "
                        "additional targeted trim if the ratio cap alone "
                        "isn't enough.")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def normalize_response(resp: str) -> str:
    """Take the first token / first line of the raw model response."""
    if not isinstance(resp, str):
        return ""
    r = resp.strip().splitlines()[0].strip() if resp.strip() else ""
    r = r.strip("*` '\"()[]<>.,")
    r = r.split()[0] if r else ""
    return r.lower()


def build_confusion_prior(records: List[Dict], classes: List[str]
                          ) -> Dict[str, Counter]:
    """For each gt class, count how often SFT confused it with each other
    class. Used as a prior when the sample was answered correctly and we
    still need a plausible hard-negative."""
    prior: Dict[str, Counter] = {c: Counter() for c in classes}
    for r in records:
        gt = r.get("label")
        pred = normalize_response(r.get("response", ""))
        if gt in classes and pred in classes and pred != gt:
            prior[gt][pred] += 1
    # if a gt class has no observed mistakes at all (SFT is perfect on it),
    # fall back to uniform over other classes so we still get a usable prior.
    for c in classes:
        if not prior[c]:
            for other in classes:
                if other != c:
                    prior[c][other] = 1
    return prior


def sample_from_counter(rng: random.Random, cnt: Counter) -> str:
    items, weights = zip(*cnt.items())
    return rng.choices(items, weights=weights, k=1)[0]


def pick_rejected(rec: Dict, classes: List[str],
                  prior: Dict[str, Counter], strategy: str,
                  rng: random.Random) -> str:
    gt = rec["label"]
    pred = normalize_response(rec.get("response", ""))
    if strategy == "mistake" and pred in classes and pred != gt:
        return pred
    if strategy == "uniform":
        others = [c for c in classes if c != gt]
        return rng.choice(others)
    # 'confusion' or 'mistake'-fallback
    return sample_from_counter(rng, prior[gt])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    records = load_jsonl(args.infer_jsonl)
    print(f"[INFO] loaded {len(records)} inference records from "
          f"{args.infer_jsonl}")

    # -------- audio duration filtering --------
    # DPO OOMs badly on long clips (see file header). We probe wav headers
    # (fast, no data load) and drop the long tail here at data-build time so
    # every subsequent step (class rebalance, confusion prior, etc.) sees a
    # clean pool.
    max_sec = args.max_audio_sec if args.max_audio_sec and args.max_audio_sec > 0 else None
    min_sec = args.min_audio_sec if args.min_audio_sec and args.min_audio_sec > 0 else None
    if max_sec is not None or min_sec is not None:
        # NOTE: probe_wav_duration() does 2 filesystem calls (stat + open) per
        # audio; on shared/networked storage (Ceph/NFS) this is ~5-50ms per
        # sample, and running it silently on ~10k samples looks like the
        # process has hung. Two mitigations:
        #   1) print progress every N samples so it's obvious we're working;
        #   2) parallelise across a small thread pool because the workload is
        #      purely IO-bound (wav header read, no CPU decode).
        # The user can bypass this whole stage by passing --max-audio-sec 0
        # (already supported above via the `max_sec = ... or None` guard).
        import time
        from concurrent.futures import ThreadPoolExecutor

        total = len(records)
        print(f"[INFO] probing audio durations for {total} records "
              f"(max_sec={max_sec}, min_sec={min_sec}) — this touches the "
              f"filesystem so it can be slow on network storage; pass "
              f"--max-audio-sec 0 to skip entirely.")

        def _probe_one(rec):
            audios = rec.get("audios", []) or []
            if not audios:
                return rec, None, False  # no audio -> keep, unknown dur
            d = max_audio_duration(audios)
            return rec, d, True

        n_workers = min(16, (os.cpu_count() or 4) * 2)
        kept, drop_long, drop_short, drop_unreadable = [], 0, 0, 0
        t0 = time.time()
        last_print = t0
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for rec, d, had_audio in ex.map(_probe_one, records, chunksize=64):
                done += 1
                if not had_audio:
                    kept.append(rec)
                elif d is None:
                    if args.drop_unreadable_audio:
                        drop_unreadable += 1
                    else:
                        kept.append(rec)
                elif max_sec is not None and d > max_sec:
                    drop_long += 1
                elif min_sec is not None and d < min_sec:
                    drop_short += 1
                else:
                    kept.append(rec)
                now = time.time()
                if now - last_print >= 5.0 or done == total:
                    rate = done / max(1e-6, now - t0)
                    eta = (total - done) / max(1e-6, rate)
                    print(f"[INFO]   probed {done}/{total} "
                          f"({100*done/total:.1f}%)  "
                          f"kept={len(kept)}  drop_long={drop_long}  "
                          f"drop_short={drop_short}  "
                          f"drop_unread={drop_unreadable}  "
                          f"rate={rate:.0f}/s  eta={eta:.0f}s",
                          flush=True)
                    last_print = now
        print(f"[INFO] audio duration filter: kept={len(kept)}/{len(records)}"
              f"  drop_long(>{max_sec}s)={drop_long}"
              f"  drop_short(<{min_sec}s)={drop_short}"
              f"  drop_unreadable={drop_unreadable}"
              f"  took={time.time()-t0:.1f}s")
        records = kept

    # Split into wrong / right per class.
    wrong_by_class: Dict[str, List[Dict]] = defaultdict(list)
    right_by_class: Dict[str, List[Dict]] = defaultdict(list)
    skipped = 0
    for r in records:
        gt = r.get("label")
        if gt not in classes:
            skipped += 1
            continue
        pred = normalize_response(r.get("response", ""))
        if pred == gt:
            right_by_class[gt].append(r)
        else:
            wrong_by_class[gt].append(r)
    if skipped:
        print(f"[WARN] skipped {skipped} records with unknown label")

    print("[INFO] per-class split (right / wrong / total):")
    for c in classes:
        n_r = len(right_by_class[c])
        n_w = len(wrong_by_class[c])
        print(f"    {c:<8}  right={n_r:<6d}  wrong={n_w:<5d}  total={n_r+n_w}")

    prior = build_confusion_prior(records, classes)
    print("[INFO] confusion prior (per gt class → predicted wrong class):")
    for c in classes:
        top = prior[c].most_common(4)
        print(f"    {c:<8} → " + ", ".join(f"{k}:{v}" for k, v in top))

    # ---------- assemble final pool ----------
    final_pool: List[Dict] = []

    # 1) all wrong-samples (with optional up-sample floor)
    for c in classes:
        pool = list(wrong_by_class[c])
        if args.min_wrong_per_class > 0 and pool and \
                len(pool) < args.min_wrong_per_class:
            # up-sample with replacement to hit the floor
            need = args.min_wrong_per_class - len(pool)
            pool = pool + rng.choices(pool, k=need)
            print(f"[INFO] up-sampled wrong[{c}] from "
                  f"{len(wrong_by_class[c])} → {len(pool)}")
        final_pool.extend(pool)

    # 1.5) Path-B: upsample `hard non-neutral mistakes` -- samples whose GT
    #      is a NON-neutral class AND SFT predicted `neutral`. These are the
    #      exact pairs where DPO must "raise chosen=rare, push down rejected=
    #      neutral". They are usually far rarer than the aggregate pool, and
    #      duplicating them concentrates gradient at the SFT weakness point.
    #      Only fires when 'neutral' is one of the training classes.
    if args.upsample_hard_nonneutral_mistakes > 1 and "neutral" in classes:
        factor = args.upsample_hard_nonneutral_mistakes
        # Collect: GT is a non-neutral class in `classes`, SFT actually
        # predicted `neutral` (per normalize_response).
        hard = []
        for c in classes:
            if c == "neutral":
                continue
            for r in wrong_by_class[c]:
                pred = normalize_response(r.get("response", ""))
                if pred == "neutral":
                    hard.append(r)
        if hard:
            # Duplicate `factor-1` extra times (factor=3 → 2 extra copies).
            extra = hard * (factor - 1)
            final_pool.extend(extra)
            hard_by_gt = Counter(r["label"] for r in hard)
            print(f"[INFO] Path-B upsample: hard non-neutral mistakes "
                  f"(GT!=neutral & SFT_pred=neutral) n={len(hard)}, "
                  f"replicated x{factor} → added {len(extra)} extra pairs. "
                  f"GT breakdown: " +
                  ", ".join(f"{k}:{v}" for k, v in hard_by_gt.most_common()))
        else:
            print(f"[INFO] Path-B upsample: no hard non-neutral mistakes "
                  f"found (all non-neutral classes are perfectly served by "
                  f"SFT or SFT's mistakes are non-neutral) — nothing to "
                  f"duplicate.")

    # 2) capped right-samples
    #    We also record the *leftover* right-samples that got cut by
    #    --keep-right-per-class, because --min-chosen-per-class (below) will
    #    prefer to top up from those before touching the wrong pool.
    leftover_right_by_class: Dict[str, List[Dict]] = defaultdict(list)
    for c in classes:
        if args.drop_right_samples:
            if right_by_class[c]:
                print(f"[INFO] drop-right-samples: discarded "
                      f"{len(right_by_class[c])} right[{c}] pairs "
                      f"(zero-gradient for DPO).")
            continue
        pool = right_by_class[c]
        if args.keep_right_per_class > 0 and \
                len(pool) > args.keep_right_per_class:
            kept = rng.sample(pool, args.keep_right_per_class)
            kept_ids = {id(x) for x in kept}
            leftover_right_by_class[c] = [x for x in pool
                                          if id(x) not in kept_ids]
            pool = kept
            print(f"[INFO] down-sampled right[{c}] from "
                  f"{len(right_by_class[c])} → {len(pool)} "
                  f"(leftover kept for min-chosen top-up: "
                  f"{len(leftover_right_by_class[c])})")
        final_pool.extend(pool)

    # 3) per-class chosen floor (opt-in via --min-chosen-per-class).
    #    This is the literal `equal-per-class preference pair sampling`
    #    prescription: after wrong+right assembly, ensure every class has at
    #    least N pairs where chosen == c. Top-up priority:
    #       (a) leftover right-samples that got trimmed by keep_right_per_class
    #           (real, unused data — safest);
    #       (b) up-sample with replacement from that class's current final
    #           pool contribution (right ∪ wrong) — synthetic duplication.
    #    Rationale for (a) before (b): duplicating pairs increases DPO gradient
    #    magnitude on those exact clips, which risks overfitting; using real
    #    leftover data first minimises that risk.
    if args.min_chosen_per_class > 0:
        current_counts = Counter(r["label"] for r in final_pool)
        for c in classes:
            have = current_counts.get(c, 0)
            need = args.min_chosen_per_class - have
            if need <= 0:
                continue
            added_from_leftover = 0
            added_from_dup = 0
            # (a) draw from leftover right pool (no replacement while it lasts)
            leftover = leftover_right_by_class.get(c, [])
            if leftover:
                take = min(need, len(leftover))
                picked = rng.sample(leftover, take)
                final_pool.extend(picked)
                added_from_leftover = take
                need -= take
            # (b) up-sample with replacement from this class's assembled pool
            if need > 0:
                class_pool = [r for r in final_pool if r.get("label") == c]
                if class_pool:
                    dup = rng.choices(class_pool, k=need)
                    final_pool.extend(dup)
                    added_from_dup = need
                    need = 0
            print(f"[INFO] min-chosen top-up[{c}]: had={have} "
                  f"target={args.min_chosen_per_class} "
                  f"added_leftover={added_from_leftover} "
                  f"added_dup={added_from_dup} "
                  f"still_short={max(0, need)}")

    rng.shuffle(final_pool)
    print(f"[INFO] final DPO pool size = {len(final_pool)}")

    # ---------- compute per-class rejected caps for token-flow balance ----------
    # Root cause of the observed DPO collapse: some tokens (porn/noise) appeared
    # in `rejected` far more often than in `chosen`, so DPO systematically
    # pushed their logits down until they disappeared from top-1 at inference.
    # We fix that by capping count(rejected==c) at min(count(chosen==c), user_cap).
    chosen_counts: Counter = Counter(r["label"] for r in final_pool)
    if args.balance_token_flow:
        rejected_cap: Dict[str, int] = {}
        for c in classes:
            cap = chosen_counts.get(c, 0)
            if args.max_rejected_per_class > 0:
                cap = min(cap, args.max_rejected_per_class)
            rejected_cap[c] = cap
        # Path-B primary knob: additionally clamp rejected_cap['neutral'] at
        # floor(pool_size * ratio) so `rejected==neutral` cannot exceed that
        # fraction of the final pool. Overflow reroutes via _resample_rejected
        # (the existing token-flow rerouter). This is what breaks the "DPO
        # indiscriminately pushes neutral logit down → collapses rare classes"
        # coupling that all five hyperparameter-only fixes (v5-v9) failed on.
        if args.max_rejected_neutral_ratio > 0 and "neutral" in classes:
            hard_cap = int(len(final_pool) * args.max_rejected_neutral_ratio)
            old_cap = rejected_cap.get("neutral", 0)
            rejected_cap["neutral"] = min(old_cap, hard_cap)
            print(f"[INFO] Path-B max-rejected-neutral-ratio="
                  f"{args.max_rejected_neutral_ratio}: "
                  f"pool={len(final_pool)}, hard_cap={hard_cap}, "
                  f"neutral cap {old_cap} → {rejected_cap['neutral']} "
                  f"({'clamped' if rejected_cap['neutral'] < old_cap else 'unchanged'}).")
        print("[INFO] token-flow balance active. Per-class rejected caps:")
        for c in classes:
            print(f"    {c:<8}  chosen={chosen_counts.get(c,0):<5d}  "
                  f"rejected_cap={rejected_cap[c]}")
    else:
        rejected_cap = None
        print("[WARN] --balance-token-flow=0: token flow is NOT balanced. "
              "This is the legacy behaviour and is known to collapse "
              "minority classes.")

    # ---------- write DPO jsonl ----------
    if args.dry_run:
        print("[INFO] --dry-run set, not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_bad = 0
    n_capped = 0
    used_class_counter: Counter = Counter()
    used_rejected_counter: Counter = Counter()
    used_pair_counter: Counter = Counter()  # (gt, rejected)

    def _resample_rejected(rec, exclude_set):
        """Try to pick a rejected class that is not in `exclude_set` (already
        saturated) and != gt. Falls back to uniform over the remaining
        classes; if every non-gt class is saturated, returns None so the
        caller can drop the pair."""
        gt_local = rec["label"]
        candidates = [c for c in classes
                      if c != gt_local and c not in exclude_set]
        if not candidates:
            return None
        # weight by confusion prior when possible, else uniform
        cnt = Counter({c: max(1, prior[gt_local].get(c, 0))
                       for c in candidates})
        return sample_from_counter(rng, cnt)

    with args.output.open("w", encoding="utf-8") as fout:
        for r in final_pool:
            gt = r["label"]
            rejected = pick_rejected(r, classes, prior,
                                     args.rejected_strategy, rng)
            if rejected == gt:
                n_bad += 1
                continue

            # Path-B: drop `easy neutral-rejected` pairs — chosen is a class
            # SFT already handles well (joy / surprise) and rejected==neutral.
            # These push the neutral logit down without teaching the model
            # anything the SFT baseline doesn't already know, which is
            # exactly the §0 pollution vector on rare classes.
            if args.drop_easy_neutral_rejected and rejected == "neutral" \
                    and gt in ("joy", "surprise"):
                n_bad += 1
                continue

            # --- token-flow cap: if this rejected class is already saturated,
            # resample from non-saturated classes. If ALL non-gt classes are
            # saturated, drop the pair (rare tail).
            if rejected_cap is not None:
                saturated = {c for c in classes
                             if used_rejected_counter[c] >= rejected_cap[c]}
                if rejected in saturated:
                    new_rej = _resample_rejected(r, saturated)
                    if new_rej is None:
                        n_capped += 1
                        continue
                    rejected = new_rej

            # Rebuild messages, ensuring the assistant target is the GT label
            # (source jsonl already has this, but we defensively overwrite in
            # case the file is polluted).
            src_msgs = r.get("messages") or []
            user_msgs = [m for m in src_msgs if m.get("role") != "assistant"]
            new_messages = list(user_msgs) + [
                {"role": "assistant", "content": gt}
            ]

            out_row = {
                "messages": new_messages,
                "audios": r.get("audios", []),
                "rejected_response": rejected,
            }
            # Preserve label / key for later analysis (ms-swift ignores extra
            # fields).
            if "label" in r:
                out_row["label"] = r["label"]
            if "key" in r:
                out_row["key"] = r["key"]

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_written += 1
            used_class_counter[gt] += 1
            used_rejected_counter[rejected] += 1
            used_pair_counter[(gt, rejected)] += 1

    print(f"[INFO] wrote {n_written} DPO pairs → {args.output} "
          f"(dropped {n_bad} where rejected==chosen, "
          f"{n_capped} dropped by token-flow cap)")
    print("[INFO] final token-flow balance (chosen vs rejected counts):")
    print(f"    {'class':<8}  {'chosen':>7}  {'rejected':>8}  {'delta':>6}")
    for c in classes:
        ch = used_class_counter[c]
        rj = used_rejected_counter[c]
        print(f"    {c:<8}  {ch:>7d}  {rj:>8d}  {rj-ch:>+6d}")
    # Path-B key metric: the fraction of pairs whose rejected==neutral.
    # v5..v9 evaluations show anything above ~0.35 collapses rare classes;
    # target < 0.30 (recommended --max-rejected-neutral-ratio 0.30).
    if "neutral" in classes and n_written > 0:
        neutral_rej = used_rejected_counter.get("neutral", 0)
        ratio = neutral_rej / n_written
        print(f"[INFO] Path-B check: rejected==neutral = {neutral_rej}/"
              f"{n_written} = {ratio:.1%}  "
              f"(target < 30%; anything > 35% is known to collapse rare "
              f"classes in offline DPO).")
    print("[INFO] top (chosen → rejected) pair frequencies:")
    for (gt, rej), n in used_pair_counter.most_common(15):
        print(f"    {gt:<8} → {rej:<8}  n={n}")


if __name__ == "__main__":
    main()
