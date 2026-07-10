# -*- coding: utf-8 -*-
"""Standalone sanity test for MeldEmotionB1LossScale (does NOT need GPU).

Run:
    cd /apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift
    python external_plugins/test_meld_loss_scale.py

Verifies that on real assistant responses drawn from
    data_meld/train.r1omni_self_asr.jsonl
the loss_scale plugin:
  1. splits into >=2 segments,
  2. segment(s) covering "<transcript>...</transcript>\\n" get weight 0.0,
  3. segment(s) covering "<answer>...</answer>" get weight 1.0,
  4. concatenating segments equals the original assistant content
     (no bytes lost / duplicated).

Exit code 0 on success, non-zero + diagnostic dump on failure.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, REPO)

# Importing the plugin triggers registration into swift.loss_scale.mapping.
from external_plugins import meld_loss_scale  # noqa: F401
from swift.loss_scale import get_loss_scale


def check_split(response: str, ls) -> dict:
    """Return a diagnostic dict for one response string."""
    segments, weights = ls.get_loss_scale(response)
    joined = ''.join(segments)
    return {
        'response': response,
        'segments': segments,
        'weights': weights,
        'roundtrip_ok': joined == response,
        'has_zero': any(w == 0.0 for w in weights),
        'has_one': any(w == 1.0 for w in weights),
    }


def main():
    ls = get_loss_scale('meld_emotion_b1')
    assert ls.__class__.__name__ == 'MeldEmotionB1LossScale', ls

    # ---------- Case 1: synthetic, well-formed ----------
    resp1 = '<transcript>Oh my God, hes lost it. Hes totally lost it.</transcript>\n<answer>sadness</answer>'
    r1 = check_split(resp1, ls)
    assert r1['roundtrip_ok'], f'roundtrip failed: {r1}'
    # We expect: [transcript+"\n" with w=0, "<answer>sadness</answer>" with w=1]
    transcript_zero = [(s, w) for s, w in zip(r1['segments'], r1['weights']) if w == 0.0]
    answer_one = [(s, w) for s, w in zip(r1['segments'], r1['weights']) if w == 1.0]
    assert any('<transcript>' in s and '</transcript>' in s for s, _ in transcript_zero), r1
    assert any('<answer>sadness</answer>' in s for s, _ in answer_one), r1
    print('[PASS] Case 1 (synthetic): segments={}, weights={}'.format(r1['segments'], r1['weights']))

    # ---------- Case 2: 3 real samples from the dataset ----------
    ds_path = os.path.join(
        REPO,
        'UltraEval-Audio/project/stepaudio/data_meld/train.r1omni_self_asr.jsonl',
    )
    if not os.path.exists(ds_path):
        print(f'[SKIP] dataset not found: {ds_path}')
        return 0

    n_ok = 0
    n_bad = 0
    with open(ds_path) as f:
        for line in f:
            if n_ok >= 5:
                break
            d = json.loads(line)
            asst = next((m['content'] for m in d['messages'] if m['role'] == 'assistant'), None)
            if asst is None:
                continue
            r = check_split(asst, ls)
            if not r['roundtrip_ok']:
                print(f'[FAIL] roundtrip failed on key={d.get("key")}: {r}')
                n_bad += 1
                continue
            if not (r['has_zero'] and r['has_one']):
                print(f'[FAIL] missing zero or one weight on key={d.get("key")}: {r}')
                n_bad += 1
                continue
            n_ok += 1
            transcript_chars = sum(len(s) for s, w in zip(r['segments'], r['weights']) if w == 0.0)
            answer_chars = sum(len(s) for s, w in zip(r['segments'], r['weights']) if w == 1.0)
            print(
                f'[PASS] key={d.get("key")}  '
                f'#segments={len(r["segments"])}  transcript_chars(w=0)={transcript_chars}  '
                f'answer_chars(w=1)={answer_chars}  answer={[s for s,w in zip(r["segments"],r["weights"]) if w==1.0]}'
            )

    if n_bad:
        print(f'[FAIL] {n_bad} samples failed.')
        return 1
    print(f'[OK] all {n_ok} sampled train assistant contents split correctly.')

    # ---------- Case 3: is_binary_loss_scale ----------
    assert ls.is_binary_loss_scale is True, 'expected binary loss scale (weights are 0/1 only)'
    print('[PASS] is_binary_loss_scale = True (labels fast-path)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
