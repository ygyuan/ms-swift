import json, os

FILES = [
    ('adapter',
     '/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/project/stepaudio/infer_results/result_v16-20260629-162422_checkpoint-1600_20260629_175311.jsonl'),
    ('merged_v1_a',
     '/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/project/stepaudio/infer_results/result_v16-20260629-162422_checkpoint-1600-merged_20260702_172015.jsonl'),
    ('merged_v1_b',
     '/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/project/stepaudio/infer_results/result_v16-20260629-162422_checkpoint-1600-merged_20260702_173508.jsonl'),
    ('merged_v3',
     '/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/project/stepaudio/infer_results/result_v16-20260629-162422_checkpoint-1600-merged-v3_20260702_180808.jsonl'),
]

out = []
for tag, f in FILES:
    out.append('=' * 100)
    out.append(f'[{tag}] {os.path.basename(f)}   size={os.path.getsize(f):,}B')
    with open(f) as fp:
        lines = [fp.readline() for _ in range(3)]
    for i, line in enumerate(lines):
        if not line:
            continue
        d = json.loads(line)
        if i == 0:
            out.append(f'  top-level keys: {sorted(d.keys())}')
        out.append(f'  --- row {i} ---')
        out.append(f'    labels   = {d.get("labels")!r}')
        resp = d.get('response')
        if isinstance(resp, str):
            out.append(f'    response = {resp!r}')
        else:
            out.append(f'    response = (type={type(resp).__name__}) {str(resp)[:200]}')
        lp = d.get('logprobs')
        if isinstance(lp, dict) and 'content' in lp and lp['content']:
            first_tok = lp['content'][0]
            tops = first_tok.get('top_logprobs', [])[:6]
            out.append(f'    1st_tok  = {first_tok.get("token")!r}')
            out.append(f'    top6     = {[(t["token"], round(t["logprob"],3)) for t in tops]}')
        elif isinstance(lp, list) and lp:
            out.append(f'    logprobs list len={len(lp)} first_keys={list(lp[0].keys()) if isinstance(lp[0],dict) else type(lp[0])}')
        else:
            out.append(f'    logprobs = {lp}')

out.append('=' * 100)
out.append('bulk stats:')
for tag, f in FILES:
    n_total = 0
    n_empty = 0
    n_audio_in_resp = 0
    n_ttspad = 0
    n_ttsend = 0
    resp_len_hist = {}
    first_char_hist = {}
    with open(f) as fp:
        for line in fp:
            if not line.strip():
                continue
            n_total += 1
            d = json.loads(line)
            resp = d.get('response') or ''
            if resp == '':
                n_empty += 1
            if '<audio_' in resp:
                n_audio_in_resp += 1
            if '<tts_pad>' in resp:
                n_ttspad += 1
            if '<tts_end>' in resp:
                n_ttsend += 1
            L = len(resp)
            bucket = 0 if L == 0 else (1 if L <= 8 else (2 if L <= 32 else 3))
            resp_len_hist[bucket] = resp_len_hist.get(bucket, 0) + 1
            fc = resp[:1] if resp else ''
            first_char_hist[fc] = first_char_hist.get(fc, 0) + 1
    out.append(f'[{tag:12s}] total={n_total}  empty_resp={n_empty}  contain_audio_tok={n_audio_in_resp}  tts_pad={n_ttspad}  tts_end={n_ttsend}')
    out.append(f'             len_hist(0/1-8/9-32/>32) = {resp_len_hist}')
    out.append(f'             first_char top10 = {sorted(first_char_hist.items(), key=lambda x:-x[1])[:10]}')

open('/tmp/inspect_infer.out', 'w').write('\n'.join(out))
print('done ->', '/tmp/inspect_infer.out')
