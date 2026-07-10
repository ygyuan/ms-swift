"""Quick smoke test for the new StepAudioAccuracyRecall reward.

Runs offline (no swift install needed) by stubbing out `swift.rewards`.
Prints computed vs expected rewards for canonical rollouts.

Usage:
    python3 project/stepaudio/tools/_smoke_stepaudio_plugin.py
"""
import sys, os, types

def _install_swift_stub():
    mod = types.ModuleType('swift'); sys.modules['swift'] = mod
    rew = types.ModuleType('swift.rewards'); sys.modules['swift.rewards'] = rew
    class ORM: pass
    rew.ORM = ORM
    rew.orms = {}
_install_swift_stub()

os.environ['STEPAUDIO_CLASS_WEIGHTS'] = 'neutral=0.5,joy=1.5,anger=2.0,surprise=2.5,sadness=3.5,disgust=5.0,fear=5.0'

# ---------- v1 params: LAZY={neutral}, penalty=0.5 ----------
os.environ['STEPAUDIO_LAZY_LABELS'] = 'neutral'
os.environ['STEPAUDIO_LAZY_PENALTY'] = '0.5'
os.environ.pop('STEPAUDIO_LAZY_STRICT', None)

sys.path.insert(0, '/apdcephfs_qy3/share_301069248/users/yougenyuan/workspace/github/ms-swift/examples/train/grpo/plugin/stepaudio')
# 保证之前如果被 import 过，可以拿到最新 env
if 'stepaudio_plugin' in sys.modules:
    del sys.modules['stepaudio_plugin']
import stepaudio_plugin as sp

R1 = sp.StepAudioAccuracyRecall()
messages = [[{'role': 'user',
              'content': 'pick exactly one of [surprise, anger, neutral, joy, sadness, fear, disgust]. what emotion?'}]] * 6
completions_v1 = ['disgust', 'neutral', 'anger', 'neutral', 'joy', 'neutral']
labels_v1 = ['disgust', 'disgust', 'disgust', 'neutral', 'neutral', '']
r1 = R1(completions=completions_v1, label=labels_v1, messages=messages)
expected_v1 = [5.0, -0.5, 0.0, 0.5, 0.0, 0.0]
print('---- v1 config (LAZY=neutral, penalty=0.5, strict=off) ----')
print('rewards  =', r1)
print('expected =', expected_v1)
assert all(abs(a - b) < 1e-6 for a, b in zip(r1, expected_v1)), 'FAIL v1'

# ---------- v2 params: LAZY={neutral, joy}, penalty=0.8, non-strict ----------
os.environ['STEPAUDIO_LAZY_LABELS'] = 'neutral,joy'
os.environ['STEPAUDIO_LAZY_PENALTY'] = '0.8'
os.environ.pop('STEPAUDIO_LAZY_STRICT', None)
if 'stepaudio_plugin' in sys.modules:
    del sys.modules['stepaudio_plugin']
_install_swift_stub()  # reset orms dict too
import stepaudio_plugin as sp2

R2 = sp2.StepAudioAccuracyRecall()
messages2 = [[{'role': 'user',
               'content': 'pick exactly one of [surprise, anger, neutral, joy, sadness, fear, disgust]. what emotion?'}]] * 8
completions_v2 = [
    'disgust',   # 1) gt=disgust pred=disgust        -> +5.0
    'neutral',   # 2) gt=disgust pred=neutral (LAZY) -> -0.8
    'joy',       # 3) gt=disgust pred=joy    (LAZY!) -> -0.8   <-- 新增惩罚
    'anger',     # 4) gt=disgust pred=anger honest   ->  0.0
    'neutral',   # 5) gt=neutral pred=neutral        -> +0.5
    'joy',       # 6) gt=joy     pred=joy            -> +1.5
    'neutral',   # 7) gt=joy     pred=neutral (LAZY, gt also LAZY, non-strict)
                 #    -> gt in LAZY so *not* penalized -> 0.0
    'surprise',  # 8) gt=neutral pred=surprise       ->  0.0 (majority miss, no penalty)
]
labels_v2 = ['disgust', 'disgust', 'disgust', 'disgust',
             'neutral', 'joy', 'joy', 'neutral']
r2 = R2(completions=completions_v2, label=labels_v2, messages=messages2)
expected_v2 = [5.0, -0.8, -0.8, 0.0, 0.5, 1.5, 0.0, 0.0]
print()
print('---- v2 config (LAZY=neutral,joy, penalty=0.8, strict=off) ----')
print('rewards  =', r2)
print('expected =', expected_v2)
assert all(abs(a - b) < 1e-6 for a, b in zip(r2, expected_v2)), 'FAIL v2'

# ---------- v4 params: LAZY={neutral, joy}, penalty=0.8, STRICT=1 ----------
os.environ['STEPAUDIO_LAZY_LABELS'] = 'neutral,joy'
os.environ['STEPAUDIO_LAZY_PENALTY'] = '0.8'
os.environ['STEPAUDIO_LAZY_STRICT'] = '1'
if 'stepaudio_plugin' in sys.modules:
    del sys.modules['stepaudio_plugin']
_install_swift_stub()
import stepaudio_plugin as sp4

R4 = sp4.StepAudioAccuracyRecall()
# v4 关键差异: gt=neutral,pred=joy 现在也罚; gt=joy,pred=neutral 也罚
completions_v4 = [
    'disgust',   # 1) gt=disgust pred=disgust                 -> +5.0
    'neutral',   # 2) gt=disgust pred=neutral (LAZY,!=gt)     -> -0.8
    'joy',       # 3) gt=disgust pred=joy (LAZY,!=gt)         -> -0.8
    'anger',     # 4) gt=disgust pred=anger honest            ->  0.0
    'neutral',   # 5) gt=neutral pred=neutral (LAZY==gt)      -> +0.5  正确路径
    'joy',       # 6) gt=joy     pred=joy (LAZY==gt)          -> +1.5  正确路径
    'neutral',   # 7) gt=joy     pred=neutral (LAZY,!=gt)     -> -0.8  <-- v4 新罚
    'joy',       # 8) gt=neutral pred=joy (LAZY,!=gt)         -> -0.8  <-- v4 新罚
    'anger',     # 9) gt=neutral pred=anger honest (not LAZY) ->  0.0
]
labels_v4 = ['disgust', 'disgust', 'disgust', 'disgust',
             'neutral', 'joy', 'joy', 'neutral', 'neutral']
messages4 = [[{'role': 'user',
               'content': 'pick exactly one of [surprise, anger, neutral, joy, sadness, fear, disgust]. what emotion?'}]] * 9
r4 = R4(completions=completions_v4, label=labels_v4, messages=messages4)
expected_v4 = [5.0, -0.8, -0.8, 0.0, 0.5, 1.5, -0.8, -0.8, 0.0]
print()
print('---- v4 config (LAZY=neutral,joy, penalty=0.8, STRICT=1) ----')
print('rewards  =', r4)
print('expected =', expected_v4)
assert all(abs(a - b) < 1e-6 for a, b in zip(r4, expected_v4)), 'FAIL v4'

print()
print('ALL_OK')
